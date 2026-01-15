package com.continuousauth.observability

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.Process
import android.util.Log
import androidx.annotation.RequiresApi
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.io.File
import java.io.RandomAccessFile
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Performance sample snapshot.
 */
data class PerformanceSample(
    val timestamp: Long, // Sample timestamp
    val memoryUsageMB: Long, // Memory usage (MB)
    val memoryAvailableMB: Long, // Available memory (MB)
    val memoryUsagePercent: Double, // Memory usage (%)
    val cpuUsagePercent: Double, // CPU usage (%)
    val heapUsageMB: Long, // Heap usage (MB)
    val heapMaxMB: Long, // Max heap (MB)
    val nativeMemoryMB: Long // Native memory usage (MB)
)

/**
 * Performance statistics over a time window.
 */
data class PerformanceStats(
    val duration: Long, // Duration (ms)
    val sampleCount: Int, // Sample count
    
    // Memory stats
    val memoryUsageAvg: Double, // Avg memory usage (MB)
    val memoryUsagePeak: Long, // Peak memory usage (MB)
    val memoryUsagePercentAvg: Double, // Avg memory usage (%)
    val memoryUsagePercentPeak: Double, // Peak memory usage (%)
    
    // CPU stats
    val cpuUsageAvg: Double, // Avg CPU usage (%)
    val cpuUsagePeak: Double, // Peak CPU usage (%)
    
    // Heap stats
    val heapUsageAvg: Double, // Avg heap usage (MB)
    val heapUsagePeak: Long, // Peak heap usage (MB)
    val heapUtilization: Double, // Heap utilization (%)
    
    // GC stats
    val gcCount: Int, // GC count
    val gcTime: Long // GC time (ms)
)

/**
 * Ring buffer for efficiently storing a fixed number of samples.
 */
class RingBuffer<T>(private val capacity: Int) {
    private val buffer = arrayOfNulls<Any>(capacity)
    private var head = 0
    private var tail = 0
    private var size = 0
    
    /**
     * Add an item to the buffer.
     */
    @Synchronized
    fun add(item: T) {
        @Suppress("UNCHECKED_CAST")
        buffer[tail] = item as Any
        tail = (tail + 1) % capacity
        
        if (size < capacity) {
            size++
        } else {
            head = (head + 1) % capacity
        }
    }
    
    /**
     * Get all items in the buffer.
     */
    @Synchronized
    fun getAll(): List<T> {
        val result = mutableListOf<T>()
        var current = head
        
        repeat(size) {
            @Suppress("UNCHECKED_CAST")
            result.add(buffer[current] as T)
            current = (current + 1) % capacity
        }
        
        return result
    }
    
    /**
     * Get the most recent N items.
     */
    @Synchronized
    fun getRecent(count: Int): List<T> {
        val actualCount = minOf(count, size)
        val result = mutableListOf<T>()
        
        var current = (tail - actualCount + capacity) % capacity
        repeat(actualCount) {
            @Suppress("UNCHECKED_CAST")
            result.add(buffer[current] as T)
            current = (current + 1) % capacity
        }
        
        return result
    }
    
    fun size(): Int = size
    
    @Synchronized
    fun clear() {
        head = 0
        tail = 0
        size = 0
    }
}

/**
 * Performance monitor interface.
 */
interface PerformanceMonitor {
    /**
     * Start performance monitoring.
     *
     * @param intervalMs Sampling interval (ms)
     */
    suspend fun startMonitoring(intervalMs: Long = 5000L)
    
    /**
     * Stop performance monitoring.
     */
    suspend fun stopMonitoring()
    
    /**
     * Get recent performance samples.
     */
    fun getRecentSamples(count: Int = 60): List<PerformanceSample>
    
    /**
     * Get aggregated performance statistics for a time window.
     */
    fun getPerformanceStats(durationMs: Long = 60000L): PerformanceStats
    
    /**
     * Get current performance snapshot.
     */
    suspend fun getCurrentSnapshot(): PerformanceSample
    
    /**
     * Get monitoring state flow.
     */
    fun getMonitoringStateFlow(): StateFlow<Boolean>
    
    /**
     * Clear history.
     */
    fun clearHistory()
}

/**
 * Performance monitor implementation.
 *
 * Periodically collects CPU and memory usage in the background using coroutines.
 */
@Singleton
class PerformanceMonitorImpl @Inject constructor(
    @ApplicationContext private val context: Context
) : PerformanceMonitor {
    
    companion object {
        private const val TAG = "PerformanceMonitor"
        private const val BUFFER_CAPACITY = 720 // Stores ~12 minutes of data (one sample per 5s)
    }
    
    // Ring buffer storing samples
    private val performanceBuffer = RingBuffer<PerformanceSample>(BUFFER_CAPACITY)
    
    // Monitoring state
    private val _monitoringStateFlow = MutableStateFlow(false)
    override fun getMonitoringStateFlow(): StateFlow<Boolean> = _monitoringStateFlow.asStateFlow()
    
    // Coroutine scope and job
    private val monitorScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var monitoringJob: Job? = null
    
    // System services
    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    
    // CPU usage calculation state
    private var lastCpuTime = 0L
    private var lastAppCpuTime = 0L
    
    
    override suspend fun startMonitoring(intervalMs: Long) {
        if (_monitoringStateFlow.value) {
            Log.w(TAG, "Performance monitoring is already running")
            return
        }
        
        Log.i(TAG, "Starting performance monitoring; interval: ${intervalMs}ms")
        _monitoringStateFlow.value = true
        
        // Initialize CPU time baseline
        initializeCpuBaseline()
        
        monitoringJob = monitorScope.launch {
            while (isActive && _monitoringStateFlow.value) {
                try {
                    val sample = collectPerformanceSample()
                    performanceBuffer.add(sample)
                    
                    Log.v(
                        TAG,
                        "Performance sample: memory=${sample.memoryUsageMB}MB, CPU=${String.format("%.1f", sample.cpuUsagePercent)}%"
                    )
                    
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to collect performance sample", e)
                }
                
                delay(intervalMs)
            }
        }
        
        Log.i(TAG, "Performance monitoring started")
    }
    
    override suspend fun stopMonitoring() {
        if (!_monitoringStateFlow.value) {
            return
        }
        
        Log.i(TAG, "Stopping performance monitoring")
        _monitoringStateFlow.value = false
        
        monitoringJob?.cancel()
        monitoringJob = null
        
        Log.i(TAG, "Performance monitoring stopped")
    }
    
    override fun getRecentSamples(count: Int): List<PerformanceSample> {
        return performanceBuffer.getRecent(count)
    }
    
    override fun getPerformanceStats(durationMs: Long): PerformanceStats {
        val samples = performanceBuffer.getAll()
        if (samples.isEmpty()) {
            return createEmptyStats()
        }
        
        val currentTime = System.currentTimeMillis()
        val startTime = currentTime - durationMs
        
        // Filter samples within the time window
        val filteredSamples = samples.filter { it.timestamp >= startTime }
        
        if (filteredSamples.isEmpty()) {
            return createEmptyStats()
        }
        
        return calculateStats(filteredSamples, durationMs)
    }
    
    override suspend fun getCurrentSnapshot(): PerformanceSample {
        return collectPerformanceSample()
    }
    
    override fun clearHistory() {
        performanceBuffer.clear()
        Log.i(TAG, "Performance monitoring history cleared")
    }
    
    /**
     * Collect a performance sample.
     */
    private suspend fun collectPerformanceSample(): PerformanceSample = withContext(Dispatchers.IO) {
        val timestamp = System.currentTimeMillis()
        
        // Memory info
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        
        val runtime = Runtime.getRuntime()
        val heapUsed = runtime.totalMemory() - runtime.freeMemory()
        val heapMax = runtime.maxMemory()
        
        // Process memory info
        val processMemInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(processMemInfo)
        
        val memoryUsageMB = processMemInfo.totalPss / 1024L // PSS (KB -> MB)
        val memoryAvailableMB = memInfo.availMem / 1024L / 1024L
        val totalMemoryMB = memInfo.totalMem / 1024L / 1024L
        val memoryUsagePercent = (totalMemoryMB - memoryAvailableMB).toDouble() / totalMemoryMB * 100.0
        
        val heapUsageMB = heapUsed / 1024L / 1024L
        val heapMaxMB = heapMax / 1024L / 1024L
        val nativeMemoryMB = processMemInfo.nativePss / 1024L
        
        // CPU usage
        val cpuUsagePercent = calculateCpuUsage()
        
        PerformanceSample(
            timestamp = timestamp,
            memoryUsageMB = memoryUsageMB,
            memoryAvailableMB = memoryAvailableMB,
            memoryUsagePercent = memoryUsagePercent,
            cpuUsagePercent = cpuUsagePercent,
            heapUsageMB = heapUsageMB,
            heapMaxMB = heapMaxMB,
            nativeMemoryMB = nativeMemoryMB
        )
    }
    
    /**
     * Initialize CPU time baseline.
     */
    private fun initializeCpuBaseline() {
        try {
            lastCpuTime = getTotalCpuTime()
            lastAppCpuTime = getProcessCpuTime()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to initialize CPU baseline", e)
            lastCpuTime = 0L
            lastAppCpuTime = 0L
        }
    }
    
    /**
     * Calculate CPU usage (%).
     */
    private fun calculateCpuUsage(): Double {
        return try {
            val currentCpuTime = getTotalCpuTime()
            val currentAppCpuTime = getProcessCpuTime()
            
            if (lastCpuTime == 0L || lastAppCpuTime == 0L) {
                initializeCpuBaseline()
                return 0.0
            }
            
            val totalCpuDelta = currentCpuTime - lastCpuTime
            val appCpuDelta = currentAppCpuTime - lastAppCpuTime
            
            val cpuUsage = if (totalCpuDelta > 0) {
                (appCpuDelta.toDouble() / totalCpuDelta * 100.0).coerceIn(0.0, 100.0)
            } else {
                0.0
            }
            
            lastCpuTime = currentCpuTime
            lastAppCpuTime = currentAppCpuTime
            
            cpuUsage
        } catch (e: Exception) {
            Log.w(TAG, "Failed to calculate CPU usage", e)
            0.0
        }
    }
    
    /**
     * Get total CPU time.
     */
    private fun getTotalCpuTime(): Long {
        return try {
            val file = RandomAccessFile("/proc/stat", "r")
            val line = file.readLine()
            file.close()
            
            // Parse first line: cpu user nice system idle iowait irq softirq
            val tokens = line.split("\\s+".toRegex())
            var totalTime = 0L
            for (i in 1 until minOf(tokens.size, 8)) {
                totalTime += tokens[i].toLongOrNull() ?: 0L
            }
            totalTime
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get total CPU time", e)
            0L
        }
    }
    
    /**
     * Get process CPU time.
     */
    private fun getProcessCpuTime(): Long {
        return try {
            val pid = Process.myPid()
            val file = RandomAccessFile("/proc/$pid/stat", "r")
            val line = file.readLine()
            file.close()
            
            // Parse /proc/<pid>/stat: utime and stime are fields 14 and 15
            val tokens = line.split("\\s+".toRegex())
            if (tokens.size >= 15) {
                val utime = tokens[13].toLongOrNull() ?: 0L
                val stime = tokens[14].toLongOrNull() ?: 0L
                utime + stime
            } else {
                0L
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get process CPU time", e)
            0L
        }
    }
    
    /**
     * Calculate performance statistics.
     */
    private fun calculateStats(samples: List<PerformanceSample>, duration: Long): PerformanceStats {
        if (samples.isEmpty()) return createEmptyStats()
        
        // Compute statistics
        val memoryUsages = samples.map { it.memoryUsageMB.toDouble() }
        val memoryUsagePercents = samples.map { it.memoryUsagePercent }
        val cpuUsages = samples.map { it.cpuUsagePercent }
        val heapUsages = samples.map { it.heapUsageMB.toDouble() }
        
        return PerformanceStats(
            duration = duration,
            sampleCount = samples.size,
            
            memoryUsageAvg = memoryUsages.average(),
            memoryUsagePeak = samples.maxOfOrNull { it.memoryUsageMB } ?: 0L,
            memoryUsagePercentAvg = memoryUsagePercents.average(),
            memoryUsagePercentPeak = memoryUsagePercents.maxOrNull() ?: 0.0,
            
            cpuUsageAvg = cpuUsages.average(),
            cpuUsagePeak = cpuUsages.maxOrNull() ?: 0.0,
            
            heapUsageAvg = heapUsages.average(),
            heapUsagePeak = samples.maxOfOrNull { it.heapUsageMB } ?: 0L,
            heapUtilization = samples.lastOrNull()?.let { 
                if (it.heapMaxMB > 0) it.heapUsageMB.toDouble() / it.heapMaxMB * 100.0 else 0.0 
            } ?: 0.0,
            
            gcCount = 0, // reserved; not implemented yet
            gcTime = 0L // reserved; not implemented yet
        )
    }
    
    /**
     * Create empty stats.
     */
    private fun createEmptyStats(): PerformanceStats {
        return PerformanceStats(
            duration = 0L,
            sampleCount = 0,
            memoryUsageAvg = 0.0,
            memoryUsagePeak = 0L,
            memoryUsagePercentAvg = 0.0,
            memoryUsagePercentPeak = 0.0,
            cpuUsageAvg = 0.0,
            cpuUsagePeak = 0.0,
            heapUsageAvg = 0.0,
            heapUsagePeak = 0L,
            heapUtilization = 0.0,
            gcCount = 0,
            gcTime = 0L
        )
    }
    
    /**
     * Get formatted performance report.
     */
    fun getFormattedReport(durationMs: Long = 60000L): String {
        val stats = getPerformanceStats(durationMs)
        val recent = getRecentSamples(1).firstOrNull()
        
        val sb = StringBuilder()
        sb.appendLine("=== Performance Report ===")
        sb.appendLine("Window: ${durationMs / 1000}s")
        sb.appendLine("Samples: ${stats.sampleCount}")
        sb.appendLine("")
        
        sb.appendLine("Memory:")
        sb.appendLine("  Current: ${recent?.memoryUsageMB ?: 0}MB")
        sb.appendLine("  Average: ${"%.1f".format(stats.memoryUsageAvg)}MB")
        sb.appendLine("  Peak: ${stats.memoryUsagePeak}MB")
        sb.appendLine("  System usage: ${"%.1f".format(stats.memoryUsagePercentPeak)}%")
        sb.appendLine("")
        
        sb.appendLine("CPU:")
        sb.appendLine("  Current: ${"%.1f".format(recent?.cpuUsagePercent ?: 0.0)}%")
        sb.appendLine("  Average: ${"%.1f".format(stats.cpuUsageAvg)}%")
        sb.appendLine("  Peak: ${"%.1f".format(stats.cpuUsagePeak)}%")
        sb.appendLine("")
        
        sb.appendLine("Heap:")
        sb.appendLine("  Current: ${recent?.heapUsageMB ?: 0}MB")
        sb.appendLine("  Average: ${"%.1f".format(stats.heapUsageAvg)}MB")
        sb.appendLine("  Peak: ${stats.heapUsagePeak}MB")
        sb.appendLine("  Max: ${recent?.heapMaxMB ?: 0}MB")
        sb.appendLine("  Utilization: ${"%.1f".format(stats.heapUtilization)}%")
        
        return sb.toString()
    }
    
    /**
     * Cleanup resources.
     */
    fun cleanup() {
        monitorScope.launch {
            stopMonitoring()
        }
        monitorScope.cancel()
    }
}
