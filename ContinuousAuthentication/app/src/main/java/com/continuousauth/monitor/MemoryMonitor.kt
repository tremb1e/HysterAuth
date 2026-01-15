package com.continuousauth.monitor

import android.util.Log
import kotlinx.coroutines.*
import kotlin.coroutines.coroutineContext
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Memory monitor.
 *
 * Monitors heap usage and suggests GC when memory usage exceeds configured thresholds.
 */
@Singleton
class MemoryMonitor @Inject constructor() {
    
    companion object {
        private const val DEFAULT_THRESHOLD = 0.8 // Default warning threshold: 80%
        private const val CRITICAL_THRESHOLD = 0.9 // Critical threshold: 90%
        private const val MONITORING_INTERVAL_MS = 5000L // Monitoring interval: 5s
        private const val HISTORY_SIZE = 60 // Keep ~1 minute of history (one sample per 5s)
        private const val GC_COOLDOWN_MS = 30000L // GC cooldown: 30s
        private const val TAG = "MemoryMonitor"
    }
    
    // Monitoring state
    private val isMonitoring = AtomicBoolean(false)
    private val lastGcTime = AtomicLong(0L)
    
    // History buffer
    private val memoryHistory = ConcurrentLinkedQueue<MemorySnapshot>()
    
    // Coroutine scope
    private val monitoringScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var monitoringJob: Job? = null
    
    // Configuration
    private var warningThreshold = DEFAULT_THRESHOLD
    private var criticalThreshold = CRITICAL_THRESHOLD
    
    // Callbacks
    private var onMemoryWarning: ((MemoryStatus) -> Unit)? = null
    private var onMemoryCritical: ((MemoryStatus) -> Unit)? = null
    
    /**
     * Start memory monitoring.
     */
    fun startMonitoring(
        warningThreshold: Double = DEFAULT_THRESHOLD,
        criticalThreshold: Double = CRITICAL_THRESHOLD
    ) {
        if (isMonitoring.get()) {
            Log.w(TAG, "Memory monitoring is already running")
            return
        }
        
        this.warningThreshold = warningThreshold
        this.criticalThreshold = criticalThreshold
        
        monitoringJob = monitoringScope.launch {
            performMonitoring()
        }
        
        isMonitoring.set(true)
        Log.i(
            TAG,
            "Memory monitoring started - warning: ${(warningThreshold * 100).toInt()}%, critical: ${(criticalThreshold * 100).toInt()}%"
        )
    }
    
    /**
     * Stop memory monitoring.
     */
    fun stopMonitoring() {
        if (!isMonitoring.get()) {
            return
        }
        
        monitoringJob?.cancel()
        isMonitoring.set(false)
        
        Log.i(TAG, "Memory monitoring stopped")
    }
    
    /**
     * Set memory warning callback.
     */
    fun setOnMemoryWarning(callback: (MemoryStatus) -> Unit) {
        onMemoryWarning = callback
    }
    
    /**
     * Set memory critical callback.
     */
    fun setOnMemoryCritical(callback: (MemoryStatus) -> Unit) {
        onMemoryCritical = callback
    }
    
    /**
     * Get current memory status.
     */
    fun getCurrentMemoryStatus(): MemoryStatus {
        val runtime = Runtime.getRuntime()
        val maxMemory = runtime.maxMemory()
        val totalMemory = runtime.totalMemory()
        val freeMemory = runtime.freeMemory()
        val usedMemory = totalMemory - freeMemory
        val usageRatio = usedMemory.toDouble() / maxMemory
        
        return MemoryStatus(
            maxMemoryBytes = maxMemory,
            totalMemoryBytes = totalMemory,
            usedMemoryBytes = usedMemory,
            freeMemoryBytes = freeMemory,
            usageRatio = usageRatio,
            isWarning = usageRatio >= warningThreshold,
            isCritical = usageRatio >= criticalThreshold,
            timestamp = System.currentTimeMillis()
        )
    }
    
    /**
     * Get memory history snapshots.
     */
    fun getMemoryHistory(): List<MemorySnapshot> {
        return memoryHistory.toList()
    }
    
    /**
     * Suggest a GC run (subject to cooldown).
     */
    fun suggestGC(): Boolean {
        val currentTime = System.currentTimeMillis()
        val lastGc = lastGcTime.get()
        
        if (currentTime - lastGc < GC_COOLDOWN_MS) {
            Log.d(TAG, "GC cooldown active; skipping - since last: ${currentTime - lastGc}ms")
            return false
        }
        
        Log.i(TAG, "Suggesting GC")
        System.gc()
        lastGcTime.set(currentTime)
        
        return true
    }
    
    /**
     * Get simplified memory info.
     */
    fun getMemoryInfo(): MemoryInfo {
        val status = getCurrentMemoryStatus()
        return MemoryInfo(
            usedMemoryMB = (status.usedMemoryBytes / (1024 * 1024)).toInt(),
            totalMemoryMB = (status.maxMemoryBytes / (1024 * 1024)).toInt(),
            memoryUsagePercent = (status.usageRatio * 100).toFloat()
        )
    }
    
    /**
     * Get monitoring statistics.
     */
    fun getMonitoringStats(): MonitoringStats {
        val history = memoryHistory.toList()
        
        if (history.isEmpty()) {
            return MonitoringStats(
                isActive = isMonitoring.get(),
                samplesCount = 0,
                averageUsageRatio = 0.0,
                peakUsageRatio = 0.0,
                warningCount = 0,
                criticalCount = 0
            )
        }
        
        val averageUsage = history.map { it.usageRatio }.average()
        val peakUsage = history.maxByOrNull { it.usageRatio }?.usageRatio ?: 0.0
        val warningCount = history.count { it.isWarning }
        val criticalCount = history.count { it.isCritical }
        
        return MonitoringStats(
            isActive = isMonitoring.get(),
            samplesCount = history.size,
            averageUsageRatio = averageUsage,
            peakUsageRatio = peakUsage,
            warningCount = warningCount,
            criticalCount = criticalCount
        )
    }
    
    /**
     * Monitoring loop.
     */
    private suspend fun performMonitoring() {
        while (coroutineContext.isActive) {
            try {
                val memoryStatus = getCurrentMemoryStatus()
                
                // Add to history
                addToHistory(memoryStatus)
                
                // Check thresholds and invoke callbacks
                when {
                    memoryStatus.isCritical -> {
                        Log.w(TAG, "Memory usage at critical level: ${(memoryStatus.usageRatio * 100).toInt()}%")
                        onMemoryCritical?.invoke(memoryStatus)
                        
                        // Auto-suggest GC
                        suggestGC()
                    }
                    memoryStatus.isWarning -> {
                        Log.i(TAG, "Memory usage at warning level: ${(memoryStatus.usageRatio * 100).toInt()}%")
                        onMemoryWarning?.invoke(memoryStatus)
                    }
                }
                
                delay(MONITORING_INTERVAL_MS)
                
            } catch (e: Exception) {
                Log.e(TAG, "Memory monitoring error", e)
                delay(MONITORING_INTERVAL_MS)
            }
        }
    }
    
    /**
     * Add memory snapshot to history.
     */
    private fun addToHistory(status: MemoryStatus) {
        val snapshot = MemorySnapshot(
            usageRatio = status.usageRatio,
            usedMemoryMB = status.usedMemoryBytes / (1024 * 1024),
            isWarning = status.isWarning,
            isCritical = status.isCritical,
            timestamp = status.timestamp
        )
        
        memoryHistory.offer(snapshot)
        
        // Keep history bounded
        while (memoryHistory.size > HISTORY_SIZE) {
            memoryHistory.poll()
        }
    }
    
    /**
     * Cleanup resources.
     */
    fun cleanup() {
        stopMonitoring()
        monitoringScope.cancel()
    }
}

/**
 * Memory status.
 */
data class MemoryStatus(
    val maxMemoryBytes: Long, // Max available memory
    val totalMemoryBytes: Long, // Allocated memory
    val usedMemoryBytes: Long, // Used memory
    val freeMemoryBytes: Long, // Free memory
    val usageRatio: Double, // Usage ratio
    val isWarning: Boolean, // Warning threshold reached
    val isCritical: Boolean, // Critical threshold reached
    val timestamp: Long // Timestamp
)

/**
 * Memory snapshot (for history).
 */
data class MemorySnapshot(
    val usageRatio: Double, // Usage ratio
    val usedMemoryMB: Long, // Used memory (MB)
    val isWarning: Boolean, // Warning
    val isCritical: Boolean, // Critical
    val timestamp: Long // Timestamp
)

/**
 * Monitoring statistics.
 */
data class MonitoringStats(
    val isActive: Boolean, // Monitoring active
    val samplesCount: Int, // Sample count
    val averageUsageRatio: Double, // Average usage ratio
    val peakUsageRatio: Double, // Peak usage ratio
    val warningCount: Int, // Warning events
    val criticalCount: Int // Critical events
)

/**
 * Simplified memory info.
 */
data class MemoryInfo(
    val usedMemoryMB: Int, // Used memory (MB)
    val totalMemoryMB: Int, // Total memory (MB)
    val memoryUsagePercent: Float // Memory usage (%)
)
