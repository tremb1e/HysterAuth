package com.continuousauth.observability

import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Metric types collected by the app.
 */
enum class MetricType {
    // Batch metrics
    BATCHES_PROCESSED, // Batches processed
    BATCHES_CREATED, // Batches created
    BATCHES_ENCRYPTED, // Batches encrypted
    
    // Upload metrics
    UPLOADS_SUCCESS, // Successful uploads
    UPLOADS_FAILED_NETWORK, // Upload failures due to network errors
    UPLOADS_FAILED_AUTH, // Upload failures due to auth errors
    UPLOADS_FAILED_SERVER, // Upload failures due to server errors
    UPLOADS_FAILED_TIMEOUT, // Upload failures due to timeouts
    UPLOADS_RETRY, // Upload retries
    
    // Sensor metrics
    SENSOR_SAMPLES_COLLECTED, // Sensor samples collected
    SENSOR_SAMPLES_ACCELEROMETER, // Accelerometer samples
    SENSOR_SAMPLES_GYROSCOPE, // Gyroscope samples
    SENSOR_SAMPLES_MAGNETOMETER, // Magnetometer samples
    
    // Anomaly detection metrics
    ANOMALIES_DETECTED, // Total anomalies detected
    ANOMALIES_DEVICE_UNLOCK, // Device unlock anomalies
    ANOMALIES_ACCELEROMETER_SPIKE, // Accelerometer spike anomalies
    ANOMALIES_SENSITIVE_APP, // Sensitive app entry anomalies
    
    // Transmission mode metrics
    MODE_SWITCHES_TO_FAST, // Switches to fast mode
    MODE_SWITCHES_TO_SLOW, // Switches to slow mode
    FAST_MODE_TOTAL_TIME, // Total time in fast mode (ms)
    
    // Performance metrics
    MEMORY_USAGE_PEAK, // Peak memory usage (bytes)
    MEMORY_GC_COUNT, // GC count
    CPU_USAGE_PEAK, // Peak CPU usage (%)
    
    // Network metrics
    NETWORK_BYTES_SENT, // Bytes sent
    NETWORK_BYTES_RECEIVED, // Bytes received
    NETWORK_CONNECTIONS_CREATED, // Connections created
    NETWORK_CONNECTIONS_FAILED, // Connection failures
    
    // Latency metrics
    UPLOAD_LATENCY_MS, // Upload latency (ms)
    ENCRYPTION_LATENCY_MS, // Encryption latency (ms)
    PROCESSING_LATENCY_MS // Processing latency (ms)
}

/**
 * Metric value type.
 *
 * Supports counters and timestamped values.
 */
sealed class MetricValue {
    /**
     * Counter metric.
     */
    data class Counter(val value: Long) : MetricValue()
    
    /**
     * Timestamped value metric.
     */
    data class TimedValue(val value: Double, val timestamp: Long) : MetricValue()
}

/**
 * Metrics snapshot at a point in time.
 */
data class MetricsSnapshot(
    val timestamp: Long,
    val counters: Map<MetricType, Long>,
    val timedValues: Map<MetricType, List<Pair<Double, Long>>>, // value + timestamp
    val summary: MetricsSummary
)

/**
 * Metrics summary with aggregated values.
 */
data class MetricsSummary(
    val totalBatchesProcessed: Long,
    val totalUploadsSuccess: Long,
    val totalUploadsFailed: Long,
    val totalAnomaliesDetected: Long,
    val currentUploadSuccessRate: Double, // Upload success rate (%)
    val averageUploadLatency: Double, // Average upload latency (ms)
    val peakMemoryUsage: Long, // Peak memory usage (bytes)
    val peakCpuUsage: Double // Peak CPU usage (%)
)

/**
 * Metrics collector interface.
 */
interface MetricsCollector {
    /**
     * Increment a counter metric.
     *
     * @param type Metric type
     * @param increment Increment amount (default: 1)
     */
    fun incrementCounter(type: MetricType, increment: Long = 1L)
    
    /**
     * Set a counter metric.
     *
     * @param type Metric type
     * @param value Counter value
     */
    fun setCounter(type: MetricType, value: Long)
    
    /**
     * Record a timestamped value metric.
     *
     * @param type Metric type
     * @param value Value
     * @param timestamp Timestamp (default: now)
     */
    fun recordValue(type: MetricType, value: Double, timestamp: Long = System.currentTimeMillis())
    
    /**
     * Get counter value.
     */
    fun getCounter(type: MetricType): Long
    
    /**
     * Get recent timestamped values.
     *
     * @param type Metric type
     * @param count Number of values to return (default: 1)
     */
    fun getRecentValues(type: MetricType, count: Int = 1): List<Pair<Double, Long>>
    
    /**
     * Get current metrics snapshot.
     */
    fun getSnapshot(): MetricsSnapshot
    
    /**
     * Clear all metrics.
     */
    fun clearAll()
    
    /**
     * Clear a specific metric type.
     */
    fun clear(type: MetricType)
}

/**
 * Metrics collector implementation.
 *
 * Uses ConcurrentHashMap to store counters and timestamped values.
 */
@Singleton
class MetricsCollectorImpl @Inject constructor() : MetricsCollector {
    
    companion object {
        private const val TAG = "MetricsCollector"
        private const val MAX_TIMED_VALUES_PER_METRIC = 1000 // Max timed values per metric
    }
    
    // Counter storage
    private val counters = ConcurrentHashMap<MetricType, AtomicLong>()
    
    // Timestamped value storage (per metric)
    private val timedValues = ConcurrentHashMap<MetricType, MutableList<Pair<Double, Long>>>()
    
    // Metrics collection start time
    private val startTime = System.currentTimeMillis()
    
    override fun incrementCounter(type: MetricType, increment: Long) {
        counters.computeIfAbsent(type) { AtomicLong(0) }.addAndGet(increment)
        
        Log.v(TAG, "Counter updated: $type += $increment")
    }
    
    override fun setCounter(type: MetricType, value: Long) {
        counters.computeIfAbsent(type) { AtomicLong(0) }.set(value)
        
        Log.v(TAG, "Counter set: $type = $value")
    }
    
    override fun recordValue(type: MetricType, value: Double, timestamp: Long) {
        val valuesList = timedValues.computeIfAbsent(type) { mutableListOf() }
        
        synchronized(valuesList) {
            valuesList.add(Pair(value, timestamp))
            
            // Keep list size within the limit.
            if (valuesList.size > MAX_TIMED_VALUES_PER_METRIC) {
                valuesList.removeAt(0) // Drop oldest value
            }
        }
        
        Log.v(TAG, "Timed value recorded: $type = $value @ $timestamp")
    }
    
    override fun getCounter(type: MetricType): Long {
        return counters[type]?.get() ?: 0L
    }
    
    override fun getRecentValues(type: MetricType, count: Int): List<Pair<Double, Long>> {
        val valuesList = timedValues[type] ?: return emptyList()
        
        synchronized(valuesList) {
            val size = valuesList.size
            val startIndex = maxOf(0, size - count)
            return valuesList.subList(startIndex, size).toList()
        }
    }
    
    override fun getSnapshot(): MetricsSnapshot {
        val currentTime = System.currentTimeMillis()
        
        // Snapshot counters
        val counterSnapshot = counters.mapValues { it.value.get() }
        
        // Snapshot timed values
        val timedSnapshot = mutableMapOf<MetricType, List<Pair<Double, Long>>>()
        for ((type, valuesList) in timedValues) {
            synchronized(valuesList) {
                timedSnapshot[type] = valuesList.toList()
            }
        }
        
        // Compute summary
        val summary = calculateSummary(counterSnapshot, timedSnapshot)
        
        return MetricsSnapshot(
            timestamp = currentTime,
            counters = counterSnapshot,
            timedValues = timedSnapshot,
            summary = summary
        )
    }
    
    override fun clearAll() {
        counters.clear()
        timedValues.clear()
        Log.i(TAG, "All metrics cleared")
    }
    
    override fun clear(type: MetricType) {
        counters.remove(type)
        timedValues.remove(type)
        Log.i(TAG, "Metric cleared: $type")
    }
    
    /**
     * Calculate metrics summary.
     */
    private fun calculateSummary(
        counters: Map<MetricType, Long>,
        timedValues: Map<MetricType, List<Pair<Double, Long>>>
    ): MetricsSummary {
        
        val totalBatchesProcessed = counters[MetricType.BATCHES_PROCESSED] ?: 0L
        val totalUploadsSuccess = counters[MetricType.UPLOADS_SUCCESS] ?: 0L
        val totalUploadsFailed = (counters[MetricType.UPLOADS_FAILED_NETWORK] ?: 0L) +
                (counters[MetricType.UPLOADS_FAILED_AUTH] ?: 0L) +
                (counters[MetricType.UPLOADS_FAILED_SERVER] ?: 0L) +
                (counters[MetricType.UPLOADS_FAILED_TIMEOUT] ?: 0L)
        
        val totalAnomaliesDetected = counters[MetricType.ANOMALIES_DETECTED] ?: 0L
        
        // Upload success rate
        val totalUploads = totalUploadsSuccess + totalUploadsFailed
        val uploadSuccessRate = if (totalUploads > 0) {
            totalUploadsSuccess.toDouble() / totalUploads * 100.0
        } else 0.0
        
        // Average upload latency
        val uploadLatencies = timedValues[MetricType.UPLOAD_LATENCY_MS] ?: emptyList()
        val averageUploadLatency = if (uploadLatencies.isNotEmpty()) {
            uploadLatencies.map { it.first }.average()
        } else 0.0
        
        // Peak memory usage
        val memoryUsageValues = timedValues[MetricType.MEMORY_USAGE_PEAK] ?: emptyList()
        val peakMemoryUsage = memoryUsageValues.maxOfOrNull { it.first }?.toLong() ?: 0L
        
        // Peak CPU usage
        val cpuUsageValues = timedValues[MetricType.CPU_USAGE_PEAK] ?: emptyList()
        val peakCpuUsage = cpuUsageValues.maxOfOrNull { it.first } ?: 0.0
        
        return MetricsSummary(
            totalBatchesProcessed = totalBatchesProcessed,
            totalUploadsSuccess = totalUploadsSuccess,
            totalUploadsFailed = totalUploadsFailed,
            totalAnomaliesDetected = totalAnomaliesDetected,
            currentUploadSuccessRate = uploadSuccessRate,
            averageUploadLatency = averageUploadLatency,
            peakMemoryUsage = peakMemoryUsage,
            peakCpuUsage = peakCpuUsage
        )
    }
    
    /**
     * Get uptime (ms).
     */
    fun getUptime(): Long {
        return System.currentTimeMillis() - startTime
    }
    
    /**
     * Get formatted metrics string (useful for debugging).
     */
    fun getFormattedMetrics(): String {
        val snapshot = getSnapshot()
        val sb = StringBuilder()
        
        sb.appendLine("=== Metrics Summary ===")
        sb.appendLine("Uptime: ${getUptime() / 1000}s")
        sb.appendLine("Batches processed: ${snapshot.summary.totalBatchesProcessed}")
        sb.appendLine("Uploads succeeded: ${snapshot.summary.totalUploadsSuccess}")
        sb.appendLine("Uploads failed: ${snapshot.summary.totalUploadsFailed}")
        sb.appendLine("Success rate: ${"%.1f".format(snapshot.summary.currentUploadSuccessRate)}%")
        sb.appendLine("Avg latency: ${"%.1f".format(snapshot.summary.averageUploadLatency)}ms")
        sb.appendLine("Anomalies detected: ${snapshot.summary.totalAnomaliesDetected}")
        sb.appendLine("Peak memory: ${snapshot.summary.peakMemoryUsage / 1024 / 1024}MB")
        sb.appendLine("Peak CPU: ${"%.1f".format(snapshot.summary.peakCpuUsage)}%")
        
        return sb.toString()
    }
}
