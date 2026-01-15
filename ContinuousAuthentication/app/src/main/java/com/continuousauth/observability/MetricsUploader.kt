package com.continuousauth.observability

import android.util.Log
import com.continuousauth.proto.MetricsReport
import com.continuousauth.proto.MetricsResponse
import com.continuousauth.proto.SensorDataServiceGrpc
import com.continuousauth.utils.UserIdManager
import io.grpc.ManagedChannel
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.min

/**
 * Metrics uploader.
 *
 * Periodically uploads aggregated observability metrics to the server.
 * Note: only aggregated metrics are uploaded; no sensitive raw data is included.
 *
 * This uploader is disabled by default.
 * Enable via {@link #enableMetricsReporting}.
 */
@Singleton
class MetricsUploader @Inject constructor(
    private val metricsCollector: MetricsCollectorImpl,
    private val performanceMonitor: PerformanceMonitorImpl,
    private val userIdManager: UserIdManager
) {
    
    companion object {
        private const val TAG = "MetricsUploader"
        
        // Reporting configuration
        private const val DEFAULT_REPORT_INTERVAL_MS = 60000L  // Default: once per minute
        private const val MIN_REPORT_INTERVAL_MS = 10000L      // Minimum interval: 10s
        private const val MAX_RETRY_ATTEMPTS = 3               // Max retry attempts
        private const val RETRY_DELAY_MS = 5000L               // Retry delay
    }
    
    // State
    private val _enabled = MutableStateFlow(false)
    val enabled: StateFlow<Boolean> = _enabled.asStateFlow()
    
    private var reportingJob: Job? = null
    private val uploadScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    
    // gRPC channel and client (initialized when enabled)
    private var channel: ManagedChannel? = null
    private var serviceStub: SensorDataServiceGrpc.SensorDataServiceStub? = null
    
    // Reporting stats
    private var totalReportsSent = 0L
    private var totalReportsSuccess = 0L
    private var totalReportsFailed = 0L
    private var lastReportTime = 0L
    
    /**
     * Enables metrics reporting.
     *
     * @param serverEndpoint Server endpoint
     * @param intervalMs Reporting interval (ms)
     */
    suspend fun enableMetricsReporting(
        serverEndpoint: String,
        intervalMs: Long = DEFAULT_REPORT_INTERVAL_MS
    ) {
        if (_enabled.value) {
            Log.w(TAG, "Metrics reporting is already enabled")
            return
        }
        
        val actualInterval = max(intervalMs, MIN_REPORT_INTERVAL_MS)
        
        try {
            // TODO: Initialize gRPC channel.
            // channel = ManagedChannelBuilder.forTarget(serverEndpoint)
            //     .usePlaintext() // or use TLS
            //     .build()
            // serviceStub = SensorDataServiceGrpc.newStub(channel)
            
            _enabled.value = true
            
            // Start periodic reporting.
            reportingJob = uploadScope.launch {
                while (isActive && _enabled.value) {
                    try {
                        uploadMetrics()
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to upload metrics", e)
                    }
                    delay(actualInterval)
                }
            }
            
            Log.i(TAG, "Metrics reporting enabled; interval: ${actualInterval}ms")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to enable metrics reporting", e)
            _enabled.value = false
        }
    }
    
    /**
     * Disables metrics reporting.
     */
    suspend fun disableMetricsReporting() {
        if (!_enabled.value) {
            return
        }
        
        _enabled.value = false
        reportingJob?.cancel()
        reportingJob = null
        
        // Close gRPC channel.
        channel?.shutdown()
        channel = null
        serviceStub = null
        
        Log.i(TAG, "Metrics reporting disabled")
    }
    
    /**
     * Uploads metrics to the server.
     */
    private suspend fun uploadMetrics() = withContext(Dispatchers.IO) {
        val currentTime = System.currentTimeMillis()
        val reportPeriod = if (lastReportTime > 0) {
            currentTime - lastReportTime
        } else {
            DEFAULT_REPORT_INTERVAL_MS
        }
        
        // Build metrics report.
        val report = buildMetricsReport(currentTime, reportPeriod)
        
        // Retry logic.
        var retryCount = 0
        var success = false
        
        while (retryCount < MAX_RETRY_ATTEMPTS && !success) {
            try {
                // TODO: Send to server.
                // val response = serviceStub?.reportMetrics(report)?.await()
                
                // Simulated send (currently logs only).
                Log.d(TAG, "Preparing to report metrics: " +
                    "batches=${report.batchesProcessed}, " +
                    "uploads_success=${report.uploadsSuccess}, " +
                    "uploads_failed=${report.uploadsFailed}, " +
                    "success_rate=${report.uploadSuccessRate}%")
                
                // Simulate success.
                success = true
                totalReportsSuccess++
                lastReportTime = currentTime
                
                Log.i(TAG, "Metrics uploaded successfully")
                
            } catch (e: Exception) {
                retryCount++
                totalReportsFailed++
                
                if (retryCount < MAX_RETRY_ATTEMPTS) {
                    Log.w(TAG, "Metrics upload failed; retry $retryCount/$MAX_RETRY_ATTEMPTS", e)
                    delay(RETRY_DELAY_MS)
                } else {
                    Log.e(TAG, "Metrics upload failed permanently", e)
                }
            }
        }
        
        totalReportsSent++
    }
    
    /**
     * Builds a metrics report.
     *
     * Note: includes only aggregated metrics, not sensitive raw data.
     */
    private fun buildMetricsReport(timestamp: Long, periodMs: Long): MetricsReport {
        // Fetch metric snapshots.
        val metricsSnapshot = metricsCollector.getSnapshot()
        val performanceStats = performanceMonitor.getPerformanceStats(periodMs)
        
        // Device identifier HMAC (not plaintext).
        // TODO: Replace with HMAC using a server-provisioned key.
        val deviceIdHash = userIdManager.getUserId().hashCode().toString()
        
        return MetricsReport.newBuilder().apply {
            this.deviceIdHash = deviceIdHash
            this.timestampMs = timestamp
            this.reportingPeriodMs = periodMs
            
            // Batch and upload metrics.
            batchesProcessed = metricsSnapshot.counters[MetricType.BATCHES_PROCESSED] ?: 0
            uploadsSuccess = metricsSnapshot.counters[MetricType.UPLOADS_SUCCESS] ?: 0
            uploadsFailed = (metricsSnapshot.counters[MetricType.UPLOADS_FAILED_NETWORK] ?: 0) +
                    (metricsSnapshot.counters[MetricType.UPLOADS_FAILED_AUTH] ?: 0) +
                    (metricsSnapshot.counters[MetricType.UPLOADS_FAILED_SERVER] ?: 0) +
                    (metricsSnapshot.counters[MetricType.UPLOADS_FAILED_TIMEOUT] ?: 0)
            
            // Sensor and anomaly metrics.
            sensorSamplesCollected = metricsSnapshot.counters[MetricType.SENSOR_SAMPLES_COLLECTED] ?: 0
            anomaliesDetected = metricsSnapshot.counters[MetricType.ANOMALIES_DETECTED] ?: 0
            
            // Performance metrics.
            avgUploadLatencyMs = metricsSnapshot.summary.averageUploadLatency
            avgCpuUsagePercent = performanceStats.cpuUsageAvg
            peakMemoryUsageMb = performanceStats.memoryUsagePeak
            
            // Transmission stats (no mode switching needed anymore).
            // modeSwitchesToFast = 0
            // modeSwitchesToSlow = 0
            // fastModeTotalTimeMs = 0
            
            // Network stats (fields not available in proto; keep disabled).
            // networkBytesSent = metricsSnapshot.counters[MetricType.NETWORK_BYTES_SENT] ?: 0
            // networkBytesReceived = metricsSnapshot.counters[MetricType.NETWORK_BYTES_RECEIVED] ?: 0
            
            // Upload success rate.
            uploadSuccessRate = metricsSnapshot.summary.currentUploadSuccessRate
        }.build()
    }
    
    /**
     * Returns reporting statistics.
     */
    fun getUploadStats(): MetricsUploadStats {
        return MetricsUploadStats(
            enabled = _enabled.value,
            totalReportsSent = totalReportsSent,
            totalReportsSuccess = totalReportsSuccess,
            totalReportsFailed = totalReportsFailed,
            lastReportTime = lastReportTime,
            successRate = if (totalReportsSent > 0) {
                (totalReportsSuccess.toDouble() / totalReportsSent * 100)
            } else 0.0
        )
    }
    
    /**
     * Cleans up resources.
     */
    fun cleanup() {
        uploadScope.launch {
            disableMetricsReporting()
        }
        uploadScope.cancel()
    }
}

/**
 * Metrics upload statistics.
 */
data class MetricsUploadStats(
    val enabled: Boolean,
    val totalReportsSent: Long,
    val totalReportsSuccess: Long,
    val totalReportsFailed: Long,
    val lastReportTime: Long,
    val successRate: Double
)

// Helper extension function
private fun max(a: Long, b: Long): Long = if (a > b) a else b
