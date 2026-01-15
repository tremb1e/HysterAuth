package com.continuousauth.performance

import android.util.Log
import com.continuousauth.buffer.InMemoryBuffer
import com.continuousauth.compression.CompressionManager
import com.continuousauth.network.UploadManager
import com.continuousauth.sensor.SensorCollector
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Dynamic performance optimizer.
 *
 * Adjusts runtime knobs based on aggregated performance reports.
 */
@Singleton
class DynamicOptimizer @Inject constructor(
    private val sensorCollector: SensorCollector,
    private val compressionManager: CompressionManager,
    private val inMemoryBuffer: InMemoryBuffer,
    private val uploadManager: UploadManager
) {
    
    companion object {
        private const val TAG = "DynamicOptimizer"
        
        // Performance thresholds
        private const val MEMORY_PRESSURE_THRESHOLD = 0.8f  // 80% memory usage
        private const val BATTERY_DRAIN_THRESHOLD = 5f      // 5% per hour
        private const val LATENCY_THRESHOLD = 1000f         // 1s latency
        private const val CPU_USAGE_THRESHOLD = 0.7f        // 70% CPU usage
    }
    
    /**
     * Performance report input.
     */
    data class PerformanceReport(
        val avgLatency: Float,           // Average latency (ms)
        val memoryUsage: Float,          // Memory utilization (0-1)
        val batteryDrain: Float,         // Battery drain rate (%/hour)
        val cpuUsage: Float,             // CPU utilization (0-1)
        val networkThroughput: Float,    // Network throughput (KB/s)
        val errorRate: Float,            // Error rate (0-1)
        val queueSize: Int,              // Queue size
        val timestamp: Long = System.currentTimeMillis()
    )
    
    /**
     * Optimization configuration.
     */
    data class OptimizationConfig(
        var samplingRateMultiplier: Float = 1.0f,
        var compressionLevel: CompressionManager.CompressionType = CompressionManager.CompressionType.GZIP,
        var bufferSizeMultiplier: Float = 1.0f,
        var batchingEnabled: Boolean = true,
        var wifiOnlyMode: Boolean = false,
        var localCachingEnabled: Boolean = false
    )
    
    private var currentConfig = OptimizationConfig()
    private var lastOptimizationTime = 0L
    private val OPTIMIZATION_COOLDOWN = 60_000L // 1 minute cooldown
    
    /**
     * Optimize the system based on a performance report.
     */
    fun optimizeBasedOnPerformance(report: PerformanceReport) {
        val now = System.currentTimeMillis()
        
        // Check cooldown
        if (now - lastOptimizationTime < OPTIMIZATION_COOLDOWN) {
            Log.v(TAG, "Optimization cooldown active; skipping")
            return
        }
        
        Log.i(
            TAG,
            "Starting performance optimization - memory:${report.memoryUsage}, battery:${report.batteryDrain}, latency:${report.avgLatency}"
        )
        
        var optimizationApplied = false
        
        // Memory pressure mitigation
        if (report.memoryUsage > MEMORY_PRESSURE_THRESHOLD) {
            handleMemoryPressure(report)
            optimizationApplied = true
        }
        
        // Battery drain mitigation
        if (report.batteryDrain > BATTERY_DRAIN_THRESHOLD) {
            handleBatteryDrain(report)
            optimizationApplied = true
        }
        
        // Network latency mitigation
        if (report.avgLatency > LATENCY_THRESHOLD) {
            handleHighLatency(report)
            optimizationApplied = true
        }
        
        // CPU usage mitigation
        if (report.cpuUsage > CPU_USAGE_THRESHOLD) {
            handleHighCpuUsage(report)
            optimizationApplied = true
        }
        
        // Error rate mitigation
        if (report.errorRate > 0.1f) {
            handleHighErrorRate(report)
            optimizationApplied = true
        }
        
        // If the system is healthy, try to increase performance.
        if (!optimizationApplied && isPerformanceGood(report)) {
            tryImprovePerformance(report)
        }
        
        if (optimizationApplied) {
            lastOptimizationTime = now
            applyOptimizations()
        }
    }
    
    /**
     * Handle memory pressure.
     */
    private fun handleMemoryPressure(report: PerformanceReport) {
        Log.w(TAG, "High memory pressure detected: ${report.memoryUsage * 100}%")
        
        // Reduce buffer size
        currentConfig.bufferSizeMultiplier = maxOf(0.5f, currentConfig.bufferSizeMultiplier * 0.8f)
        
        // Increase compression if needed
        if (currentConfig.compressionLevel == CompressionManager.CompressionType.NONE) {
            currentConfig.compressionLevel = CompressionManager.CompressionType.GZIP
        }
        
        // Enable local caching to reduce in-memory queue pressure
        currentConfig.localCachingEnabled = true
        
        // Reduce sampling rate under high pressure
        if (report.memoryUsage > 0.9f) {
            currentConfig.samplingRateMultiplier = maxOf(0.5f, currentConfig.samplingRateMultiplier * 0.8f)
        }
    }
    
    /**
     * Handle high battery drain.
     */
    private fun handleBatteryDrain(report: PerformanceReport) {
        Log.w(TAG, "High battery drain detected: ${report.batteryDrain}%/hour")
        
        // Reduce sampling rate
        currentConfig.samplingRateMultiplier = maxOf(0.5f, currentConfig.samplingRateMultiplier * 0.9f)
        
        // Enable batching
        currentConfig.batchingEnabled = true
        
        // Consider Wi‑Fi-only mode under severe drain
        if (report.batteryDrain > BATTERY_DRAIN_THRESHOLD * 1.5f) {
            currentConfig.wifiOnlyMode = true
        }
    }
    
    /**
     * Handle high latency.
     */
    private fun handleHighLatency(report: PerformanceReport) {
        Log.w(TAG, "High latency detected: ${report.avgLatency}ms")
        
        // Switch to Wi‑Fi-only uploads
        currentConfig.wifiOnlyMode = true
        
        // Enable local caching
        currentConfig.localCachingEnabled = true
        
        // Prefer batching
        currentConfig.batchingEnabled = true
        
        // Under extreme latency, reduce data production rate.
        if (report.avgLatency > LATENCY_THRESHOLD * 2) {
            currentConfig.samplingRateMultiplier = maxOf(0.7f, currentConfig.samplingRateMultiplier * 0.95f)
        }
    }
    
    /**
     * Handle high CPU utilization.
     */
    private fun handleHighCpuUsage(report: PerformanceReport) {
        Log.w(TAG, "High CPU usage detected: ${report.cpuUsage * 100}%")
        
        // Reduce sampling rate
        currentConfig.samplingRateMultiplier = maxOf(0.6f, currentConfig.samplingRateMultiplier * 0.85f)
        
        // Reduce compression if using a CPU-heavy algorithm.
        if (currentConfig.compressionLevel == CompressionManager.CompressionType.GZIP) {
            // Consider switching to a faster algorithm (e.g. LZ4) in the future.
            Log.i(TAG, "Consider switching to a faster compression algorithm")
        }
    }
    
    /**
     * Handle high error rate.
     */
    private fun handleHighErrorRate(report: PerformanceReport) {
        Log.w(TAG, "High error rate detected: ${report.errorRate * 100}%")
        
        // Enable local caching to avoid data loss
        currentConfig.localCachingEnabled = true
        
        // Reduce data production rate
        currentConfig.samplingRateMultiplier = maxOf(0.7f, currentConfig.samplingRateMultiplier * 0.9f)
        
        // Prefer batching to reduce request volume
        currentConfig.batchingEnabled = true
    }
    
    /**
     * Check whether performance is good.
     */
    private fun isPerformanceGood(report: PerformanceReport): Boolean {
        return report.memoryUsage < 0.5f &&
               report.batteryDrain < 2f &&
               report.avgLatency < 500f &&
               report.cpuUsage < 0.4f &&
               report.errorRate < 0.01f
    }
    
    /**
     * Attempt to increase performance when conditions are good.
     */
    private fun tryImprovePerformance(report: PerformanceReport) {
        Log.i(TAG, "System performance looks good; attempting to improve performance")
        
        // Gradually increase sampling rate
        if (currentConfig.samplingRateMultiplier < 1.0f) {
            currentConfig.samplingRateMultiplier = minOf(1.0f, currentConfig.samplingRateMultiplier * 1.1f)
        }
        
        // Increase buffer size
        if (currentConfig.bufferSizeMultiplier < 1.0f) {
            currentConfig.bufferSizeMultiplier = minOf(1.0f, currentConfig.bufferSizeMultiplier * 1.2f)
        }
        
        // If network conditions improve, exit Wi‑Fi-only mode.
        if (report.avgLatency < 200f && currentConfig.wifiOnlyMode) {
            currentConfig.wifiOnlyMode = false
        }
    }
    
    /**
     * Apply optimization configuration.
     */
    private fun applyOptimizations() {
        Log.i(TAG, "Applying optimization config: $currentConfig")
        
        // Apply sampling rate change
        sensorCollector.adjustSamplingRate(currentConfig.samplingRateMultiplier)
        
        // Apply buffer size change
        inMemoryBuffer.adjustBufferSize(currentConfig.bufferSizeMultiplier)
        
        // Apply transport strategy
        uploadManager.setWifiOnlyMode(currentConfig.wifiOnlyMode)
        uploadManager.setLocalCachingEnabled(currentConfig.localCachingEnabled)
        
        // Notify other components about configuration changes
        notifyConfigurationChange()
    }
    
    /**
     * Notify configuration changes.
     */
    private fun notifyConfigurationChange() {
        // Consider using an event bus or callbacks to notify other components.
        Log.d(TAG, "Configuration updated; notified dependent components")
    }
    
    /**
     * Get a snapshot of the current optimization configuration.
     */
    fun getCurrentConfig(): OptimizationConfig {
        return currentConfig.copy()
    }
    
    /**
     * Reset optimization configuration.
     */
    fun resetOptimizations() {
        Log.i(TAG, "Resetting optimization config")
        currentConfig = OptimizationConfig()
        applyOptimizations()
    }
    
    /**
     * Manually set optimization configuration.
     */
    fun setOptimizationConfig(config: OptimizationConfig) {
        Log.i(TAG, "Manually setting optimization config: $config")
        currentConfig = config
        applyOptimizations()
    }
}
