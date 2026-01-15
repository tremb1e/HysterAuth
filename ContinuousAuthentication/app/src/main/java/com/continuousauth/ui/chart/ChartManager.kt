package com.continuousauth.ui.chart

import android.util.Log
import com.continuousauth.sensor.SensorCollector
import com.continuousauth.model.SensorType
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Chart manager.
 *
 * Manages sensor data visualization and batches updates to avoid dropping samples.
 */
@Singleton
class ChartManager @Inject constructor(
    private val sensorCollector: SensorCollector
) {
    
    companion object {
        private const val TAG = "ChartManager"
        private const val UPDATE_INTERVAL_MS = 100L // Chart update interval
    }
    
    private var chartView: SensorChartView? = null
    private var isVisualizationRunning = false
    private var visualizationJob: Job? = null
    private val chartScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    
    /**
     * Bind chart view.
     */
    fun bindChartView(chartView: SensorChartView) {
        this.chartView = chartView
        Log.d(TAG, "Chart view bound")
    }
    
    /**
     * Unbind chart view.
     */
    fun unbindChartView() {
        this.chartView = null
        Log.d(TAG, "Chart view unbound")
    }
    
    /**
     * Start visualization.
     *
     * Collects sensor samples from SensorCollector and updates the chart in batches.
     */
    fun startVisualization() {
        if (isVisualizationRunning) {
            Log.w(TAG, "Visualization is already running")
            return
        }
        
        isVisualizationRunning = true
        Log.i(TAG, "Starting visualization - subscribing to sensor stream")
        
        visualizationJob = chartScope.launch {
            try {
                // Sensor data stream
                val sensorDataFlow = sensorCollector.getSensorDataFlow()
                
                // Batch buffer to avoid dropping samples
                val batchBuffer = mutableListOf<com.continuousauth.model.SensorSample>()
                var lastBatchTime = System.currentTimeMillis()
                
                // Collect samples and update the chart in batches
                sensorDataFlow
                    .takeWhile { isVisualizationRunning }
                    .collect { sample ->
                        // Add sample to buffer
                        batchBuffer.add(sample)
                        
                        val currentTime = System.currentTimeMillis()
                        
                        // Flush every UPDATE_INTERVAL_MS or once the buffer reaches a threshold
                        if (currentTime - lastBatchTime >= UPDATE_INTERVAL_MS || batchBuffer.size >= 50) {
                            // Process all buffered samples
                            processBatchedSamples(batchBuffer.toList())
                            batchBuffer.clear()
                            lastBatchTime = currentTime
                        }
                    }
                    
            } catch (e: CancellationException) {
                Log.i(TAG, "Visualization job cancelled")
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Visualization data stream error", e)
                // If data stream fails, fall back to mock data
                fallbackToMockData()
            }
        }
    }
    
    /**
     * Process a batch of sensor samples.
     */
    private fun processBatchedSamples(samples: List<com.continuousauth.model.SensorSample>) {
        if (chartView == null || samples.isEmpty()) return
        
        Log.d(TAG, "Processing batch of ${samples.size} sensor samples")
        
        // Group by sensor type
        val groupedSamples = samples.groupBy { it.type }
        
        // Add samples for each sensor type
        groupedSamples.forEach { (type, typeSamples) ->
            typeSamples.forEach { sample ->
                when (type) {
                    SensorType.ACCELEROMETER -> {
                        chartView?.addAccelerometerData(sample.x, sample.y, sample.z)
                    }
                    SensorType.GYROSCOPE -> {
                        chartView?.addGyroscopeData(sample.x, sample.y, sample.z)
                    }
                    SensorType.MAGNETOMETER -> {
                        chartView?.addMagnetometerData(sample.x, sample.y, sample.z)
                    }
                }
            }
        }
    }
    
    /**
     * Fall back to mock data (when real sensor data is unavailable).
     */
    private suspend fun fallbackToMockData() {
        Log.w(TAG, "Falling back to mock data mode")
        
        while (isVisualizationRunning && visualizationJob?.isActive == true) {
            try {
                updateChartWithMockData()
                delay(UPDATE_INTERVAL_MS)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Mock data update error", e)
                delay(1000L)
            }
        }
    }
    
    /**
     * Stop visualization.
     */
    fun stopVisualization() {
        if (!isVisualizationRunning) {
            return
        }
        
        isVisualizationRunning = false
        visualizationJob?.cancel()
        Log.i(TAG, "Stopping visualization")
    }
    
    /**
     * Clear chart data.
     */
    fun clearChart() {
        chartView?.clearData()
        Log.d(TAG, "Chart data cleared")
    }
    
    /**
     * Add a single sensor sample to the chart.
     */
    fun addSensorData(sensorType: String, x: Float, y: Float, z: Float) {
        chartView?.let { chart ->
            when (sensorType) {
                "accelerometer" -> chart.addAccelerometerData(x, y, z)
                "gyroscope" -> chart.addGyroscopeData(x, y, z)
                "magnetometer" -> chart.addMagnetometerData(x, y, z)
                else -> Log.w(TAG, "Unknown sensor type: $sensorType")
            }
        }
    }
    
    /**
     * Get chart stats.
     */
    fun getChartStats(): ChartStats? {
        return chartView?.getDataStats()
    }
    
    /**
     * Whether visualization is running.
     */
    fun isVisualizationRunning(): Boolean {
        return isVisualizationRunning
    }
    
    /**
     * Update chart with mock data (testing/demo).
     */
    private fun updateChartWithMockData() {
        if (chartView == null) return
        
        val time = System.currentTimeMillis()
        val factor = (time / 1000.0) % (2 * Math.PI)
        
        // Mock accelerometer data
        val accX = (Math.sin(factor) * 5).toFloat()
        val accY = (Math.cos(factor) * 3).toFloat()
        val accZ = (Math.sin(factor * 2) * 2).toFloat()
        addSensorData("accelerometer", accX, accY, accZ)
        
        // Mock gyroscope data
        val gyroX = (Math.cos(factor * 1.5) * 2).toFloat()
        val gyroY = (Math.sin(factor * 1.2) * 1.5).toFloat()
        val gyroZ = (Math.cos(factor * 0.8) * 1).toFloat()
        addSensorData("gyroscope", gyroX, gyroY, gyroZ)
        
        // Mock magnetometer data
        val magX = (Math.sin(factor * 0.5) * 10).toFloat()
        val magY = (Math.cos(factor * 0.7) * 8).toFloat()
        val magZ = (Math.sin(factor * 0.3) * 6).toFloat()
        addSensorData("magnetometer", magX, magY, magZ)
    }
    
    /**
     * Cleanup resources.
     */
    fun cleanup() {
        stopVisualization()
        chartScope.cancel()
        chartView = null
        Log.d(TAG, "Chart manager cleaned up")
    }
}
