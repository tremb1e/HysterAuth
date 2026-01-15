package com.continuousauth.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock
import com.continuousauth.buffer.RingBuffer
import com.continuousauth.model.SensorSample
import com.continuousauth.model.SensorType
import com.continuousauth.pool.SensorEventPool
import com.continuousauth.utils.ForegroundAppDetector
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Sensor data collector implementation.
 *
 * Registers listeners via SensorManager on a dedicated coroutine dispatcher and
 * applies fixed sampling rates with optional batching via sensor FIFO.
 */
@Singleton
class SensorCollectorImpl @Inject constructor(
    @ApplicationContext private val context: Context,
    private val foregroundAppDetector: ForegroundAppDetector,
    private val ringBuffer: RingBuffer,
    private val sensorEventPool: SensorEventPool
) : SensorCollector, SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    
    // Use a single-thread dispatcher dedicated to sensor processing
    private val sensorDispatcher: CoroutineDispatcher = Dispatchers.IO.limitedParallelism(1)
    private val sensorScope = CoroutineScope(SupervisorJob() + sensorDispatcher)
    
    // Sensor instances
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
    
    // Output stream channel
    private val sensorDataChannel = Channel<SensorSample>(Channel.UNLIMITED)
    
    // Window batching coroutine (1-second window)
    private var windowBatchingJob: Job? = null
    
    // Current window sample buffer
    private val currentWindowSamples = mutableListOf<SensorSample>()
    private val windowMutex = Mutex()
    
    // State
    private val isCollecting = AtomicBoolean(false)
    private val collectionMutex = Mutex()
    
    // Sequence number generator
    private val sequenceNumber = AtomicLong(0L)
    
    // Fixed sampling periods (microseconds):
    // accelerometer + gyroscope: 200 Hz (5000 us), magnetometer: 100 Hz (10000 us)
    private val ACCELEROMETER_SAMPLING_PERIOD_US = 5000  // 200Hz
    private val GYROSCOPE_SAMPLING_PERIOD_US = 5000     // 200Hz
    private val MAGNETOMETER_SAMPLING_PERIOD_US = 10000  // 100Hz
    private var maxReportLatencyUs: Int = 0
    
    override suspend fun startCollection() {
        collectionMutex.withLock {
            if (isCollecting.get()) {
                return@withLock
            }
            
            // Detect and apply optimal batching latency based on hardware FIFO.
            detectOptimalSamplingRate()
            
            // Register sensor listeners with fixed sampling rates.
            val registrationResults = listOf(
                registerSensorIfAvailable(accelerometer, "Accelerometer", ACCELEROMETER_SAMPLING_PERIOD_US),
                registerSensorIfAvailable(gyroscope, "Gyroscope", GYROSCOPE_SAMPLING_PERIOD_US),
                registerSensorIfAvailable(magnetometer, "Magnetometer", MAGNETOMETER_SAMPLING_PERIOD_US)
            )
            
            if (registrationResults.any { it }) {
                isCollecting.set(true)
                
                // Start window batching coroutine (1s window).
                startWindowBatching()
                
                android.util.Log.i("SensorCollector", "Sensor collection started (1s window batching)")
            } else {
                throw IllegalStateException("No sensors available")
            }
        }
    }
    
    override suspend fun stopCollection() {
        collectionMutex.withLock {
            if (!isCollecting.get()) {
                return@withLock
            }
            
            sensorManager.unregisterListener(this)
            isCollecting.set(false)
            
            // Stop window batching
            windowBatchingJob?.cancel()
            windowBatchingJob = null
            
            // Clear ring buffer
            ringBuffer.clear()
            
            android.util.Log.i("SensorCollector", "Sensor collection stopped")
        }
    }
    
    override fun getSensorDataFlow(): Flow<SensorSample> {
        return sensorDataChannel.receiveAsFlow()
    }
    
    override fun isCollecting(): Boolean {
        return isCollecting.get()
    }
    
    override fun getSensorInfo(): SensorInfo {
        // Max sampling rate (Hz) = 1_000_000 / minDelayUs
        fun calculateMaxRate(sensor: Sensor?): Float {
            return sensor?.minDelay?.let { minDelay ->
                if (minDelay > 0) {
                    1000000.0f / minDelay
                } else {
                    0f
                }
            } ?: 0f
        }
        
        // Return sensor info, including fixed sampling rates.
        return SensorInfo(
            accelerometerMaxDelay = accelerometer?.maxDelay ?: 0,
            gyroscopeMaxDelay = gyroscope?.maxDelay ?: 0,
            magnetometerMaxDelay = magnetometer?.maxDelay ?: 0,
            accelerometerMaxRange = accelerometer?.maximumRange ?: 0f,
            gyroscopeMaxRange = gyroscope?.maximumRange ?: 0f,
            magnetometerMaxRange = magnetometer?.maximumRange ?: 0f,
            accelerometerFifoSize = accelerometer?.fifoMaxEventCount ?: 0,
            gyroscopeFifoSize = gyroscope?.fifoMaxEventCount ?: 0,
            magnetometerFifoSize = magnetometer?.fifoMaxEventCount ?: 0,
            accelerometerMaxRate = calculateMaxRate(accelerometer),
            gyroscopeMaxRate = calculateMaxRate(gyroscope),
            magnetometerMaxRate = calculateMaxRate(magnetometer),
            // Fixed sampling rate values
            accelerometerCurrentRate = 200f, // fixed at 200 Hz
            gyroscopeCurrentRate = 200f, // fixed at 200 Hz
            magnetometerCurrentRate = 100f // fixed at 100 Hz
        )
    }
    
    override fun onSensorChanged(event: SensorEvent) {
        if (!isCollecting.get()) return
        
        val sensorType = when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> SensorType.ACCELEROMETER
            Sensor.TYPE_GYROSCOPE -> SensorType.GYROSCOPE
            Sensor.TYPE_MAGNETIC_FIELD -> SensorType.MAGNETOMETER
            else -> return
        }
        
        // Process sensor events on the dedicated dispatcher.
        sensorScope.launch {
            // Get current foreground app
            val currentForegroundApp = try {
                foregroundAppDetector.getCurrentForegroundApp()
            } catch (e: Exception) {
                android.util.Log.e("SensorCollector", "Failed to get foreground app", e)
                ""
            }
            
            // Use object pool to reduce allocations and GC pressure.
            val pooledSample = sensorEventPool.acquire()
            pooledSample.setSensorData(
                type = sensorType,
                eventTimestampNs = event.timestamp,
                x = event.values[0],
                y = event.values[1],
                z = event.values[2],
                accuracy = event.accuracy,
                seqNo = sequenceNumber.incrementAndGet(),
                foregroundApp = currentForegroundApp
            )
            
            val sample = pooledSample.toSensorSample()
            
            // Add to the current window buffer.
            windowMutex.withLock {
                currentWindowSamples.add(sample)
            }
            
            // Return to pool
            pooledSample.release()
        }
    }
    
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Log accuracy changes
        sensor?.let {
            android.util.Log.d("SensorCollector", "Sensor ${it.name} accuracy changed to: $accuracy")
        }
    }
    
    /**
     * Detect and apply optimal batching latency.
     *
     * Uses fixed sampling rates: accelerometer + gyroscope 200 Hz, magnetometer 100 Hz.
     */
    private fun detectOptimalSamplingRate() {
        // Set maxReportLatencyUs based on FIFO size to better leverage hardware FIFO.
        val minFifoSize = minOf(
            accelerometer?.fifoMaxEventCount ?: Int.MAX_VALUE,
            gyroscope?.fifoMaxEventCount ?: Int.MAX_VALUE,
            magnetometer?.fifoMaxEventCount ?: Int.MAX_VALUE
        )
        
        if (minFifoSize > 0 && minFifoSize != Int.MAX_VALUE) {
            // Compute a suitable latency based on FIFO size and sampling rate.
            // Using 200 Hz as a baseline (5 ms interval).
            val samplingIntervalMs = 5 // 200Hz
            maxReportLatencyUs = (minFifoSize * samplingIntervalMs * 1000).coerceAtMost(1000000) // Max 1 second
            
            android.util.Log.i("SensorCollector", 
                "Detected min FIFO size: $minFifoSize; setting maxReportLatencyUs: ${maxReportLatencyUs}us")
            android.util.Log.i("SensorCollector", 
                "Sampling rates: accelerometer 200 Hz, gyroscope 200 Hz, magnetometer 100 Hz")
        } else {
            maxReportLatencyUs = 0 // Real-time reporting
            android.util.Log.i("SensorCollector", "Batching not supported; using real-time mode")
        }
    }
    
    /**
     * Register a sensor listener if the sensor is available.
     *
     * @param sensor Sensor instance
     * @param sensorName Sensor name (for logging)
     * @param samplingPeriodUs Sampling period (microseconds)
     */
    private fun registerSensorIfAvailable(sensor: Sensor?, sensorName: String, samplingPeriodUs: Int): Boolean {
        return sensor?.let {
            val success = sensorManager.registerListener(
                this,
                it,
                samplingPeriodUs,
                maxReportLatencyUs
            )
            
            if (success) {
                val samplingRateHz = 1000000.0f / samplingPeriodUs
                android.util.Log.i("SensorCollector", 
                    "$sensorName registered - rate: ${samplingRateHz}Hz, period: ${samplingPeriodUs}us, maxLatency: ${maxReportLatencyUs}us")
            } else {
                android.util.Log.w("SensorCollector", "$sensorName registration failed")
            }
            
            success
        } ?: run {
            android.util.Log.w("SensorCollector", "$sensorName unavailable")
            false
        }
    }
    
    /**
     * Start window batching coroutine.
     *
     * Batches samples every second and emits them via the output flow.
     */
    private fun startWindowBatching() {
        windowBatchingJob = sensorScope.launch {
            while (isCollecting.get()) {
                try {
                    // Wait for the 1-second window
                    delay(1000)
                    
                    // Collect samples in the current window
                    val windowSamples = windowMutex.withLock {
                        val samples = currentWindowSamples.toList()
                        currentWindowSamples.clear()
                        samples
                    }
                    
                    if (windowSamples.isNotEmpty()) {
                        android.util.Log.d("SensorCollector", 
                            "Window batch: collected ${windowSamples.size} samples")
                        
                        // Emit samples downstream for further processing
                        windowSamples.forEach { sample ->
                            sensorDataChannel.trySend(sample)
                        }
                    }
                } catch (e: Exception) {
                    android.util.Log.e("SensorCollector", "Window batching error", e)
                    delay(1000) // Backoff and retry
                }
            }
        }
    }
    
    /**
     * Adjust sampling rate.
     */
    override fun adjustSamplingRate(multiplier: Float) {
        android.util.Log.i("SensorCollector", "Adjust sampling rate multiplier: $multiplier")
        // TODO: Implement sampling rate adjustment.
    }
}
