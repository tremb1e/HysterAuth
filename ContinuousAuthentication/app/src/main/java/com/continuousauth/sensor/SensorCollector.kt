package com.continuousauth.sensor

import com.continuousauth.model.SensorSample
import kotlinx.coroutines.flow.Flow

/**
 * Sensor data collector interface.
 */
interface SensorCollector {
    
    /**
     * Start collecting sensor data.
     */
    suspend fun startCollection()
    
    /**
     * Stop collecting sensor data.
     */
    suspend fun stopCollection()
    
    /**
     * Get the sensor data stream.
     */
    fun getSensorDataFlow(): Flow<SensorSample>
    
    /**
     * Returns whether collection is currently active.
     */
    fun isCollecting(): Boolean
    
    /**
     * Get sensor capabilities and current configuration.
     */
    fun getSensorInfo(): SensorInfo
    
    /**
     * Adjust sampling rate by a multiplier.
     */
    fun adjustSamplingRate(multiplier: Float)
}

/**
 * Sensor information model.
 */
data class SensorInfo(
    val accelerometerMaxDelay: Int, // Accelerometer max delay
    val gyroscopeMaxDelay: Int, // Gyroscope max delay
    val magnetometerMaxDelay: Int, // Magnetometer max delay
    val accelerometerMaxRange: Float, // Accelerometer max range
    val gyroscopeMaxRange: Float, // Gyroscope max range
    val magnetometerMaxRange: Float, // Magnetometer max range
    val accelerometerFifoSize: Int, // Accelerometer FIFO size
    val gyroscopeFifoSize: Int, // Gyroscope FIFO size
    val magnetometerFifoSize: Int, // Magnetometer FIFO size
    val accelerometerMaxRate: Float = 0f, // Accelerometer max sampling rate (Hz)
    val gyroscopeMaxRate: Float = 0f, // Gyroscope max sampling rate (Hz)
    val magnetometerMaxRate: Float = 0f, // Magnetometer max sampling rate (Hz)
    val accelerometerCurrentRate: Float = 200f, // Accelerometer current rate (Hz) - fixed at 200 Hz
    val gyroscopeCurrentRate: Float = 200f, // Gyroscope current rate (Hz) - fixed at 200 Hz
    val magnetometerCurrentRate: Float = 100f // Magnetometer current rate (Hz) - fixed at 100 Hz
)
