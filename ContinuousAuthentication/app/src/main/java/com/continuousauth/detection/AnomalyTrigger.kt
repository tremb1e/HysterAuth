package com.continuousauth.detection

/**
 * Anomaly trigger types.
 *
 * Represents different kinds of anomaly detection events.
 */
sealed class AnomalyTrigger {
    /**
     * Device unlock event.
     *
     * Triggered when the user unlocks the device.
     */
    object DeviceUnlocked : AnomalyTrigger()
    
    /**
     * Accelerometer spike event.
     *
     * Triggered when accelerometer data shows an unusual spike.
     *
     * @param magnitude Observed acceleration magnitude.
     * @param threshold Trigger threshold.
     * @param deviation Deviation score.
     */
    data class AccelerometerSpike(
        val magnitude: Float,
        val threshold: Float,
        val deviation: Float
    ) : AnomalyTrigger()
    
    /**
     * Sensitive app entry event.
     *
     * Triggered when the user enters a sensitive application.
     *
     * @param packageName Sensitive app package name.
     * @param appName App name (if available).
     */
    data class SensitiveAppEntered(
        val packageName: String,
        val appName: String? = null
    ) : AnomalyTrigger()
    
    /**
     * Manual trigger event.
     *
     * Used when the user or system manually triggers a high-sensitivity mode.
     */
    object MANUAL : AnomalyTrigger()
}

/**
 * Anomaly detection listener.
 *
 * Receives anomaly detection callbacks.
 */
interface OnAnomalyListener {
    /**
     * Called when an anomaly is detected.
     *
     * @param trigger Trigger event containing the anomaly type and associated data.
     */
    fun onAnomalyDetected(trigger: AnomalyTrigger)
    
    /**
     * Called when an anomaly state is cleared (optional).
     *
     * @param trigger The trigger that has been cleared.
     */
    fun onAnomalyCleared(trigger: AnomalyTrigger) {}
}

/**
 * Anomaly detector interface.
 *
 * Defines basic behavior for the anomaly detection module.
 */
interface AnomalyDetector {
    /**
     * Start anomaly detection.
     */
    suspend fun startDetection()
    
    /**
     * Stop anomaly detection.
     */
    suspend fun stopDetection()
    
    /**
     * Set anomaly listener.
     *
     * @param listener Listener instance.
     */
    fun setOnAnomalyListener(listener: OnAnomalyListener?)
    
    /**
     * Update detection policy.
     *
     * Allows updating internal thresholds and configuration.
     *
     * @param config New detection policy.
     */
    suspend fun updatePolicy(config: DetectionPolicy)
    
    /**
     * Process sensor data (used for accelerometer spike detection).
     *
     * @param x Acceleration along X axis.
     * @param y Acceleration along Y axis.
     * @param z Acceleration along Z axis.
     * @param timestamp Timestamp.
     */
    fun processSensorData(x: Float, y: Float, z: Float, timestamp: Long)
    
    /**
     * Returns whether detection is currently active.
     */
    fun isDetecting(): Boolean
}

/**
 * Detection policy configuration.
 *
 * Contains thresholds and parameters for anomaly detection.
 */
data class DetectionPolicy(
    // Accelerometer spike detection parameters
    val accelerometerSpikeThreshold: Float = 3.0f, // Spike detection threshold multiplier
    val accelerometerWindowSize: Int = 20, // Sliding window size
    val accelerometerCooldownMs: Long = 2000L, // Cooldown (ms)
    
    // Sensitive apps
    val sensitiveApps: Set<String> = emptySet(), // Sensitive app package names
    val appCheckIntervalMs: Long = 1000L, // App check interval (ms)
    
    // Device unlock detection
    val deviceUnlockEnabled: Boolean = true, // Enable device unlock detection
    val deviceUnlockCooldownMs: Long = 5000L, // Device unlock cooldown (ms)
    
    // Global settings
    val enabled: Boolean = true, // Enable anomaly detection
    val debugMode: Boolean = false // Debug mode
)
