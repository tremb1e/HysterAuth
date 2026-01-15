package com.continuousauth.model

/**
 * Sensor type enum.
 */
enum class SensorType {
    ACCELEROMETER,  // Accelerometer
    GYROSCOPE,      // Gyroscope
    MAGNETOMETER    // Magnetometer
}

/**
 * Sensor sample model.
 */
data class SensorSample(
    val type: SensorType,                   // Sensor type
    val eventTimestampNs: Long,             // Relative timestamp (ns, based on elapsedRealtimeNanos)
    val x: Float,                           // X-axis
    val y: Float,                           // Y-axis
    val z: Float,                           // Z-axis
    val accuracy: Int,                      // Sensor accuracy
    val seqNo: Long,                        // Per-sample sequence number (anti-replay)
    val foregroundApp: String = ""          // Foreground app package name
)

/**
 * Transmission profile enum.
 */
enum class TransmissionProfile {
    WIFI_ONLY,      // Upload on Wi‑Fi only
    UNRESTRICTED    // No network restrictions (default)
}
