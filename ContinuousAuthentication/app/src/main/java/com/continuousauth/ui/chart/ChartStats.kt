package com.continuousauth.ui.chart

/**
 * Chart stats.
 */
data class ChartStats(
    val totalDataPoints: Int,        // Total points
    val accelerometerPoints: Int,    // Accelerometer points
    val gyroscopePoints: Int,        // Gyroscope points
    val magnetometerPoints: Int,     // Magnetometer points
    val minValue: Float,             // Min value
    val maxValue: Float,             // Max value
    val timeRangeSeconds: Float      // Time range (seconds)
)
