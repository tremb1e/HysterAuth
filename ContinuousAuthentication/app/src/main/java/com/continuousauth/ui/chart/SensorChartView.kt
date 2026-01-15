package com.continuousauth.ui.chart

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import androidx.core.view.GestureDetectorCompat
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.math.max
import kotlin.math.min

/**
 * Sensor chart view.
 *
 * Displays XYZ axes with interactive zoom and pan.
 */
class SensorChartView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    data class DataPoint(
        val x: Float,
        val y: Float,
        val z: Float,
        val timestamp: Long
    )
    
    // Data storage
    private val dataPoints = ConcurrentLinkedQueue<DataPoint>()
    private val maxDataPoints = 1000 // Keep up to 1000 points
    private var startTime = 0L
    
    // Paint objects
    private val xPaint = Paint().apply {
        strokeWidth = 2f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    
    private val yPaint = Paint().apply {
        strokeWidth = 2f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    
    private val zPaint = Paint().apply {
        strokeWidth = 2f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    
    private val gridPaint = Paint().apply {
        color = Color.GRAY
        alpha = 50
        strokeWidth = 1f
        style = Paint.Style.STROKE
    }
    
    private val textPaint = Paint().apply {
        color = Color.BLACK
        textSize = 24f
        isAntiAlias = true
    }
    
    private val axisPaint = Paint().apply {
        color = Color.BLACK
        strokeWidth = 2f
        style = Paint.Style.STROKE
    }
    
    // Interaction state
    private var scaleFactor = 1f
    private var translateX = 0f
    private var minValue = -10f
    private var maxValue = 10f
    private var isPaused = false
    private var isInteractive = false
    
    // Gesture detectors
    private val scaleGestureDetector = ScaleGestureDetector(context, ScaleListener())
    private val gestureDetector = GestureDetectorCompat(context, GestureListener())
    
    // Paths
    private val xPath = Path()
    private val yPath = Path()
    private val zPath = Path()
    
    // Detail view mode
    private var isDetailView = false
    
    // Stats
    private var accelerometerPointsCount = 0
    private var gyroscopePointsCount = 0
    private var magnetometerPointsCount = 0
    
    init {
        // Default colors
        setColorScheme(Color.RED, Color.GREEN, Color.BLUE)
    }
    
    /**
     * Set color scheme.
     */
    fun setColorScheme(xColor: Int, yColor: Int, zColor: Int) {
        xPaint.color = xColor
        yPaint.color = yColor
        zPaint.color = zColor
    }
    
    /**
     * Add a data point.
     */
    fun addDataPoint(x: Float, y: Float, z: Float, timestamp: Long) {
        if (isPaused) return
        
        if (startTime == 0L) {
            startTime = timestamp
        }
        
        dataPoints.add(DataPoint(x, y, z, timestamp))
        
        // Cap number of points
        while (dataPoints.size > maxDataPoints) {
            dataPoints.poll()
        }
        
        // Dynamically adjust display range
        val allValues = dataPoints.flatMap { listOf(it.x, it.y, it.z) }
        if (allValues.isNotEmpty()) {
            minValue = allValues.minOrNull() ?: -10f
            maxValue = allValues.maxOrNull() ?: 10f
            
            // Add some padding
            val range = maxValue - minValue
            minValue -= range * 0.1f
            maxValue += range * 0.1f
        }
        
        postInvalidate()
    }
    
    /**
     * Enable interactive mode.
     */
    fun enableInteractiveMode() {
        isInteractive = true
        setOnClickListener {
            // Tap to toggle detail view mode
            isDetailView = !isDetailView
            invalidate()
        }
    }
    
    /**
     * Pause updates.
     */
    fun pauseUpdates() {
        isPaused = true
    }
    
    /**
     * Resume updates.
     */
    fun resumeUpdates() {
        isPaused = false
    }
    
    /**
     * Cleanup resources.
     */
    fun cleanup() {
        dataPoints.clear()
        startTime = 0L
        accelerometerPointsCount = 0
        gyroscopePointsCount = 0
        magnetometerPointsCount = 0
    }
    
    /**
     * Clear data (alias for cleanup).
     */
    fun clearData() {
        cleanup()
    }
    
    /**
     * Add accelerometer sample.
     */
    fun addAccelerometerData(x: Float, y: Float, z: Float) {
        addDataPoint(x, y, z, System.nanoTime())
        accelerometerPointsCount++
    }
    
    /**
     * Add gyroscope sample.
     */
    fun addGyroscopeData(x: Float, y: Float, z: Float) {
        addDataPoint(x, y, z, System.nanoTime())
        gyroscopePointsCount++
    }
    
    /**
     * Add magnetometer sample.
     */
    fun addMagnetometerData(x: Float, y: Float, z: Float) {
        addDataPoint(x, y, z, System.nanoTime())
        magnetometerPointsCount++
    }
    
    /**
     * Get chart stats.
     */
    fun getDataStats(): ChartStats {
        val pointsList = dataPoints.toList()
        val timeRangeSeconds = if (pointsList.size > 1) {
            (pointsList.last().timestamp - pointsList.first().timestamp) / 1_000_000_000f
        } else {
            0f
        }
        
        return ChartStats(
            totalDataPoints = dataPoints.size,
            accelerometerPoints = accelerometerPointsCount,
            gyroscopePoints = gyroscopePointsCount,
            magnetometerPoints = magnetometerPointsCount,
            minValue = minValue,
            maxValue = maxValue,
            timeRangeSeconds = timeRangeSeconds
        )
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val width = width.toFloat()
        val height = height.toFloat()
        val padding = 60f
        val chartWidth = width - padding * 2
        val chartHeight = height - padding * 2
        
        // Draw background
        canvas.drawColor(Color.WHITE)
        
        // Draw grid
        drawGrid(canvas, padding, padding, chartWidth, chartHeight)
        
        // Draw axes
        drawAxes(canvas, padding, padding, chartWidth, chartHeight)
        
        // Draw data
        if (isDetailView) {
            drawDataDetailed(canvas, padding, padding, chartWidth, chartHeight)
        } else {
            drawDataCompressed(canvas, padding, padding, chartWidth, chartHeight)
        }
        
        // Draw legend
        drawLegend(canvas, padding)
        
        // Draw time labels
        drawTimeLabels(canvas, padding, padding + chartHeight, chartWidth)
    }
    
    private fun drawGrid(canvas: Canvas, startX: Float, startY: Float, width: Float, height: Float) {
        // Horizontal lines
        for (i in 0..10) {
            val y = startY + height * i / 10f
            canvas.drawLine(startX, y, startX + width, y, gridPaint)
        }
        
        // Vertical lines
        for (i in 0..10) {
            val x = startX + width * i / 10f
            canvas.drawLine(x, startY, x, startY + height, gridPaint)
        }
    }
    
    private fun drawAxes(canvas: Canvas, startX: Float, startY: Float, width: Float, height: Float) {
        // Y-axis
        canvas.drawLine(startX, startY, startX, startY + height, axisPaint)
        
        // X-axis
        canvas.drawLine(startX, startY + height, startX + width, startY + height, axisPaint)
        
        // Y-axis labels
        for (i in 0..5) {
            val y = startY + height * i / 5f
            val value = maxValue - (maxValue - minValue) * i / 5f
            textPaint.textAlign = Paint.Align.RIGHT
            canvas.drawText("%.1f".format(value), startX - 10f, y + 5f, textPaint)
        }
    }
    
    private fun drawDataCompressed(canvas: Canvas, startX: Float, startY: Float, width: Float, height: Float) {
        if (dataPoints.isEmpty()) return
        
        // Reset paths
        xPath.reset()
        yPath.reset()
        zPath.reset()
        
        val pointsList = dataPoints.toList()
        if (pointsList.isEmpty()) return
        
        val firstTime = pointsList.first().timestamp
        val lastTime = pointsList.last().timestamp
        val timeRange = lastTime - firstTime
        
        // Ensure a minimum time range
        val displayRange = max(timeRange, 1000L)
        
        var isFirst = true
        pointsList.forEach { point ->
            // Compute normalized x position from start time
            val timeDiff = point.timestamp - firstTime
            val x = startX + (timeDiff.toFloat() / displayRange) * width
            
            // X-axis series
            val yX = startY + height - ((point.x - minValue) / (maxValue - minValue)) * height
            if (isFirst) {
                xPath.moveTo(x, yX)
            } else {
                xPath.lineTo(x, yX)
            }
            
            // Y-axis series
            val yY = startY + height - ((point.y - minValue) / (maxValue - minValue)) * height
            if (isFirst) {
                yPath.moveTo(x, yY)
            } else {
                yPath.lineTo(x, yY)
            }
            
            // Z-axis series
            val yZ = startY + height - ((point.z - minValue) / (maxValue - minValue)) * height
            if (isFirst) {
                zPath.moveTo(x, yZ)
                isFirst = false
            } else {
                zPath.lineTo(x, yZ)
            }
        }
        
        // Draw paths
        canvas.drawPath(xPath, xPaint)
        canvas.drawPath(yPath, yPaint)
        canvas.drawPath(zPath, zPaint)
    }
    
    private fun drawDataDetailed(canvas: Canvas, startX: Float, startY: Float, width: Float, height: Float) {
        if (dataPoints.isEmpty()) return
        
        // Detail view: enable zoom and pan
        canvas.save()
        canvas.clipRect(startX, startY, startX + width, startY + height)
        
        // Apply pan and zoom
        canvas.translate(translateX, 0f)
        canvas.scale(scaleFactor, 1f, startX + width / 2, startY + height / 2)
        
        drawDataCompressed(canvas, startX, startY, width, height)
        
        canvas.restore()
    }
    
    private fun drawLegend(canvas: Canvas, startY: Float) {
        val legendY = startY - 20f
        val spacing = 120f
        
        // X label
        textPaint.color = xPaint.color
        textPaint.textAlign = Paint.Align.LEFT
        canvas.drawText("X", 60f, legendY, textPaint)
        canvas.drawLine(80f, legendY - 5f, 120f, legendY - 5f, xPaint)
        
        // Y label
        textPaint.color = yPaint.color
        canvas.drawText("Y", 60f + spacing, legendY, textPaint)
        canvas.drawLine(80f + spacing, legendY - 5f, 120f + spacing, legendY - 5f, yPaint)
        
        // Z label
        textPaint.color = zPaint.color
        canvas.drawText("Z", 60f + spacing * 2, legendY, textPaint)
        canvas.drawLine(80f + spacing * 2, legendY - 5f, 120f + spacing * 2, legendY - 5f, zPaint)
        
        // Mode and zoom level
        textPaint.color = Color.BLACK
        if (isDetailView) {
            canvas.drawText("Detail mode (zoom: %.1fx)".format(scaleFactor), width - 250f, legendY, textPaint)
        } else {
            canvas.drawText("Overview mode", width - 150f, legendY, textPaint)
        }
    }
    
    private fun drawTimeLabels(canvas: Canvas, startX: Float, y: Float, width: Float) {
        if (dataPoints.isEmpty()) return
        
        val pointsList = dataPoints.toList()
        val firstTime = pointsList.first().timestamp
        val lastTime = pointsList.last().timestamp
        val timeRange = (lastTime - firstTime) / 1000.0 // Convert to seconds
        
        textPaint.color = Color.BLACK
        textPaint.textAlign = Paint.Align.CENTER
        
        // Draw time labels
        canvas.drawText("0s", startX, y + 25f, textPaint)
        canvas.drawText("%.1fs".format(timeRange), startX + width, y + 25f, textPaint)
        
        // Midpoints
        for (i in 1..3) {
            val x = startX + width * i / 4f
            val time = timeRange * i / 4f
            canvas.drawText("%.1fs".format(time), x, y + 25f, textPaint)
        }
    }
    
    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (!isInteractive || !isDetailView) return super.onTouchEvent(event)
        
        scaleGestureDetector.onTouchEvent(event)
        gestureDetector.onTouchEvent(event)
        
        return true
    }
    
    // Scale listener
    private inner class ScaleListener : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            scaleFactor *= detector.scaleFactor
            scaleFactor = max(0.5f, min(scaleFactor, 5f))
            invalidate()
            return true
        }
    }
    
    // Gesture listener
    private inner class GestureListener : GestureDetector.SimpleOnGestureListener() {
        override fun onScroll(
            e1: MotionEvent?,
            e2: MotionEvent,
            distanceX: Float,
            distanceY: Float
        ): Boolean {
            translateX -= distanceX / scaleFactor
            val maxTranslate = width * (scaleFactor - 1) / 2
            translateX = max(-maxTranslate, min(translateX, maxTranslate))
            invalidate()
            return true
        }
        
        override fun onDoubleTap(e: MotionEvent): Boolean {
            // Double-tap to reset zoom and position
            scaleFactor = 1f
            translateX = 0f
            invalidate()
            return true
        }
    }
}
