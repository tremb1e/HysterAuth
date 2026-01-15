package com.continuousauth.ui.compose.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp

/**
 * Simple line chart.
 *
 * Used to display historical performance trends.
 */
@Composable
fun LineChart(
    data: List<Float>,
    modifier: Modifier = Modifier,
    lineColor: Color = MaterialTheme.colorScheme.primary,
    backgroundColor: Color = MaterialTheme.colorScheme.surface,
    gridLineColor: Color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f),
    maxValue: Float? = null,
    minValue: Float = 0f,
    showGrid: Boolean = true,
    strokeWidth: Float = 2f
) {
    val density = LocalDensity.current
    
    Canvas(
        modifier = modifier
            .fillMaxSize()
            .background(backgroundColor)
    ) {
        if (data.isEmpty()) return@Canvas
        
        val maxY = maxValue ?: data.maxOrNull() ?: 100f
        val minY = minValue
        val rangeY = maxY - minY
        
        // Draw grid.
        if (showGrid) {
            drawGridLines(
                gridLineColor = gridLineColor,
                horizontalLines = 4
            )
        }
        
        // Draw data line.
        drawDataLine(
            data = data,
            lineColor = lineColor,
            strokeWidth = strokeWidth,
            minY = minY,
            rangeY = rangeY
        )
        
        // Draw data points.
        drawDataPoints(
            data = data,
            pointColor = lineColor,
            minY = minY,
            rangeY = rangeY
        )
    }
}

/**
 * Draws grid lines.
 */
private fun DrawScope.drawGridLines(
    gridLineColor: Color,
    horizontalLines: Int = 4
) {
    val strokeWidth = 1.dp.toPx()
    val pathEffect = PathEffect.dashPathEffect(floatArrayOf(10f, 10f), 0f)
    
    // Draw horizontal grid lines.
    for (i in 0..horizontalLines) {
        val y = size.height * i / horizontalLines
        drawLine(
            color = gridLineColor,
            start = Offset(0f, y),
            end = Offset(size.width, y),
            strokeWidth = strokeWidth,
            pathEffect = pathEffect
        )
    }
}

/**
 * Draws the data line.
 */
private fun DrawScope.drawDataLine(
    data: List<Float>,
    lineColor: Color,
    strokeWidth: Float,
    minY: Float,
    rangeY: Float
) {
    if (data.size < 2) return
    
    val path = Path()
    val xStep = size.width / (data.size - 1).toFloat()
    
    data.forEachIndexed { index, value ->
        val x = index * xStep
        val normalizedValue = ((value - minY) / rangeY).coerceIn(0f, 1f)
        val y = size.height * (1f - normalizedValue)
        
        if (index == 0) {
            path.moveTo(x, y)
        } else {
            // Use a Bezier curve to smooth the line.
            val prevX = (index - 1) * xStep
            val prevValue = ((data[index - 1] - minY) / rangeY).coerceIn(0f, 1f)
            val prevY = size.height * (1f - prevValue)
            
            val controlX1 = prevX + xStep / 3f
            val controlY1 = prevY
            val controlX2 = x - xStep / 3f
            val controlY2 = y
            
            path.cubicTo(
                controlX1, controlY1,
                controlX2, controlY2,
                x, y
            )
        }
    }
    
    drawPath(
        path = path,
        color = lineColor,
        style = Stroke(
            width = strokeWidth.dp.toPx(),
            cap = StrokeCap.Round
        )
    )
}

/**
 * Draws data points.
 */
private fun DrawScope.drawDataPoints(
    data: List<Float>,
    pointColor: Color,
    minY: Float,
    rangeY: Float
) {
    val xStep = size.width / (data.size - 1).toFloat()
    val pointRadius = 3.dp.toPx()
    
    // Only show the most recent points to avoid excessive density.
    val maxPoints = 20
    val startIndex = if (data.size > maxPoints) data.size - maxPoints else 0
    
    data.subList(startIndex, data.size).forEachIndexed { relativeIndex, value ->
        val index = startIndex + relativeIndex
        val x = index * xStep
        val normalizedValue = ((value - minY) / rangeY).coerceIn(0f, 1f)
        val y = size.height * (1f - normalizedValue)
        
        // Draw point.
        drawCircle(
            color = pointColor,
            radius = pointRadius,
            center = Offset(x, y)
        )
        
        // Draw halo.
        drawCircle(
            color = pointColor.copy(alpha = 0.3f),
            radius = pointRadius * 1.5f,
            center = Offset(x, y)
        )
    }
}
