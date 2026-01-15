package com.continuousauth.ui.compose.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.continuousauth.ui.MainViewModel
import com.continuousauth.ui.theme.ExtendedColors
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlin.math.cos
import kotlin.math.sin

/**
 * Details screen.
 *
 * Dashboard showing system metrics, performance data, and debug information.
 */
@Composable
fun DetailsScreen(viewModel: MainViewModel) {
    val scrollState = rememberScrollState()
    
    // Mock data
    val transmissionMode by remember { mutableStateOf("SLOW_MODE") }
    val creditScore by remember { mutableStateOf(95) }
    
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(scrollState)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Header
            DashboardHeader()
            
            // Key metrics
            MetricsGrid()
            
            // Transmission status
            TransmissionStatusCard(transmissionMode)
            
            // Credit score
            CreditScoreCard(creditScore)
            
            // System performance
            SystemPerformanceCard()
            
            // Hardware info
            HardwareInfoCard()
            
            // Policy configuration
            PolicyConfigCard()
            
            // TLS security info
            TlsSecurityCard(viewModel)
            
            // Export logs
            ExportLogsCard()
            
            Spacer(modifier = Modifier.height(80.dp))
        }
    }
}

/**
 * Dashboard header.
 */
@Composable
fun DashboardHeader() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
        )
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.horizontalGradient(
                        colors = listOf(
                            ExtendedColors.gradientStart.copy(alpha = 0.1f),
                            ExtendedColors.gradientEnd.copy(alpha = 0.1f)
                        )
                    )
                )
                .padding(24.dp)
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(
                    imageVector = Icons.Filled.Dashboard,
                    contentDescription = null,
                    modifier = Modifier.size(48.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "System Dashboard",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )
                Text(
                    text = "Monitor system performance and transmission status in real time",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f),
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }
    }
}

/**
 * Key metrics grid.
 */
@Composable
fun MetricsGrid() {
    val metrics = listOf(
        MetricItem(Icons.Outlined.Upload, "Uploads succeeded", "1,234", ExtendedColors.success),
        MetricItem(Icons.Outlined.Error, "Uploads failed", "12", ExtendedColors.error),
        MetricItem(Icons.Outlined.Timer, "ACK latency", "45ms", ExtendedColors.info),
        MetricItem(Icons.Outlined.Storage, "Queue size", "23MB", ExtendedColors.warning),
        MetricItem(Icons.Outlined.Memory, "Memory usage", "156MB", MaterialTheme.colorScheme.primary),
        MetricItem(Icons.Outlined.Speed, "Sampling rate", "200Hz", MaterialTheme.colorScheme.secondary)
    )
    
    LazyVerticalGrid(
        columns = GridCells.Fixed(2),
        modifier = Modifier
            .fillMaxWidth()
            .height(240.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        userScrollEnabled = false
    ) {
        items(metrics) { metric ->
            MetricCard(metric)
        }
    }
}

/**
 * Metric card.
 */
@Composable
fun MetricCard(metric: MetricItem) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = metric.color.copy(alpha = 0.1f)
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = metric.icon,
                contentDescription = null,
                tint = metric.color,
                modifier = Modifier.size(24.dp)
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = metric.value,
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = metric.color
            )
            Text(
                text = metric.label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

/**
 * Transmission status card.
 */
@Composable
fun TransmissionStatusCard(mode: String) {
    var remainingTime by remember { mutableStateOf(0) }
    val isFastMode = mode == "FAST_MODE"
    
    LaunchedEffect(isFastMode) {
        if (isFastMode) {
            remainingTime = 5
            while (remainingTime > 0 && isActive) {
                delay(1000)
                remainingTime--
            }
        }
    }
    
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (isFastMode) 
                ExtendedColors.fastMode.copy(alpha = 0.1f)
            else 
                ExtendedColors.slowMode.copy(alpha = 0.1f)
        )
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "Transmission mode",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = if (isFastMode) "Fast mode" else "Slow mode",
                        style = MaterialTheme.typography.headlineSmall,
                        color = if (isFastMode) ExtendedColors.fastMode else ExtendedColors.slowMode,
                        fontWeight = FontWeight.Bold
                    )
                }
                
                // Animated icon
                AnimatedTransmissionIcon(isFastMode)
            }
            
            if (isFastMode && remainingTime > 0) {
                Spacer(modifier = Modifier.height(16.dp))
                LinearProgressIndicator(
                    progress = remainingTime / 5f,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp)
                        .clip(RoundedCornerShape(4.dp)),
                    color = ExtendedColors.fastMode,
                    trackColor = ExtendedColors.fastMode.copy(alpha = 0.2f)
                )
                Text(
                    text = "Switching to slow mode in $remainingTime s",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 8.dp)
                )
            }
            
            Spacer(modifier = Modifier.height(12.dp))
            
            // Trigger info
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                TriggerChip("Device unlock", Icons.Outlined.LockOpen)
                TriggerChip("App switch", Icons.Outlined.Apps)
                TriggerChip("Acceleration spike", Icons.Outlined.ShowChart)
            }
        }
    }
}

/**
 * Trigger chip.
 */
@Composable
fun TriggerChip(label: String, icon: ImageVector) {
    AssistChip(
        onClick = { },
        label = {
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall
            )
        },
        leadingIcon = {
            Icon(
                imageVector = icon,
                contentDescription = null,
                modifier = Modifier.size(16.dp)
            )
        },
        modifier = Modifier.height(32.dp)
    )
}

/**
 * Animated transmission icon.
 */
@Composable
fun AnimatedTransmissionIcon(isFastMode: Boolean) {
    val infiniteTransition = rememberInfiniteTransition(label = "transmission")
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(if (isFastMode) 1000 else 3000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "rotation"
    )
    
    Box(
        modifier = Modifier.size(60.dp),
        contentAlignment = Alignment.Center
    ) {
        Canvas(
            modifier = Modifier
                .size(60.dp)
                .rotate(rotation)
        ) {
            val color = if (isFastMode) ExtendedColors.fastMode else ExtendedColors.slowMode
            drawCircularIndicator(color)
        }
        
        Icon(
            imageVector = if (isFastMode) Icons.Filled.FlashOn else Icons.Filled.PowerSettingsNew,
            contentDescription = null,
            tint = if (isFastMode) ExtendedColors.fastMode else ExtendedColors.slowMode,
            modifier = Modifier.size(24.dp)
        )
    }
}

/**
 * Credit score card.
 */
@Composable
fun CreditScoreCard(score: Int) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "Trust score",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Circular indicator
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier.size(120.dp)
            ) {
                CircularProgressIndicator(
                    progress = score / 100f,
                    modifier = Modifier.fillMaxSize(),
                    strokeWidth = 12.dp,
                    color = when {
                        score >= 80 -> ExtendedColors.success
                        score >= 60 -> ExtendedColors.warning
                        else -> ExtendedColors.error
                    },
                    trackColor = MaterialTheme.colorScheme.surfaceVariant
                )
                
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = score.toString(),
                        style = MaterialTheme.typography.headlineLarge,
                        fontWeight = FontWeight.Bold,
                        color = when {
                            score >= 80 -> ExtendedColors.success
                            score >= 60 -> ExtendedColors.warning
                            else -> ExtendedColors.error
                        }
                    )
                    Text(
                        text = when {
                            score >= 80 -> "Excellent"
                            score >= 60 -> "Good"
                            else -> "Needs improvement"
                        },
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            
            Text(
                text = "Based on upload success rate, connection stability, and other factors",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(top = 12.dp)
            )
        }
    }
}

/**
 * System performance card.
 */
@Composable
fun SystemPerformanceCard() {
    var cpuUsage by remember { mutableStateOf(0) }
    var memoryUsage by remember { mutableStateOf(0) }
    var batteryLevel by remember { mutableStateOf(85) }
    var temperature by remember { mutableStateOf(36.5f) }
    
    LaunchedEffect(Unit) {
        while (isActive) {
            cpuUsage = (20..40).random()
            memoryUsage = (150..250).random()
            delay(2000)
        }
    }
    
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Text(
                text = "System performance",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 16.dp)
            )
            
            PerformanceRow(
                icon = Icons.Outlined.Memory,
                label = "CPU usage",
                value = "$cpuUsage%",
                progress = cpuUsage / 100f,
                color = ExtendedColors.info
            )
            
            PerformanceRow(
                icon = Icons.Outlined.Storage,
                label = "Memory usage",
                value = "${memoryUsage}MB",
                progress = memoryUsage / 512f,
                color = MaterialTheme.colorScheme.primary
            )
            
            PerformanceRow(
                icon = Icons.Outlined.BatteryFull,
                label = "Battery",
                value = "$batteryLevel%",
                progress = batteryLevel / 100f,
                color = ExtendedColors.success
            )
            
            PerformanceRow(
                icon = Icons.Outlined.Thermostat,
                label = "Temperature",
                value = "${temperature}°C",
                progress = temperature / 50f,
                color = if (temperature > 40) ExtendedColors.warning else ExtendedColors.info
            )
        }
    }
}

/**
 * Performance metric row.
 */
@Composable
fun PerformanceRow(
    icon: ImageVector,
    label: String,
    value: String,
    progress: Float,
    color: Color
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = color,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = label,
                    style = MaterialTheme.typography.bodyMedium
                )
            }
            Text(
                text = value,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Bold,
                color = color
            )
        }
        
        Spacer(modifier = Modifier.height(4.dp))
        
        LinearProgressIndicator(
            progress = progress.coerceIn(0f, 1f),
            modifier = Modifier
                .fillMaxWidth()
                .height(4.dp)
                .clip(RoundedCornerShape(2.dp)),
            color = color,
            trackColor = color.copy(alpha = 0.2f)
        )
    }
}

/**
 * Hardware information card.
 */
@Composable
fun HardwareInfoCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Text(
                text = "Hardware",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 12.dp)
            )
            
            InfoRow("Device model", android.os.Build.MODEL)
            InfoRow("Android version", "API ${android.os.Build.VERSION.SDK_INT}")
            InfoRow("Keystore", "Hardware-backed")
            InfoRow("StrongBox support", "Yes")
            InfoRow("Sensor FIFO", "8192 events")
            InfoRow("Max sampling rate", "200 Hz")
        }
    }
}

/**
 * Policy configuration card.
 */
@Composable
fun PolicyConfigCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Text(
                text = "Current policy",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 12.dp)
            )
            
            InfoRow("Slow mode interval", "1000ms")
            InfoRow("Fast mode interval", "150ms")
            InfoRow("Fast mode duration", "5000ms")
            InfoRow("Acceleration threshold", "3.0σ")
            InfoRow("Sensitive apps", "12")
            InfoRow("Last updated", "2 min ago")
        }
    }
}

/**
 * Info row.
 */
@Composable
fun InfoRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium
        )
    }
}

/**
 * TLS security information card.
 *
 * Displays TLS configuration and certificate pinning details.
 */
@Composable
fun TlsSecurityCard(viewModel: MainViewModel) {
    val tlsConfig by viewModel.tlsConfigInfo.observeAsState()
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Filled.Security,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(12.dp))
                Text(
                    text = "TLS security",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // TLS version info
            val supportsTls13 = tlsConfig?.supportsTls13 ?: false
            InfoRow("TLS version", if (supportsTls13) "TLS 1.3" else "TLS 1.2")
            InfoRow("Enforce TLS 1.3", if (supportsTls13) "Enabled" else "Unavailable")
            InfoRow("Supported protocols", "${tlsConfig?.supportedProtocols?.size ?: 0}")
            
            Divider(
                modifier = Modifier.padding(vertical = 8.dp),
                color = MaterialTheme.colorScheme.surfaceVariant
            )
            
            // Certificate pinning info
            InfoRow("SPKI pinning", "Configured")
            InfoRow("Certificate pins", "0") // Controlled by server policy
            InfoRow("Pin rotation", "Automatic")
            
            Divider(
                modifier = Modifier.padding(vertical = 8.dp),
                color = MaterialTheme.colorScheme.surfaceVariant
            )
            
            // Cipher suite info
            val recommendedCiphers = tlsConfig?.recommendedCiphers ?: emptyList()
            val preferredCipher = recommendedCiphers.firstOrNull { it.contains("AES") && it.contains("GCM") }
                ?: recommendedCiphers.firstOrNull()
                ?: "Not configured"
            InfoRow("Preferred cipher", if (preferredCipher.length > 30) preferredCipher.take(27) + "..." else preferredCipher)
            InfoRow("Cipher count", recommendedCiphers.size.toString())
            InfoRow("Last verification", "Just now")
        }
    }
}

/**
 * Export logs card.
 */
@Composable
fun ExportLogsCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            OutlinedButton(
                onClick = { /* Export logs */ },
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(
                    imageVector = Icons.Outlined.FileDownload,
                    contentDescription = null,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Export logs")
            }
            
            OutlinedButton(
                onClick = { /* Share diagnostics */ },
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(
                    imageVector = Icons.Outlined.Share,
                    contentDescription = null,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Share diagnostics")
            }
        }
    }
}

/**
 * Draws a circular indicator.
 */
fun DrawScope.drawCircularIndicator(color: Color) {
    val strokeWidth = 3.dp.toPx()
    val radius = (size.minDimension - strokeWidth) / 2
    val center = Offset(size.width / 2, size.height / 2)
    
    // Draw multiple arcs.
    for (i in 0..3) {
        val startAngle = i * 90f
        drawArc(
            color = color.copy(alpha = 0.3f + i * 0.2f),
            startAngle = startAngle,
            sweepAngle = 60f,
            useCenter = false,
            style = Stroke(width = strokeWidth, cap = StrokeCap.Round),
            size = size.copy(width = radius * 2, height = radius * 2),
            topLeft = Offset(center.x - radius, center.y - radius)
        )
    }
}

/**
 * Metric item model.
 */
data class MetricItem(
    val icon: ImageVector,
    val label: String,
    val value: String,
    val color: Color
)
