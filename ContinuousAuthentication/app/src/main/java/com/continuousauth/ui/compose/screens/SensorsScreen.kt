package com.continuousauth.ui.compose.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.livedata.observeAsState
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.continuousauth.R
import com.continuousauth.ui.MainViewModel
import com.continuousauth.ui.viewmodels.SensorDataViewModel
import com.continuousauth.ui.theme.ExtendedColors
import androidx.compose.ui.viewinterop.AndroidView
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.components.XAxis
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlin.math.absoluteValue

/**
 * Sensor data screen.
 *
 * Displays real-time sensor data, charts, and recent foreground apps.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SensorsScreen(
    viewModel: MainViewModel,
    sensorViewModel: SensorDataViewModel
) {
    // Chart section is collapsed by default.
    var isChartExpanded by remember { mutableStateOf(false) }
    val scrollState = rememberScrollState()
    
    // Observe ViewModel data.
    val sensorStatus by viewModel.sensorStatus.observeAsState(emptyMap())
    val isCollectionRunning by viewModel.isCollectionRunning.observeAsState(false)
    val isEncryptedUploading by viewModel.isEncryptedUploading.observeAsState(false)
    
    // Auto-start sensor sampling (without encryption/upload).
    LaunchedEffect(Unit) {
        sensorViewModel.startSensorCollection()
    }
    
    // Observe real-time sensor data.
    val accelerometerData by sensorViewModel.accelerometerData.collectAsStateWithLifecycle()
    val gyroscopeData by sensorViewModel.gyroscopeData.collectAsStateWithLifecycle()
    val magnetometerData by sensorViewModel.magnetometerData.collectAsStateWithLifecycle()
    val recentApps by sensorViewModel.recentApps.collectAsStateWithLifecycle()
    
    // Observe chart data.
    val chartAccelerometerData by sensorViewModel.chartAccelerometerData.collectAsStateWithLifecycle()
    val chartGyroscopeData by sensorViewModel.chartGyroscopeData.collectAsStateWithLifecycle()
    val chartMagnetometerData by sensorViewModel.chartMagnetometerData.collectAsStateWithLifecycle()
    
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
            // Header card
            HeaderCard()
            
            // Real-time sensor card (with timestamps)
            RealTimeSensorCard(
                sensorStatus = sensorStatus,
                isRunning = true, // Always show as running
                isEncryptedUploading = isEncryptedUploading,
                accelerometerData = accelerometerData,
                gyroscopeData = gyroscopeData,
                magnetometerData = magnetometerData
            )
            
            // Collapsible chart section (MPAndroidChart)
            AnimatedChartSection(
                isExpanded = isChartExpanded,
                onToggleExpand = { 
                    isChartExpanded = !isChartExpanded
                    if (isChartExpanded) {
                        // Reset chart data when expanding.
                        sensorViewModel.resetChartData()
                    } else {
                        // Stop chart updates when collapsing.
                        sensorViewModel.stopChartData()
                    }
                },
                accelerometerData = chartAccelerometerData,  // Chart-only stream
                gyroscopeData = chartGyroscopeData,          // Chart-only stream
                magnetometerData = chartMagnetometerData     // Chart-only stream
            )
            
            // Recent foreground apps (max 10)
            RecentAppsCard(recentApps = recentApps)
            
            Spacer(modifier = Modifier.height(80.dp)) // Space for bottom navigation
        }
    }
}

/**
 * Header card.
 */
@Composable
fun HeaderCard() {
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
                            MaterialTheme.colorScheme.primary.copy(alpha = 0.1f),
                            MaterialTheme.colorScheme.tertiary.copy(alpha = 0.1f)
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
                    imageVector = Icons.Filled.Sensors,
                    contentDescription = null,
                    modifier = Modifier.size(48.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = stringResource(R.string.sensor_data_visualization),
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )
                Text(
                    text = stringResource(R.string.app_description),
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
 * Real-time sensor card (shows sensor data and timestamps).
 */
@Composable
fun RealTimeSensorCard(
    sensorStatus: Map<String, Boolean>,
    isRunning: Boolean,
    isEncryptedUploading: Boolean,
    accelerometerData: SensorDataViewModel.SensorData,
    gyroscopeData: SensorDataViewModel.SensorData,
    magnetometerData: SensorDataViewModel.SensorData
) {
    val timeFormatter = remember { SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault()) }
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "Live data",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold
                    )
                    if (isRunning) {
                        Text(
                            text = "Time: ${timeFormatter.format(Date())}",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
                
                // Status indicators
                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Collection status indicator
                    StatusIndicator(isActive = true) // Always active in this screen
                    
                    // Encrypted upload indicator
                    EncryptionStatusIndicator(isEncrypted = isEncryptedUploading)
                }
            }
            
            // Accelerometer
            SensorDataRow(
                icon = Icons.Outlined.Speed,
                name = "Accelerometer",
                isActive = sensorStatus["accelerometer"] == true,
                values = Triple(accelerometerData.x, accelerometerData.y, accelerometerData.z),
                timestamp = accelerometerData.timestamp,
                color = ExtendedColors.chartAccelerometer
            )
            
            HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
            
            // Gyroscope
            SensorDataRow(
                icon = Icons.Outlined.RotateRight,
                name = "Gyroscope",
                isActive = sensorStatus["gyroscope"] == true,
                values = Triple(gyroscopeData.x, gyroscopeData.y, gyroscopeData.z),
                timestamp = gyroscopeData.timestamp,
                color = ExtendedColors.chartGyroscope
            )
            
            HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
            
            // Magnetometer
            SensorDataRow(
                icon = Icons.Outlined.Explore,
                name = "Magnetometer",
                isActive = sensorStatus["magnetometer"] == true,
                values = Triple(magnetometerData.x, magnetometerData.y, magnetometerData.z),
                timestamp = magnetometerData.timestamp,
                color = ExtendedColors.chartMagnetometer
            )
        }
    }
}

/**
 * Sensor data row (with timestamp).
 */
@Composable
fun SensorDataRow(
    icon: ImageVector,
    name: String,
    isActive: Boolean,
    values: Triple<Float, Float, Float>,
    timestamp: Long,
    color: Color
) {
    val timeFormatter = remember { SimpleDateFormat("mm:ss.SSS", Locale.getDefault()) }
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Icon and name
        Box(
            modifier = Modifier
                .size(40.dp)
                .clip(CircleShape)
                .background(color.copy(alpha = 0.1f)),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = icon,
                contentDescription = name,
                tint = if (isActive) color else Color.Gray,
                modifier = Modifier.size(24.dp)
            )
        }
        
        Spacer(modifier = Modifier.width(12.dp))
        
        Column(modifier = Modifier.weight(1f)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text(
                        text = name,
                        style = MaterialTheme.typography.bodyLarge,
                        fontWeight = FontWeight.Medium
                    )
                    if (isActive && timestamp > 0) {
                        Text(
                            text = timeFormatter.format(Date(timestamp)),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                        )
                    }
                }
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // XYZ values
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                ValueChip(label = "X", value = values.first, color = color)
                ValueChip(label = "Y", value = values.second, color = color)
                ValueChip(label = "Z", value = values.third, color = color)
            }
        }
    }
}

/**
 * Value chip.
 */
@Composable
fun ValueChip(label: String, value: Float, color: Color) {
    Surface(
        shape = RoundedCornerShape(8.dp),
        color = color.copy(alpha = 0.1f),
        modifier = Modifier.width(90.dp)
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Center
        ) {
            Text(
                text = "$label:",
                style = MaterialTheme.typography.labelSmall,
                color = color,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.width(4.dp))
            Text(
                text = String.format("%.2f", value),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface
            )
        }
    }
}

/**
 * Control card.
 */
@Composable
fun ControlCard(
    isCollectionRunning: Boolean,
    onToggleCollection: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (isCollectionRunning) 
                ExtendedColors.success.copy(alpha = 0.1f) 
            else 
                MaterialTheme.colorScheme.surface
        )
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            contentAlignment = Alignment.Center
        ) {
            val infiniteTransition = rememberInfiniteTransition(label = "pulse")
            val scale by infiniteTransition.animateFloat(
                initialValue = 1f,
                targetValue = 1.05f,
                animationSpec = infiniteRepeatable(
                    animation = tween(1000),
                    repeatMode = RepeatMode.Reverse
                ),
                label = "scale"
            )
            
            Button(
                onClick = onToggleCollection,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
                    .scale(if (isCollectionRunning) scale else 1f),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isCollectionRunning) 
                        ExtendedColors.error 
                    else 
                        MaterialTheme.colorScheme.primary
                ),
                shape = RoundedCornerShape(16.dp)
            ) {
                Icon(
                    imageVector = if (isCollectionRunning) Icons.Filled.Stop else Icons.Filled.PlayArrow,
                    contentDescription = null,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = if (isCollectionRunning) "Stop collection" else "Start collection",
                    style = MaterialTheme.typography.labelLarge,
                    fontSize = 16.sp
                )
            }
        }
    }
}

/**
 * Collapsible chart section (MPAndroidChart).
 */
@Composable
fun AnimatedChartSection(
    isExpanded: Boolean,
    onToggleExpand: () -> Unit,
    accelerometerData: SensorDataViewModel.SensorData,
    gyroscopeData: SensorDataViewModel.SensorData,
    magnetometerData: SensorDataViewModel.SensorData
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onToggleExpand() },
        shape = RoundedCornerShape(20.dp)
    ) {
        Column {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Visualization",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                
                val rotation by animateFloatAsState(
                    targetValue = if (isExpanded) 180f else 0f,
                    animationSpec = tween(300),
                    label = "rotation"
                )
                
                Icon(
                    imageVector = Icons.Filled.ExpandMore,
                    contentDescription = null,
                    modifier = Modifier.rotate(rotation)
                )
            }
            
            AnimatedVisibility(
                visible = isExpanded,
                enter = expandVertically() + fadeIn(),
                exit = shrinkVertically() + fadeOut()
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 20.dp, vertical = 10.dp)
                ) {
                    // Accelerometer
                    SensorChartCard(
                        title = "Accelerometer",
                        sensorData = accelerometerData,
                        color = ExtendedColors.chartAccelerometer
                    )
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    // Gyroscope
                    SensorChartCard(
                        title = "Gyroscope",
                        sensorData = gyroscopeData,
                        color = ExtendedColors.chartGyroscope
                    )
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    // Magnetometer
                    SensorChartCard(
                        title = "Magnetometer",
                        sensorData = magnetometerData,
                        color = ExtendedColors.chartMagnetometer
                    )
                }
            }
        }
    }
}

/**
 * Sensor chart card (MPAndroidChart).
 */
@Composable
fun SensorChartCard(
    title: String,
    sensorData: SensorDataViewModel.SensorData,
    color: Color
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(12.dp)
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Bold,
                color = color
            )
            
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(top = 8.dp)
            ) {
                InteractiveLineChart(
                    xValue = sensorData.x,
                    yValue = sensorData.y,
                    zValue = sensorData.z,
                    timestamp = sensorData.timestamp,
                    color = color
                )
            }
        }
    }
}

/**
 * Interactive line chart (MPAndroidChart).
 */
@Composable
fun InteractiveLineChart(
    xValue: Float,
    yValue: Float,
    zValue: Float,
    timestamp: Long,
    color: Color
) {
    val maxDataPoints = 100 // Max points to show
    val xEntries = remember { mutableStateListOf<Entry>() }
    val yEntries = remember { mutableStateListOf<Entry>() }
    val zEntries = remember { mutableStateListOf<Entry>() }
    var dataIndex by remember { mutableStateOf(0f) }
    val startTime = remember { mutableStateOf(System.currentTimeMillis()) }
    
    // Update data.
    LaunchedEffect(timestamp) {
        if (timestamp > 0) {
            // Compute relative time in seconds.
            val relativeTime = (System.currentTimeMillis() - startTime.value) / 1000f
            
            xEntries.add(Entry(relativeTime, xValue))
            yEntries.add(Entry(relativeTime, yValue))
            zEntries.add(Entry(relativeTime, zValue))
            
            // No hard limit; chart can auto-scale.
            
            dataIndex = relativeTime
        }
    }
    
    AndroidView(
        factory = { context ->
            LineChart(context).apply {
                description.isEnabled = false
                legend.isEnabled = true
                setTouchEnabled(true)
                setScaleEnabled(true)
                isDragEnabled = true
                setPinchZoom(true)
                
                // X-axis (time).
                xAxis.position = XAxis.XAxisPosition.BOTTOM
                xAxis.setDrawGridLines(false)
                xAxis.granularity = 1f
                xAxis.valueFormatter = object : com.github.mikephil.charting.formatter.ValueFormatter() {
                    override fun getFormattedValue(value: Float): String {
                        return "${value.toInt()}s"  // Seconds
                    }
                }
                
                // Y-axis.
                axisLeft.setDrawGridLines(true)
                axisRight.isEnabled = false
                
                // Background.
                setBackgroundColor(android.graphics.Color.TRANSPARENT)
                setGridBackgroundColor(android.graphics.Color.TRANSPARENT)
            }
        },
        update = { chart ->
            val dataSets = mutableListOf<LineDataSet>()
            
            // X series (soft color).
            if (xEntries.isNotEmpty()) {
                val xDataSet = LineDataSet(xEntries.toList(), "X").apply {
                    setColor(android.graphics.Color.parseColor("#C08B8B")) // Soft red
                    lineWidth = 2f
                    setDrawCircles(false)
                    setDrawValues(false)
                    mode = LineDataSet.Mode.CUBIC_BEZIER
                }
                dataSets.add(xDataSet)
            }
            
            // Y series (soft color).
            if (yEntries.isNotEmpty()) {
                val yDataSet = LineDataSet(yEntries.toList(), "Y").apply {
                    setColor(android.graphics.Color.parseColor("#8BC08B")) // Soft green
                    lineWidth = 2f
                    setDrawCircles(false)
                    setDrawValues(false)
                    mode = LineDataSet.Mode.CUBIC_BEZIER
                }
                dataSets.add(yDataSet)
            }
            
            // Z series (soft color).
            if (zEntries.isNotEmpty()) {
                val zDataSet = LineDataSet(zEntries.toList(), "Z").apply {
                    setColor(android.graphics.Color.parseColor("#8B8BC0")) // Soft blue
                    lineWidth = 2f
                    setDrawCircles(false)
                    setDrawValues(false)
                    mode = LineDataSet.Mode.CUBIC_BEZIER
                }
                dataSets.add(zDataSet)
            }
            
            if (dataSets.isNotEmpty()) {
                chart.data = LineData(dataSets.toList())
                chart.invalidate()
                
                // Auto-scroll to latest.
                chart.moveViewToX(dataIndex)
            }
        },
        modifier = Modifier.fillMaxSize()
    )
}

/**
 * Recent foreground apps card (shows up to 10 apps).
 */
@Composable
fun RecentAppsCard(
    recentApps: List<SensorDataViewModel.RecentApp>
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Text(
                text = "Recent foreground apps",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 12.dp)
            )

            if (recentApps.isEmpty()) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 16.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "No recent apps",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                    )
                }
            } else {
                // Ensure uniqueness and correct ordering.
                val uniqueApps = remember(recentApps) {
                    recentApps.distinctBy { it.packageName }
                        .take(10)
                        .sortedByDescending { it.timestamp }
                }

                LazyColumn(
                    modifier = Modifier.heightIn(max = 200.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(
                        items = uniqueApps,
                        key = { it.packageName } // Use package name as stable key
                    ) { app ->
                        AppItem(
                            packageName = app.packageName,
                            appName = app.appName,
                            timestamp = app.timestamp
                        )
                    }
                }
            }
        }
    }
}

/**
 * App list item.
 */
@Composable
fun AppItem(
    packageName: String,
    appName: String,
    timestamp: Long
) {
    val timeFormatter = remember { SimpleDateFormat("HH:mm:ss", Locale.getDefault()) }
    
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.weight(1f)
            ) {
                Icon(
                    imageVector = Icons.Outlined.Apps,
                    contentDescription = null,
                    modifier = Modifier.size(20.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.width(8.dp))
                Column {
                    Text(
                        text = appName,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium
                    )
                    Text(
                        text = packageName,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
            Text(
                text = timeFormatter.format(Date(timestamp)),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

/**
 * Status indicator.
 */
@Composable
fun StatusIndicator(isActive: Boolean) {
    val infiniteTransition = rememberInfiniteTransition(label = "status")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000),
            repeatMode = RepeatMode.Reverse
        ),
        label = "alpha"
    )
    
    Row(
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(CircleShape)
                .background(
                    if (isActive) ExtendedColors.sensorActive.copy(alpha = alpha)
                    else Color.Gray
                )
        )
        Spacer(modifier = Modifier.width(4.dp))
        Text(
            text = if (isActive) "Collecting" else "Stopped",
            style = MaterialTheme.typography.labelSmall,
            color = if (isActive) ExtendedColors.sensorActive else Color.Gray
        )
    }
}

/**
 * Encrypted upload status indicator.
 */
@Composable
fun EncryptionStatusIndicator(isEncrypted: Boolean) {
    val infiniteTransition = rememberInfiniteTransition(label = "encryption")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1500),
            repeatMode = RepeatMode.Reverse
        ),
        label = "alpha"
    )
    
    Row(
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = if (isEncrypted) Icons.Filled.Lock else Icons.Filled.LockOpen,
            contentDescription = null,
            modifier = Modifier.size(12.dp),
            tint = if (isEncrypted) 
                MaterialTheme.colorScheme.primary.copy(alpha = alpha) 
            else 
                MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
        )
        Spacer(modifier = Modifier.width(4.dp))
        Text(
            text = if (isEncrypted) "Encrypted upload" else "Unencrypted upload",
            style = MaterialTheme.typography.labelSmall,
            color = if (isEncrypted) 
                MaterialTheme.colorScheme.primary 
            else 
                MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
        )
    }
}
