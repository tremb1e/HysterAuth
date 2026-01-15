package com.continuousauth.ui.compose.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.continuousauth.R
import com.continuousauth.monitor.SystemMonitor
import com.continuousauth.ui.compose.components.LineChart
import com.continuousauth.ui.theme.ContinuousAuthTheme
import com.continuousauth.ui.viewmodels.DetailedInfoViewModel
import kotlinx.coroutines.launch
import java.text.DecimalFormat

/**
 * Detailed information screen.
 *
 * Shows detailed status, performance metrics, and transmission information.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DetailedInfoScreen(
    onNavigateBack: () -> Unit,
    viewModel: DetailedInfoViewModel = hiltViewModel()
) {
    val scrollState = rememberScrollState()
    val scope = rememberCoroutineScope()
    
    // Collected state
    val transmissionStatus by viewModel.transmissionStatus.collectAsStateWithLifecycle()
    val grpcStatus by viewModel.grpcStatus.collectAsStateWithLifecycle()
    val bufferStats by viewModel.bufferStats.collectAsStateWithLifecycle()
    val sensorInfo by viewModel.sensorInfo.collectAsStateWithLifecycle()
    val deviceInfo by viewModel.deviceInfo.collectAsStateWithLifecycle()
    val serverPolicy by viewModel.serverPolicy.collectAsStateWithLifecycle()
    val timeSyncStatus by viewModel.timeSyncStatus.collectAsStateWithLifecycle()
    val performanceMetrics by viewModel.performanceMetrics.collectAsStateWithLifecycle()
    val cpuHistory by viewModel.cpuHistory.collectAsStateWithLifecycle()
    val memoryHistory by viewModel.memoryHistory.collectAsStateWithLifecycle()
    val latencyHistory by viewModel.latencyHistory.collectAsStateWithLifecycle()
    
    val decimalFormat = remember { DecimalFormat("#.##") }
    // Toggle NTP time sync card visibility.
    var showNtpSyncCard by remember { mutableStateOf(false) }
    // Toggle transmission status card visibility.
    var showTransmissionStatusCard by remember { mutableStateOf(false) }
    Box(
        modifier = Modifier.fillMaxSize()
    ) {
        LazyColumn(
            modifier = Modifier
                .fillMaxSize(),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if(showTransmissionStatusCard){
                // Transmission status card
                item {
                    InfoCard(
                        title = "Transmission Status",
                        icon = Icons.AutoMirrored.Filled.Send
                    ) {
                        InfoRow("Transmission profile", transmissionStatus.currentProfile)
                        InfoRow(
                            "Connection",
                            if (transmissionStatus.isConnected) "Connected" else "Disconnected",
                            textColor = if (transmissionStatus.isConnected)
                                MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error
                        )
                        InfoRow("Upload queue", "${transmissionStatus.uploadQueueSize} packets")
                    }
                }
            }
            // gRPC connection status card
            item {
                InfoCard(
                    title = "gRPC Status",
                    icon = Icons.Default.Cloud
                ) {
                    InfoRow("Endpoint", grpcStatus.endpoint.ifEmpty { "Not connected" })
                    InfoRow(
                        "Status",
                        when (grpcStatus.connectionState) {
                            SystemMonitor.ConnectionState.CONNECTED -> "Connected"
                            SystemMonitor.ConnectionState.CONNECTING -> "Connecting..."
                            SystemMonitor.ConnectionState.DISCONNECTED -> "Disconnected"
                            SystemMonitor.ConnectionState.TRANSIENT_FAILURE -> "Failed"
                        },
                        textColor = when (grpcStatus.connectionState) {
                            SystemMonitor.ConnectionState.CONNECTED -> Color(0xFF4CAF50)
                            SystemMonitor.ConnectionState.CONNECTING -> Color(0xFFFF9800)
                            else -> MaterialTheme.colorScheme.error
                        }
                    )
                    InfoRow(
                        "ACK latency",
                        if (grpcStatus.lastAckLatencyMs > 0) "${grpcStatus.lastAckLatencyMs} ms" else "N/A"
                    )
                    InfoRow("Total sent", "${grpcStatus.totalPacketsSent}")
                    InfoRow("Acknowledged", "${grpcStatus.totalPacketsAcknowledged}")
                }
            }
            
            // Buffer stats card
            item {
                InfoCard(
                    title = "Buffer Statistics",
                    icon = Icons.Default.Storage
                ) {
                    InfoRow("Memory samples", "${bufferStats.memorySamples}")
                    InfoRow("Disk queue packets", "${bufferStats.packetsInDiskQueue}")
                    InfoRow("Sent", "${bufferStats.totalSentCount}")
                    InfoRow("Failed", "${bufferStats.totalFailedCount}")
                    InfoRow("Discarded", "${bufferStats.totalDiscardedCount}")
                    InfoRow("Disk queue size", "${decimalFormat.format(bufferStats.diskQueueSizeMB)} MB")
                }
            }
            // Conditionally show NTP time sync card
            if (showNtpSyncCard) {
                // NTP time sync card
                item {
                    InfoCard(
                        title = "NTP Time Sync",
                        icon = Icons.Default.Schedule
                    ) {
                        InfoRow(
                            "Sync status",
                            timeSyncStatus.syncStatus,
                            textColor = when (timeSyncStatus.syncStatus) {
                                "SUCCESS" -> Color(0xFF4CAF50)
                                "SYNCING" -> Color(0xFFFF9800)
                                "ERROR" -> MaterialTheme.colorScheme.error
                                else -> MaterialTheme.colorScheme.onSurface
                            }
                        )
                        InfoRow(
                            "Sync valid",
                            if (timeSyncStatus.isNtpSyncValid) "Yes" else "No",
                            textColor = if (timeSyncStatus.isNtpSyncValid)
                                Color(0xFF4CAF50) else MaterialTheme.colorScheme.error
                        )
                        InfoRow("NTP offset", "${timeSyncStatus.ntpOffsetMs} ms")
                        InfoRow("Accuracy", "${timeSyncStatus.syncAccuracyMs} ms")
                        if (timeSyncStatus.lastSyncTime > 0) {
                            val timeSinceSync = (System.currentTimeMillis() - timeSyncStatus.lastSyncTime) / 1000
                            InfoRow("Last sync", "${timeSinceSync} s ago")
                        }
                    }
                }
            }

            
            // Sensor info card
            item {
                InfoCard(
                    title = "Sensor Details",
                    icon = Icons.Default.Sensors
                ) {
                    sensorInfo.forEach { (sensorType, info) ->
                        Column(
                            modifier = Modifier.padding(vertical = 8.dp)
                        ) {
                            Text(
                                text = sensorType,
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.primary
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            InfoRow("Hardware max rate", "${decimalFormat.format(info.hardwareMaxSamplingRateHz)} Hz")
                            InfoRow("Current rate", "${decimalFormat.format(info.currentSamplingRateHz)} Hz")
                            InfoRow("Actual rate", "${decimalFormat.format(info.actualSamplingRateHz)} Hz")
                            InfoRow("FIFO size", "${info.fifoMaxEventCount}")
                            InfoRow("Vendor", info.vendor)
                            InfoRow("Power", "${decimalFormat.format(info.power)} mA")
                            HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
                        }
                    }
                }
            }
            
            // Device and keystore info card
            item {
                InfoCard(
                    title = "Device & Keystore",
                    icon = Icons.Default.Security
                ) {
                    InfoRow("Device model", deviceInfo.deviceModel)
                    InfoRow("Manufacturer", deviceInfo.deviceManufacturer)
                    InfoRow("Android version", deviceInfo.androidVersion)
                    InfoRow("Keystore Provider", deviceInfo.keystoreProvider)
                    InfoRow(
                        "StrongBox support",
                        if (deviceInfo.strongBoxSupported) "Yes" else "No",
                        textColor = if (deviceInfo.strongBoxSupported) 
                            Color(0xFF4CAF50) else MaterialTheme.colorScheme.onSurface
                    )
                    InfoRow("Current key version", deviceInfo.currentKeyVersion)
                    InfoRow("Key algorithm", deviceInfo.keyAlgorithm)
                    InfoRow(
                        "Key rotation",
                        if (deviceInfo.keyRotationScheduled) "Scheduled" else "Not scheduled"
                    )
                }
            }
            
            // Envelope encryption status card
            item {
                InfoCard(
                    title = "Envelope Encryption",
                    icon = Icons.Default.Lock
                ) {
                    val encryptionStatus = viewModel.encryptionStatus.collectAsStateWithLifecycle().value
                    
                    InfoRow("Scheme", "Envelope-AES256GCM")
                    InfoRow(
                        "Security lock",
                        if (encryptionStatus.isSecurityLocked) "Locked" else "Normal",
                        textColor = if (encryptionStatus.isSecurityLocked) 
                            MaterialTheme.colorScheme.error else Color(0xFF4CAF50)
                    )
                    if (encryptionStatus.consecutiveFailures > 0) {
                        InfoRow(
                            "Consecutive failures",
                            "${encryptionStatus.consecutiveFailures}/${encryptionStatus.maxFailuresThreshold}",
                            textColor = if (encryptionStatus.consecutiveFailures > encryptionStatus.maxFailuresThreshold / 2) 
                                Color(0xFFFF9800) else MaterialTheme.colorScheme.onSurface
                        )
                    }
                    InfoRow(
                        "Server public key",
                        if (encryptionStatus.hasServerPublicKey) "Configured" else "Not configured",
                        textColor = if (encryptionStatus.hasServerPublicKey) 
                            Color(0xFF4CAF50) else MaterialTheme.colorScheme.error
                    )
                    InfoRow("DEK key ID", encryptionStatus.currentDekKeyId)
                    InfoRow("Packet sequence", encryptionStatus.packetSequenceNumber.toString())
                    InfoRow("Key rotations", deviceInfo.keyRotationCount.toString())
                }
            }
            
            // Server policy card
            item {
                InfoCard(
                    title = "Server Policy",
                    icon = Icons.Default.Policy
                ) {
                    InfoRow("Version", serverPolicy.version)
                    InfoRow("Fast mode duration", "${serverPolicy.fastModeDurationSeconds} s")
                    InfoRow("Anomaly threshold", decimalFormat.format(serverPolicy.anomalyThreshold))
                    InfoRow("Transmission strategy", serverPolicy.transmissionStrategy)
                    
                    // Sampling rate info
                    Text(
                        text = "Sampling rates:",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(top = 8.dp)
                    )
                    serverPolicy.samplingRates.forEach { (sensor, rate) ->
                        InfoRow("  $sensor", "${decimalFormat.format(rate)} Hz")
                    }
                    
                    // JSON view (expandable)
                    var showJson by remember { mutableStateOf(false) }
                    TextButton(
                        onClick = { showJson = !showJson },
                        modifier = Modifier.padding(top = 8.dp)
                    ) {
                        Text(if (showJson) "Hide policy JSON" else "Show policy JSON")
                    }
                    
                    AnimatedVisibility(visible = showJson) {
                        Surface(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 8.dp),
                            color = MaterialTheme.colorScheme.surfaceVariant,
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Text(
                                text = serverPolicy.policyJson,
                                style = MaterialTheme.typography.bodySmall,
                                modifier = Modifier.padding(12.dp),
                                fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace
                            )
                        }
                    }
                }
            }
            
            // Performance charts card
            item {
                InfoCard(
                    title = "Performance",
                    icon = Icons.Default.Analytics
                ) {
                    // CPU usage
                    Text(
                        text = "App CPU usage (${decimalFormat.format(performanceMetrics.cpuUsagePercent)}%)",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold
                    )
                    if (cpuHistory.isNotEmpty()) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(100.dp)
                                .padding(vertical = 8.dp)
                        ) {
                            LineChart(
                                data = cpuHistory,
                                modifier = Modifier.fillMaxSize(),
                                lineColor = MaterialTheme.colorScheme.primary,
                                maxValue = 100f
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    // Memory usage
                    Text(
                        text = "App memory usage (${performanceMetrics.memoryUsedMB}MB / ${performanceMetrics.memoryTotalMB}MB)",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold
                    )
                    if (memoryHistory.isNotEmpty()) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(100.dp)
                                .padding(vertical = 8.dp)
                        ) {
                            LineChart(
                                data = memoryHistory,
                                modifier = Modifier.fillMaxSize(),
                                lineColor = MaterialTheme.colorScheme.secondary,
                                maxValue = 100f
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    // Upload latency
                    Text(
                        text = "Average upload latency (${performanceMetrics.averageUploadLatencyMs} ms)",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold
                    )
                    if (latencyHistory.isNotEmpty()) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(100.dp)
                                .padding(vertical = 8.dp)
                        ) {
                            LineChart(
                                data = latencyHistory.map { it.toFloat() },
                                modifier = Modifier.fillMaxSize(),
                                lineColor = MaterialTheme.colorScheme.tertiary
                            )
                        }
                    }
                    
                    HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp))
                    
                    // Other performance metrics
                    InfoRow("Battery", "${performanceMetrics.batteryLevel}%")
                    InfoRow("Temperature", "${decimalFormat.format(performanceMetrics.temperatureCelsius)}°C")
                }
            }
            
            // Manual actions card
            item {
                InfoCard(
                    title = "Actions",
                    icon = Icons.Default.TouchApp
                ) {
                    Column(
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        // Trigger fast mode
                        OutlinedButton(
                            onClick = { 
                                scope.launch {
                                    viewModel.triggerFastMode()
                                }
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(Icons.Default.Speed, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Trigger fast mode")
                        }
                        
                        // Clear local queue
                        OutlinedButton(
                            onClick = { 
                                scope.launch {
                                    viewModel.clearLocalQueue()
                                }
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(Icons.Default.DeleteSweep, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Clear local queue")
                        }
                        
                        // Export pending data
                        OutlinedButton(
                            onClick = { 
                                scope.launch {
                                    viewModel.exportPendingData()
                                }
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(Icons.Default.FileDownload, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Export pending data")
                        }
                        
                        // Force key rotation
                        OutlinedButton(
                            onClick = { 
                                scope.launch {
                                    viewModel.forceKeyRotation()
                                }
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(Icons.Default.VpnKey, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Force key rotation")
                        }
                        
                        // Update server public key
                        OutlinedButton(
                            onClick = { 
                                scope.launch {
                                    viewModel.updateServerPublicKey()
                                }
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(Icons.Default.PublishedWithChanges, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Update server public key")
                        }
                        
                        // Export debug logs
                        OutlinedButton(
                            onClick = { 
                                scope.launch {
                                    viewModel.exportDebugLog()
                                }
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(Icons.Default.Description, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Export debug logs")
                        }
                    }
                }
            }
        }
    }
}

/**
 * Info card component.
 */
@Composable
fun InfoCard(
    title: String,
    icon: ImageVector,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.padding(bottom = 12.dp)
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(12.dp))
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
            }
            
            content()
        }
    }
}

/**
 * Info row component.
 */
@Composable
fun InfoRow(
    label: String,
    value: String,
    textColor: Color = MaterialTheme.colorScheme.onSurface
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f)
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium,
            color = textColor,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.weight(1.5f)
        )
    }
}
