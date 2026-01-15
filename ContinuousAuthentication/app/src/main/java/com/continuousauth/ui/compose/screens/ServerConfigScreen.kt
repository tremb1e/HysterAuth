package com.continuousauth.ui.compose.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.getValue
import androidx.compose.ui.res.stringResource
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import com.continuousauth.R
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.continuousauth.ui.MainViewModel
import com.continuousauth.ui.theme.ExtendedColors
import com.continuousauth.network.ConnectionStatus
import com.continuousauth.privacy.ConsentState
import com.continuousauth.privacy.DeletionState
import com.continuousauth.storage.QueueStats
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Server configuration screen.
 *
 * Server settings and connection management UI.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ServerConfigScreen(viewModel: MainViewModel) {
    val scrollState = rememberScrollState()
    val scope = rememberCoroutineScope()
    val clipboardManager = LocalClipboardManager.current
    
    // Observe ViewModel state.
    val connectionStatus by viewModel.connectionStatus.observeAsState(ConnectionStatus.DISCONNECTED)
    val isCollectionRunning by viewModel.isCollectionRunning.observeAsState(false)
    val isEncryptedUploading by viewModel.isEncryptedUploading.observeAsState(false)
    val userId by viewModel.userId.observeAsState("")
    val sessionId by viewModel.sessionId.observeAsState(null)
    val sessionStartTime by viewModel.sessionStartTime.observeAsState(0L)
    val sessionDuration by viewModel.sessionDuration.observeAsState("00:00")
    val serverTestResult by viewModel.serverTestResult.observeAsState(null)
    val transmissionStats by viewModel.transmissionStats.observeAsState(null)
    val fileQueueStats by viewModel.fileQueueStats.observeAsState(null)
    
    // Local state
    var serverIp by remember { mutableStateOf("192.168.1.100") }
    var serverPort by remember { mutableStateOf("50051") }
    var isTestingConnection by remember { mutableStateOf(false) }

    // Observe privacy state.
    val consentState by viewModel.consentState.observeAsState(initial = ConsentState.UNKNOWN)
    val deletionState by viewModel.deletionState.observeAsState(initial = DeletionState.IDLE)
    // Local state
    var showWithdrawDialog by remember { mutableStateOf(false) }
    var showPrivacyPolicy by remember { mutableStateOf(false) }
    var dataRetentionDays by remember { mutableIntStateOf(30) }

    val transmissionStatus by viewModel.transmissionStatus.collectAsStateWithLifecycle()
    val timeSyncStatus by viewModel.timeSyncStatus.collectAsStateWithLifecycle()
    // Toggle NTP time sync card visibility.
    var showNtpSyncCard by remember { mutableStateOf(true) }
    // Toggle transmission status card visibility.
    var showTransmissionStatusCard by remember { mutableStateOf(true) }
    // Withdraw consent confirmation dialog
    if (showWithdrawDialog) {
        WithdrawConsentDialog(
            onConfirm = {
                scope.launch {
                    viewModel.withdrawConsentAndDeleteData()
                }
                showWithdrawDialog = false
            },
            onDismiss = { showWithdrawDialog = false }
        )
    }

    // Privacy policy dialog
    if (showPrivacyPolicy) {
        PrivacyPolicyDialog(
            onDismiss = { showPrivacyPolicy = false }
        )
    }
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
            // Connection status card
            ConnectionStatusCard(connectionStatus, serverIp, serverPort)
            
            // Identification card (shows user ID only)
            IdentificationCard(
                userId = userId,
                onCopyUserId = {
                    clipboardManager.setText(AnnotatedString(userId))
                }
            )
            
            // Server settings card
            ServerSettingsCard(
                serverIp = serverIp,
                serverPort = serverPort,
                onIpChange = { serverIp = it },
                onPortChange = { serverPort = it },
                isTestingConnection = isTestingConnection,
                serverTestResult = serverTestResult,
                onTestConnection = {
                    isTestingConnection = true
                    viewModel.testServerConnection(serverIp, serverPort)
                    scope.launch {
                        delay(3500) // Wait for the test to complete
                        isTestingConnection = false
                    }
                }
            )
            
            // Encrypted upload control card
            EncryptedUploadControlCard(
                isEncryptedUploading = isEncryptedUploading,
                connectionStatus = connectionStatus,
                sessionId = sessionId,
                sessionStartTime = sessionStartTime,
                sessionDuration = sessionDuration,
                transmissionStats = transmissionStats,
                fileQueueStats = fileQueueStats,
                onToggleUpload = {
                    if (isEncryptedUploading) {
                        viewModel.stopEncryptedUpload()
                    } else {
                        viewModel.startEncryptedUpload()
                    }
                }
            )
            
            // File queue stats card
            fileQueueStats?.let { stats ->
                FileQueueCard(
                    queueStats = stats,
                    onClearQueue = { viewModel.clearFileQueue() }
                )
            }
            // Data management card
            DataManagementCard(
                onWithdrawConsent = { showWithdrawDialog = true },
                onViewPrivacyPolicy = { showPrivacyPolicy = true },
                isDeleting = deletionState == DeletionState.IN_PROGRESS
            )
            // Encryption status card
            EncryptionStatusCard()
            if(showTransmissionStatusCard){
                // Transmission status card
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

            // Conditionally show NTP time sync card
            if (showNtpSyncCard) {
                // NTP time sync card
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
    }
}

/**
 * Encryption status card.
 */
@Composable
private fun EncryptionStatusCard() {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Default.Lock,
                    contentDescription = null,
                    tint = Color(0xFF4CAF50),
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Encryption",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            PrivacyInfoRow("Algorithm", "AES-256-GCM with StreamingAEAD")
            PrivacyInfoRow("Key management", "Android Keystore (Tink)")
            PrivacyInfoRow("Transport security", "TLS 1.3 + certificate pinning")
            PrivacyInfoRow("Compression", "GZIP (compress then encrypt)")
        }
    }
}
/**
 * Info row.
 */
@Composable
private fun PrivacyInfoRow(label: String, value: String) {
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
 * Connection status card.
 */
@Composable
fun ConnectionStatusCard(
    connectionStatus: ConnectionStatus,
    serverIp: String,
    serverPort: String
) {
    val statusColor = when (connectionStatus) {
        ConnectionStatus.CONNECTED -> ExtendedColors.success
        ConnectionStatus.CONNECTING, ConnectionStatus.RECONNECTING -> ExtendedColors.warning
        ConnectionStatus.DISCONNECTED -> ExtendedColors.error
        else -> Color.Gray
    }
    
    val context = LocalContext.current
    val statusText = when (connectionStatus) {
        ConnectionStatus.CONNECTED -> context.getString(R.string.connected)
        ConnectionStatus.CONNECTING -> context.getString(R.string.connecting)
        ConnectionStatus.RECONNECTING -> context.getString(R.string.reconnecting)
        ConnectionStatus.DISCONNECTED -> context.getString(R.string.disconnected)
        else -> context.getString(R.string.not_available)
    }
    
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(
            containerColor = statusColor.copy(alpha = 0.1f)
        )
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.linearGradient(
                        colors = listOf(
                            statusColor.copy(alpha = 0.05f),
                            statusColor.copy(alpha = 0.15f)
                        )
                    )
                )
                .padding(24.dp)
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.fillMaxWidth()
            ) {
                // Animated connection icon
                AnimatedConnectionIcon(
                    isConnected = connectionStatus == ConnectionStatus.CONNECTED,
                    color = statusColor
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                Text(
                    text = statusText,
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                    color = statusColor
                )
                
                if (connectionStatus == ConnectionStatus.CONNECTED) {
                    Text(
                        text = "$serverIp:$serverPort",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                        modifier = Modifier.padding(top = 4.dp)
                    )
                }
            }
        }
    }
}

/**
 * Animated connection icon.
 */
@Composable
fun AnimatedConnectionIcon(
    isConnected: Boolean,
    color: Color
) {
    val infiniteTransition = rememberInfiniteTransition(label = "connection")
    
    if (isConnected) {
        val scale by infiniteTransition.animateFloat(
            initialValue = 0.8f,
            targetValue = 1.2f,
            animationSpec = infiniteRepeatable(
                animation = tween(1500),
                repeatMode = RepeatMode.Reverse
            ),
            label = "scale"
        )
        
        Box(contentAlignment = Alignment.Center) {
            repeat(3) { index ->
                Box(
                    modifier = Modifier
                        .size(60.dp + (index * 20).dp)
                        .scale(scale)
                        .clip(CircleShape)
                        .background(color.copy(alpha = 0.1f - index * 0.03f))
                )
            }
            
            Icon(
                imageVector = Icons.Filled.CloudDone,
                contentDescription = null,
                modifier = Modifier.size(48.dp),
                tint = color
            )
        }
    } else {
        Icon(
            imageVector = Icons.Filled.CloudOff,
            contentDescription = null,
            modifier = Modifier.size(48.dp),
            tint = color
        )
    }
}

/**
 * Identification card (shows user ID only).
 */
@Composable
fun IdentificationCard(
    userId: String,
    onCopyUserId: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = stringResource(R.string.identity_label),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
            
            // User ID
            IdRow(
                icon = Icons.Outlined.Person,
                label = stringResource(R.string.user_id_label),
                value = userId,
                onCopy = onCopyUserId
            )
        }
    }
}

/**
 * ID row component.
 */
@OptIn(ExperimentalAnimationApi::class)
@Composable
fun IdRow(
    icon: ImageVector,
    label: String,
    value: String,
    onCopy: () -> Unit
) {
    var copied by remember { mutableStateOf(false) }
    
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(20.dp)
        )
        
        Spacer(modifier = Modifier.width(12.dp))
        
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = value,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium
            )
        }
        
        IconButton(
            onClick = {
                onCopy()
                copied = true
            }
        ) {
            AnimatedContent(
                targetState = copied,
                transitionSpec = {
                    scaleIn() + fadeIn() with scaleOut() + fadeOut()
                },
                label = "copy"
            ) { isCopied ->
                Icon(
                    imageVector = if (isCopied) Icons.Filled.Check else Icons.Outlined.ContentCopy,
                    contentDescription = stringResource(R.string.copy),
                    tint = if (isCopied) ExtendedColors.success else MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
        
        LaunchedEffect(copied) {
            if (copied) {
                delay(2000)
                copied = false
            }
        }
    }
}

/**
 * Server settings card.
 */
@Composable
fun ServerSettingsCard(
    serverIp: String,
    serverPort: String,
    onIpChange: (String) -> Unit,
    onPortChange: (String) -> Unit,
    isTestingConnection: Boolean,
    serverTestResult: String?,
    onTestConnection: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Text(
                text = stringResource(R.string.server_settings),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
            
            // IP address input
            OutlinedTextField(
                value = serverIp,
                onValueChange = onIpChange,
                label = { Text(stringResource(R.string.server_ip_label)) },
                leadingIcon = {
                    Icon(Icons.Outlined.Computer, contentDescription = null)
                },
                keyboardOptions = KeyboardOptions(
                    keyboardType = KeyboardType.Number,
                    imeAction = ImeAction.Next
                ),
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                singleLine = true
            )
            
            // Port input
            OutlinedTextField(
                value = serverPort,
                onValueChange = onPortChange,
                label = { Text(stringResource(R.string.port_label)) },
                leadingIcon = {
                    Icon(Icons.Outlined.Router, contentDescription = null)
                },
                keyboardOptions = KeyboardOptions(
                    keyboardType = KeyboardType.Number,
                    imeAction = ImeAction.Done
                ),
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                singleLine = true
            )
            
            // Test connection button
            Button(
                onClick = onTestConnection,
                modifier = Modifier.fillMaxWidth(),
                enabled = !isTestingConnection,
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.secondary
                )
            ) {
                if (isTestingConnection) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onSecondary
                    )
                } else {
                    Icon(
                        imageVector = Icons.Outlined.NetworkCheck,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp)
                    )
                }
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = if (isTestingConnection) stringResource(R.string.testing_server) else stringResource(R.string.detect_server),
                    style = MaterialTheme.typography.labelLarge
                )
            }
            
            // Test result
            serverTestResult?.let { result ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 8.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = if (result.startsWith("✓")) 
                            ExtendedColors.success.copy(alpha = 0.1f)
                        else 
                            ExtendedColors.error.copy(alpha = 0.1f)
                    ),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(
                        text = result,
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(12.dp),
                        color = if (result.startsWith("✓")) 
                            ExtendedColors.success
                        else 
                            ExtendedColors.error
                    )
                }
            }
        }
    }
}

/**
 * Encrypted upload control card.
 */
@Composable
fun EncryptedUploadControlCard(
    isEncryptedUploading: Boolean,
    connectionStatus: ConnectionStatus,
    sessionId: String?,
    sessionStartTime: Long,
    sessionDuration: String,
    transmissionStats: com.continuousauth.ui.TransmissionStats?,
    fileQueueStats: QueueStats?,
    onToggleUpload: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (isEncryptedUploading) 
                ExtendedColors.success.copy(alpha = 0.1f)
            else 
                MaterialTheme.colorScheme.surface
        )
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Filled.Lock,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                // Encrypted data upload
                Text(
                    text = stringResource(R.string.encrypted_data_upload),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )
            }
            
            // Session info and upload status
            if (isEncryptedUploading && sessionId != null) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                    ),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = stringResource(R.string.current_session_info),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Medium
                        )
                        
                        // Session ID
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = "Session ID:",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Text(
                                text = sessionId.take(8) + "...",
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.Medium
                            )
                        }
                        
                        // Start time
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = stringResource(R.string.start_time_label),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Text(
                                text = if (sessionStartTime > 0) {
                                    val dateFormat = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.getDefault())
                                    dateFormat.format(java.util.Date(sessionStartTime))
                                } else "--:--:--",
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.Medium
                            )
                        }
                        
                        // Elapsed collection time
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = stringResource(R.string.collection_duration_label),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Text(
                                text = sessionDuration,
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.Medium,
                                color = ExtendedColors.success
                            )
                        }
                        
                        // Divider
                        Divider(
                            modifier = Modifier.padding(vertical = 4.dp),
                            color = MaterialTheme.colorScheme.outlineVariant
                        )
                        
                        // Transmission status
                        Text(
                            text = "Encrypted upload status",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Medium,
                            color = MaterialTheme.colorScheme.primary
                        )
                        
                        // Live upload stats
                        transmissionStats?.let { stats ->
                            // Sent packets
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    text = "Sent:",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Text(
                                    text = "${stats.packetsSent} packets",
                                    style = MaterialTheme.typography.bodySmall,
                                    fontWeight = FontWeight.Medium,
                                    color = ExtendedColors.success
                                )
                            }
                            
                            // Pending packets
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    text = "Pending:",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Text(
                                    text = "${stats.packetsPending} packets",
                                    style = MaterialTheme.typography.bodySmall,
                                    fontWeight = FontWeight.Medium,
                                    color = if (stats.packetsPending > 0) ExtendedColors.warning else MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            
                            // Latest ACK latency
                            stats.lastAckLatency?.let { latency ->
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        text = "Latency:",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                    Text(
                                        text = "$latency ms",
                                        style = MaterialTheme.typography.bodySmall,
                                        fontWeight = FontWeight.Medium,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            }
                            
                            // Fast mode state
                            if (stats.isFastMode) {
                                Spacer(modifier = Modifier.height(4.dp))
                                Card(
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = CardDefaults.cardColors(
                                        containerColor = ExtendedColors.warning.copy(alpha = 0.2f)
                                    ),
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Row(
                                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Filled.Speed,
                                            contentDescription = null,
                                            tint = ExtendedColors.warning,
                                            modifier = Modifier.size(16.dp)
                                        )
                                        Text(
                                            text = "Fast mode enabled",
                                            style = MaterialTheme.typography.bodySmall,
                                            fontWeight = FontWeight.Bold,
                                            color = ExtendedColors.warning
                                        )
                                    }
                                }
                            }
                        }
                        
                        // File queue state
                        fileQueueStats?.let { queueStats ->
                            Spacer(modifier = Modifier.height(4.dp))
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    text = "Upload queue:",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Text(
                                    text = "${queueStats.pendingPackets}/${queueStats.totalPackets} (${formatBytes(queueStats.totalSizeBytes)})",
                                    style = MaterialTheme.typography.bodySmall,
                                    fontWeight = FontWeight.Medium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            
                            // Upload progress
                            if (queueStats.totalPackets > 0) {
                                val progress = queueStats.uploadedPackets.toFloat() / queueStats.totalPackets
                                LinearProgressIndicator(
                                    progress = { progress },
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(4.dp)
                                        .clip(RoundedCornerShape(2.dp)),
                                    color = ExtendedColors.success,
                                    trackColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                                )
                            }
                        }
                        
                        // Encryption status hint
                        Spacer(modifier = Modifier.height(4.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.Center,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                imageVector = Icons.Filled.Lock,
                                contentDescription = null,
                                tint = ExtendedColors.success,
                                modifier = Modifier.size(14.dp)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = "Data is encrypted in transit",
                                style = MaterialTheme.typography.labelSmall,
                                color = ExtendedColors.success,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }
                }
            }
            
            // Start/stop buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Start button
                Button(
                    onClick = {
                        if (!isEncryptedUploading) {
                            onToggleUpload()
                        }
                    },
                    modifier = Modifier
                        .weight(1f)
                        .height(56.dp),
                    enabled = !isEncryptedUploading && connectionStatus == ConnectionStatus.CONNECTED,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = ExtendedColors.success
                    ),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Icon(
                        imageVector = Icons.Filled.PlayArrow,
                        contentDescription = null,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = stringResource(R.string.start_button),
                        style = MaterialTheme.typography.labelLarge,
                        fontSize = 16.sp
                    )
                }
                
                // Stop button
                Button(
                    onClick = {
                        if (isEncryptedUploading) {
                            onToggleUpload()
                        }
                    },
                    modifier = Modifier
                        .weight(1f)
                        .height(56.dp),
                    enabled = isEncryptedUploading,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = ExtendedColors.error
                    ),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Icon(
                        imageVector = Icons.Filled.Stop,
                        contentDescription = null,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = stringResource(R.string.stop_button),
                        style = MaterialTheme.typography.labelLarge,
                        fontSize = 16.sp
                    )
                }
            }
        }
    }
}

/**
 * Status chip.
 */
@Composable
fun StatusChip(
    icon: ImageVector,
    label: String,
    value: String,
    color: Color
) {
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = color.copy(alpha = 0.1f)
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = color,
                modifier = Modifier.size(20.dp)
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = value,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Bold,
                color = color
            )
        }
    }
}

/**
 * File queue stats card.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FileQueueCard(
    queueStats: QueueStats,
    onClearQueue: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        shape = RoundedCornerShape(20.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier
                .padding(20.dp)
                .fillMaxWidth()
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.FolderOpen,
                        contentDescription = stringResource(R.string. file_queue),
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(24.dp)
                    )
                    Text(
                        text = stringResource(R.string.file_queue),
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                }
                
                // Clear button
                if (queueStats.totalPackets > 0) {
                    TextButton(
                        onClick = onClearQueue,
                        colors = ButtonDefaults.textButtonColors(
                            contentColor = MaterialTheme.colorScheme.error
                        )
                    ) {
                        Icon(
                            imageVector = Icons.Default.Delete,
                            contentDescription = stringResource(R.string.clear_queue),
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(stringResource(R.string.clear_queue))
                    }
                }
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Stats grid
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                // Total packets
                StatusChip(
                    icon = Icons.Default.Inventory,
                    label = stringResource(R.string.total_packets),
                    value = queueStats.totalPackets.toString(),
                    color = MaterialTheme.colorScheme.primary
                )
                // Pending
                StatusChip(
                    icon = Icons.Default.Schedule,
                    label = stringResource(R.string.pending_upload),
                    value = queueStats.pendingPackets.toString(),
                    color = if (queueStats.pendingPackets > 0) ExtendedColors.warning else Color.Gray
                )
                // Uploaded
                StatusChip(
                    icon = Icons.Default.CloudDone,
                    label = stringResource(R.string.uploaded),
                    value = queueStats.uploadedPackets.toString(),
                    color = ExtendedColors.success
                )
            }
            
            Spacer(modifier = Modifier.height(12.dp))
            
            // Storage info
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                StatusChip(
                    icon = Icons.Default.Storage,
                    label = stringResource(R.string.queue_size_label),
                    value = formatBytes(queueStats.totalSizeBytes),
                    color = MaterialTheme.colorScheme.secondary
                )
                StatusChip(
                    icon = Icons.Default.BrokenImage,
                    label = stringResource(R.string.corrupted_packets),
                    value = queueStats.corruptedPackets.toString(),
                    color = if (queueStats.corruptedPackets > 0) ExtendedColors.error else Color.Gray
                )
            }
            
            // Progress
            if (queueStats.totalPackets > 0) {
                Spacer(modifier = Modifier.height(12.dp))
                
                val progress = if (queueStats.totalPackets > 0) {
                    queueStats.uploadedPackets.toFloat() / queueStats.totalPackets
                } else 0f
                
                LinearProgressIndicator(
                    progress = { progress },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(6.dp)
                        .clip(RoundedCornerShape(3.dp)),
                    color = ExtendedColors.success,
                    trackColor = MaterialTheme.colorScheme.surfaceVariant
                )
                
                Text(
                    text = stringResource(R.string.upload_progress_format, (progress * 100).toInt()),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }
    }
}
/**
 * Data management card.
 */
@Composable
private fun DataManagementCard(
    onWithdrawConsent: () -> Unit,
    onViewPrivacyPolicy: () -> Unit,
    isDeleting: Boolean
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Data management",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Withdraw consent button
            OutlinedButton(
                onClick = onWithdrawConsent,
                modifier = Modifier.fillMaxWidth(),
                enabled = !isDeleting,
                colors = ButtonDefaults.outlinedButtonColors(
                    contentColor = Color(0xFFFF5252)
                )
            ) {
                Icon(Icons.Default.Delete, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Withdraw consent and delete data")
            }

            Spacer(modifier = Modifier.height(8.dp))

            // View privacy policy button
            OutlinedButton(
                onClick = onViewPrivacyPolicy,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(Icons.Default.Policy, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("View privacy policy")
            }
        }
    }
}

/**
 * Privacy policy dialog.
 */
@Composable
private fun PrivacyPolicyDialog(
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text("Privacy policy")
        },
        text = {
            Column(
                modifier = Modifier.verticalScroll(rememberScrollState())
            ) {
                Text(
                    text = """
                        Data Collection and Use
                        
                        1. Data collected
                        • Sensor data: accelerometer, gyroscope, magnetometer
                        • Device info: device model, OS version
                        • App usage: foreground app information (encrypted)
                        
                        2. Purpose
                        • Continuous authentication research
                        • Improve authentication accuracy
                        • Academic analysis
                        
                        3. Protection
                        • All data is encrypted using AES-256-GCM
                        • HMAC is used to protect sensitive identifiers
                        • Local cache is cleaned up automatically
                        • Secure storage on the server
                        
                        4. Sharing
                        • Raw data is not shared with third parties
                        • Only aggregated statistics may be shared
                        • Data minimization principles are followed
                        
                        5. Your rights
                        • You may withdraw consent at any time
                        • Related data will be deleted after withdrawal
                        • Data export requests are supported
                        • Data correction requests are supported
                        
                        6. Retention
                        • Local cache: configurable from 1–365 days
                        • Server: retained per research protocol
                        • Expired data is cleaned automatically
                        
                        7. Contact
                        Email: privacy@continuousauth.com
                        
                        Last updated: Jan 2024
                    """.trimIndent(),
                    style = MaterialTheme.typography.bodySmall
                )
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("Close")
            }
        }
    )
}
/**
 * Withdraw consent confirmation dialog.
 */
@Composable
private fun WithdrawConsentDialog(
    onConfirm: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        icon = {
            Icon(
                imageVector = Icons.Default.Warning,
                contentDescription = null,
                tint = Color(0xFFFF5252)
            )
        },
        title = {
            Text("Confirm consent withdrawal")
        },
        text = {
            Column {
                Text(
                    text = "Withdrawing consent will:",
                    style = MaterialTheme.typography.bodyLarge,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text("• Stop all data collection immediately")
                Text("• Delete all locally cached data")
                Text("• Send a deletion request to the server")
                Text("• Clear personal information")
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = "This action cannot be undone!",
                    color = Color(0xFFFF5252),
                    fontWeight = FontWeight.Bold
                )
            }
        },
        confirmButton = {
            TextButton(
                onClick = onConfirm,
                colors = ButtonDefaults.textButtonColors(
                    contentColor = Color(0xFFFF5252)
                )
            ) {
                Text("Delete")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}
/**
 * Formats bytes into a human-readable string.
 */
private fun formatBytes(bytes: Long): String {
    return when {
        bytes < 1024 -> "$bytes B"
        bytes < 1024 * 1024 -> String.format("%.1f KB", bytes / 1024.0)
        bytes < 1024 * 1024 * 1024 -> String.format("%.1f MB", bytes / (1024.0 * 1024.0))
        else -> String.format("%.1f GB", bytes / (1024.0 * 1024.0 * 1024.0))
    }
}
