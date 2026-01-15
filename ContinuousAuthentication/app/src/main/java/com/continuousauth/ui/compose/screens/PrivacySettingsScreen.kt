package com.continuousauth.ui.compose.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.continuousauth.privacy.ConsentState
import com.continuousauth.privacy.DeletionState
import com.continuousauth.ui.MainViewModel
import kotlinx.coroutines.launch

/**
 * Privacy settings screen.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PrivacySettingsScreen(
    onNavigateBack: () -> Unit,
    viewModel: MainViewModel = hiltViewModel()
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    // Observe privacy-related state.
    val consentState by viewModel.consentState.observeAsState(initial = ConsentState.UNKNOWN)
    val deletionState by viewModel.deletionState.observeAsState(initial = DeletionState.IDLE)
    
    // Local state
    var dataRetentionDays by remember { mutableIntStateOf(30) }
    
    // Load retention policy on first composition.
    LaunchedEffect(Unit) {
        dataRetentionDays = viewModel.getDataRetentionDays()
    }
    // Toggles for optional cards.
    var showDataRetentionCard by remember { mutableStateOf(false) }
    var showTransmissionPolicyCard by remember { mutableStateOf(false) }
    var showConsentStatusCard by remember { mutableStateOf(false) }
    
    Box(
        modifier = Modifier
            .fillMaxSize()
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 8.dp)
        ) {
                // Conditionally show consent status.
                if (showConsentStatusCard) {
                    ConsentStatusCard(
                        consentState = consentState,
                        onGrantConsent = { viewModel.grantPrivacyConsent() }
                    )

                    Spacer(modifier = Modifier.height(16.dp))
                }

                // Conditionally show data retention settings.
                if (showDataRetentionCard) {
                    // Data retention settings.
                    DataRetentionCard(
                        retentionDays = dataRetentionDays,
                        onRetentionDaysChange = { days ->
                            dataRetentionDays = days
                            viewModel.setDataRetentionDays(days)
                        }
                    )

                    Spacer(modifier = Modifier.height(16.dp))
                }
                // Conditionally show transmission policy settings.
                if (showTransmissionPolicyCard) {
                    TransmissionPolicyCard(viewModel = viewModel)

                    Spacer(modifier = Modifier.height(16.dp))
                }
            }
            // Deletion progress indicator
            if (deletionState == DeletionState.IN_PROGRESS) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(16.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Card(
                        modifier = Modifier.padding(16.dp),
                        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
                    ) {
                        Column(
                            modifier = Modifier.padding(24.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            CircularProgressIndicator()
                            Spacer(modifier = Modifier.height(16.dp))
                            Text("Deleting data...", style = MaterialTheme.typography.bodyLarge)
                        }
                    }
                }
            }
        }
    }

/**
 * Consent status card.
 */
@Composable
private fun ConsentStatusCard(
    consentState: ConsentState,
    onGrantConsent: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = when (consentState) {
                ConsentState.GRANTED -> Color(0xFF4CAF50).copy(alpha = 0.1f)
                ConsentState.WITHDRAWN -> Color(0xFFFF5252).copy(alpha = 0.1f)
                else -> MaterialTheme.colorScheme.surfaceVariant
            }
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = when (consentState) {
                        ConsentState.GRANTED -> Icons.Default.CheckCircle
                        ConsentState.WITHDRAWN -> Icons.Default.Cancel
                        else -> Icons.Default.Info
                    },
                    contentDescription = null,
                    tint = when (consentState) {
                        ConsentState.GRANTED -> Color(0xFF4CAF50)
                        ConsentState.WITHDRAWN -> Color(0xFFFF5252)
                        else -> MaterialTheme.colorScheme.primary
                    },
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Consent status",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Text(
                text = when (consentState) {
                    ConsentState.GRANTED -> "Consent granted for data collection and use."
                    ConsentState.WITHDRAWN -> "Consent withdrawn; data deleted."
                    ConsentState.NOT_GRANTED -> "Consent not granted."
                    else -> "Consent status unknown."
                },
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            
            if (consentState == ConsentState.NOT_GRANTED) {
                Spacer(modifier = Modifier.height(8.dp))
                Button(
                    onClick = onGrantConsent,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Grant consent")
                }
            }
        }
    }
}

/**
 * Data retention settings card.
 */
@Composable
private fun DataRetentionCard(
    retentionDays: Int,
    onRetentionDaysChange: (Int) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Data retention",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Text(
                text = "Cached data will be deleted after $retentionDays days.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Slider(
                value = retentionDays.toFloat(),
                onValueChange = { onRetentionDaysChange(it.toInt()) },
                valueRange = 1f..365f,
                steps = 29  // 30 steps across the range
            )
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("1 day", style = MaterialTheme.typography.bodySmall)
                Text("$retentionDays days", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Bold)
                Text("365 days", style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

/**
 * Transmission policy card.
 */
@Composable
private fun TransmissionPolicyCard(viewModel: MainViewModel) {
    // Observe upload policy state.
    val wifiOnly by viewModel.uploadPolicyWiFiOnly.observeAsState(initial = false)
    
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Upload policy",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Upload over Wi-Fi only",
                        style = MaterialTheme.typography.bodyLarge
                    )
                    Text(
                        text = if (wifiOnly) "Enabled (uploads only on Wi-Fi)" else "Disabled (uploads on any network)",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Switch(
                    checked = wifiOnly,
                    onCheckedChange = { isChecked ->
                        viewModel.setUploadPolicyWiFiOnly(isChecked)
                    }
                )
            }
        }
    }
}

