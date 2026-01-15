package com.continuousauth.ui.viewmodels

import android.content.Context
import android.widget.Toast
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.continuousauth.monitor.SystemMonitor
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * ViewModel for the detailed information screen.
 *
 * Uses {@link SystemMonitor} to provide system monitoring data.
 */
@HiltViewModel
class DetailedInfoViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val systemMonitor: SystemMonitor
) : ViewModel() {
    
    // Transmission status
    val transmissionStatus: StateFlow<SystemMonitor.TransmissionStatus> = 
        systemMonitor.transmissionStatus.stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = SystemMonitor.TransmissionStatus()
        )
    
    // gRPC status
    val grpcStatus: StateFlow<SystemMonitor.GrpcConnectionStatus> = 
        systemMonitor.grpcStatus.stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = SystemMonitor.GrpcConnectionStatus()
        )
    
    // Buffer statistics
    val bufferStats: StateFlow<SystemMonitor.BufferStatistics> = 
        systemMonitor.bufferStats.stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = SystemMonitor.BufferStatistics()
        )
    
    // Sensor information
    val sensorInfo: StateFlow<Map<String, SystemMonitor.SensorDetailedInfo>> = 
        systemMonitor.sensorInfoMap.stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyMap()
        )
    
    // Device and key information
    val deviceInfo: StateFlow<SystemMonitor.DeviceKeyInfo> = 
        systemMonitor.deviceKeyInfo.stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = SystemMonitor.DeviceKeyInfo()
        )
    
    // Server policy
    val serverPolicy: StateFlow<SystemMonitor.ServerPolicy> = 
        systemMonitor.serverPolicy.stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = SystemMonitor.ServerPolicy()
        )
    
    // Time sync status
    val timeSyncStatus: StateFlow<SystemMonitor.TimeSyncStatus> = 
        systemMonitor.timeSyncStatus.stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = SystemMonitor.TimeSyncStatus()
        )
    
    // Performance metrics
    val performanceMetrics: StateFlow<SystemMonitor.PerformanceMetrics> = 
        systemMonitor.performanceMetrics.stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = SystemMonitor.PerformanceMetrics()
        )
    
    // Performance history
    val cpuHistory: StateFlow<List<Float>> = 
        systemMonitor.cpuHistory.stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )
    
    val memoryHistory: StateFlow<List<Float>> = 
        systemMonitor.memoryHistory.stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )
    
    val latencyHistory: StateFlow<List<Long>> = 
        systemMonitor.latencyHistory.stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )
    
    // Envelope encryption status
    val encryptionStatus: StateFlow<SystemMonitor.EncryptionStatus> = 
        systemMonitor.encryptionStatus.stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = SystemMonitor.EncryptionStatus()
        )
    
    init {
        // Start system monitoring.
        systemMonitor.startMonitoring()
    }
    
    /**
     * Triggers fast mode (feature removed).
     */
    suspend fun triggerFastMode() {
        // Fast mode is no longer part of the spec; keep the action disabled.
        showToast("Fast mode has been removed.")
    }
    
    /**
     * Clears the local queue.
     */
    suspend fun clearLocalQueue() {
        try {
            systemMonitor.clearLocalQueue()
            showToast("Local queue cleared.")
        } catch (e: Exception) {
            showToast("Failed to clear queue: ${e.message}")
        }
    }
    
    /**
     * Exports pending data.
     */
    suspend fun exportPendingData() {
        try {
            val path = systemMonitor.exportPendingData()
            showToast("Data exported to: $path")
        } catch (e: Exception) {
            showToast("Failed to export data: ${e.message}")
        }
    }
    
    /**
     * Forces key rotation.
     */
    suspend fun forceKeyRotation() {
        try {
            systemMonitor.forceKeyRotation()
            showToast("Key rotation succeeded.")
        } catch (e: Exception) {
            showToast("Key rotation failed: ${e.message}")
        }
    }
    
    /**
     * Updates the server public key.
     */
    suspend fun updateServerPublicKey() {
        try {
            // In production this should fetch new key material from a trusted source.
            val sampleKeyData = "SAMPLE_PUBLIC_KEY_DATA".toByteArray()
            systemMonitor.updateServerPublicKey(sampleKeyData)
            showToast("Server public key updated.")
        } catch (e: Exception) {
            showToast("Failed to update public key: ${e.message}")
        }
    }
    
    /**
     * Exports debug logs.
     */
    suspend fun exportDebugLog() {
        try {
            val path = systemMonitor.exportDebugLog()
            showToast("Debug logs exported to: $path")
        } catch (e: Exception) {
            showToast("Failed to export logs: ${e.message}")
        }
    }
    
    /**
     * Shows a toast message.
     */
    private fun showToast(message: String) {
        viewModelScope.launch {
            Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
        }
    }
    
    override fun onCleared() {
        super.onCleared()
        // Stop monitoring.
        systemMonitor.stopMonitoring()
    }
}
