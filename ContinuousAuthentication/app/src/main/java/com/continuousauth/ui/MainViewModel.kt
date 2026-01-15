package com.continuousauth.ui

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorManager
import android.os.Build
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.continuousauth.monitor.MemoryMonitor
import com.continuousauth.monitor.SystemMonitor
import com.continuousauth.network.ConnectionStatus
import com.continuousauth.network.NetworkEnvironmentDetector
import com.continuousauth.network.NetworkState
import com.continuousauth.network.UploadManager
import com.continuousauth.observability.MetricsCollectorImpl
import com.continuousauth.observability.PerformanceMonitorImpl
import com.continuousauth.pool.SensorEventPool
import com.continuousauth.utils.UserIdManager
import com.continuousauth.network.ServerConnectionTester
import com.continuousauth.network.TlsSecurityManager
import com.continuousauth.network.TlsConfigInfo
import com.continuousauth.storage.FileQueueManager
import com.continuousauth.storage.QueueStats
import com.continuousauth.privacy.PrivacyManager
import com.continuousauth.privacy.ConsentState
import com.continuousauth.privacy.DeletionState
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.security.KeyStore
import javax.inject.Inject

/**
 * Main screen ViewModel.
 *
 * Manages UI state and business logic.
 */
@HiltViewModel
class MainViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val uploadManager: UploadManager,
    private val networkEnvironmentDetector: NetworkEnvironmentDetector,
    private val metricsCollector: MetricsCollectorImpl,
    private val performanceMonitor: PerformanceMonitorImpl,
    private val memoryMonitor: MemoryMonitor,
    private val sensorEventPool: SensorEventPool,
    private val userIdManager: UserIdManager,
    private val serverConnectionTester: ServerConnectionTester,
    private val fileQueueManager: FileQueueManager,
    private val tlsSecurityManager: TlsSecurityManager,
    private val privacyManager: PrivacyManager,
    private val systemMonitor: SystemMonitor
) : ViewModel() {
    
    companion object {
        private const val TAG = "MainViewModel"
    }
    
    // Collection status
    private val _collectionStatus = MutableLiveData<String>()
    val collectionStatus: LiveData<String> = _collectionStatus
    
    // Collection running state
    private val _isCollectionRunning = MutableLiveData<Boolean>()
    val isCollectionRunning: LiveData<Boolean> = _isCollectionRunning
    
    // Encrypted upload state: true = uploading encrypted data, false = not uploading encrypted data.
    private val _isEncryptedUploading = MutableLiveData<Boolean>()
    val isEncryptedUploading: LiveData<Boolean> = _isEncryptedUploading
    
    // Sensor status
    private val _sensorStatus = MutableLiveData<Map<String, Boolean>>()
    val sensorStatus: LiveData<Map<String, Boolean>> = _sensorStatus
    
    // Network state
    private val _networkState = MutableLiveData<NetworkState>()
    val networkState: LiveData<NetworkState> = _networkState
    
    // Connection status
    private val _connectionStatus = MutableLiveData<ConnectionStatus>()
    val connectionStatus: LiveData<ConnectionStatus> = _connectionStatus
    
    // Transmission stats
    private val _transmissionStats = MutableLiveData<TransmissionStats>()
    val transmissionStats: LiveData<TransmissionStats> = _transmissionStats
    
    // Debug mode state
    private val _debugModeEnabled = MutableLiveData<Boolean>()
    val debugModeEnabled: LiveData<Boolean> = _debugModeEnabled
    
    // Visualization state
    private val _visualizationEnabled = MutableLiveData<Boolean>()
    val visualizationEnabled: LiveData<Boolean> = _visualizationEnabled
    
    // Error message
    private val _errorMessage = MutableLiveData<String?>()
    val errorMessage: LiveData<String?> = _errorMessage
    
    // Debug info
    private val _debugInfo = MutableLiveData<String>()
    val debugInfo: LiveData<String> = _debugInfo
    
    // User ID
    private val _userId = MutableLiveData<String>()
    val userId: LiveData<String> = _userId
    
    // File queue stats
    private val _fileQueueStats = MutableLiveData<QueueStats>()
    val fileQueueStats: LiveData<QueueStats> = _fileQueueStats
    
    // Session info
    private val _sessionId = MutableLiveData<String?>()
    val sessionId: LiveData<String?> = _sessionId
    
    private val _sessionStartTime = MutableLiveData<Long>()
    val sessionStartTime: LiveData<Long> = _sessionStartTime
    
    private val _sessionDuration = MutableLiveData<String>()
    val sessionDuration: LiveData<String> = _sessionDuration
    
    // Server test result
    private val _serverTestResult = MutableLiveData<String?>()
    val serverTestResult: LiveData<String?> = _serverTestResult
    
    // TLS config info
    private val _tlsConfigInfo = MutableLiveData<TlsConfigInfo?>()
    val tlsConfigInfo: LiveData<TlsConfigInfo?> = _tlsConfigInfo
    
    // Privacy and consent state
    private val _consentState = MutableLiveData<ConsentState>()
    val consentState: LiveData<ConsentState> = _consentState
    
    private val _deletionState = MutableLiveData<DeletionState>()
    val deletionState: LiveData<DeletionState> = _deletionState
    
    // Upload policy state
    private val _uploadPolicyWiFiOnly = MutableLiveData<Boolean>()
    val uploadPolicyWiFiOnly: LiveData<Boolean> = _uploadPolicyWiFiOnly

    // Transmission status
    val transmissionStatus: StateFlow<SystemMonitor.TransmissionStatus> =
        systemMonitor.transmissionStatus.stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = SystemMonitor.TransmissionStatus()
        )
    // Time sync status
    val timeSyncStatus: StateFlow<SystemMonitor.TimeSyncStatus> =
        systemMonitor.timeSyncStatus.stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = SystemMonitor.TimeSyncStatus()
        )
    init {
        // Initialize state.
        _collectionStatus.value = "STOPPED"
        _isCollectionRunning.value = false
        _isEncryptedUploading.value = false  // Initial state: not uploading encrypted data
        _sensorStatus.value = mapOf(
            "accelerometer" to false,
            "gyroscope" to false,
            "magnetometer" to false
        )
        _connectionStatus.value = ConnectionStatus.DISCONNECTED
        _transmissionStats.value = TransmissionStats()
        _debugModeEnabled.value = false
        _visualizationEnabled.value = false
        
        // Initialize user ID.
        _userId.value = userIdManager.getUserId()
        
        // Initialize upload policy.
        initializeUploadPolicy()
        
        // Start monitoring network state.
        startNetworkMonitoring()
        
        // Start performance monitoring.
        startPerformanceMonitoring()
        
        // Start memory monitoring.
        startMemoryMonitoring()
        
        // Initial status refresh.
        refreshStatus()
        
        // Refresh TLS config info.
        refreshTlsConfig()
        
        // Initialize privacy state.
        checkPrivacyConsent()
    }
    
    /**
     * Starts monitoring network state.
     */
    private fun startNetworkMonitoring() {
        viewModelScope.launch {
            try {
                networkEnvironmentDetector.networkStateFlow.collect { networkState ->
                    _networkState.value = networkState
                    updateDebugInfo()
                    
                    // Re-apply upload policy on network changes.
                    val wifiOnly = _uploadPolicyWiFiOnly.value ?: false
                    if (wifiOnly) {
                        applyUploadPolicy(wifiOnly)
                    }
                }
            } catch (e: Exception) {
                _errorMessage.value = "Failed to start network monitoring: ${e.message}"
            }
        }
    }
    
    /**
     * Starts performance monitoring.
     */
    private fun startPerformanceMonitoring() {
        viewModelScope.launch {
            try {
                performanceMonitor.startMonitoring(5000L) // Sample every 5 seconds
            } catch (e: Exception) {
                _errorMessage.value = "Failed to start performance monitoring: ${e.message}"
            }
        }
    }
    
    /**
     * Starts memory monitoring.
     */
    private fun startMemoryMonitoring() {
        viewModelScope.launch {
            try {
                memoryMonitor.startMonitoring()
                
                // Memory warning callback.
                memoryMonitor.setOnMemoryWarning { memoryStatus ->
                    Log.w(TAG, "Memory usage warning: ${(memoryStatus.usageRatio * 100).toInt()}%")
                    // Optional: trigger mitigations here.
                }
                
                // Memory critical callback.
                memoryMonitor.setOnMemoryCritical { memoryStatus ->
                    Log.e(TAG, "Memory usage critical: ${(memoryStatus.usageRatio * 100).toInt()}%")
                    // Suggest GC.
                    memoryMonitor.suggestGC()
                }
            } catch (e: Exception) {
                _errorMessage.value = "Failed to start memory monitoring: ${e.message}"
            }
        }
    }
    
    /**
     * Starts encrypted data upload.
     */
    fun startEncryptedUpload() {
        viewModelScope.launch {
            try {
                // Require privacy consent.
                if (privacyManager.consentState.value != ConsentState.GRANTED) {
                    _errorMessage.value = "Privacy consent is required before starting data collection."
                    Log.w(TAG, "Privacy consent not granted; blocking data collection start")
                    return@launch
                }
                
                _collectionStatus.value = "STARTING"
                
                // Start a new session.
                val sessionId = userIdManager.startNewSession()
                _sessionId.value = sessionId
                _sessionStartTime.value = userIdManager.getSessionStartTime()
                
                // SmartTransmissionManager has been removed; use windowed batching.
                
                // TODO: Start sensor collection.
                // sensorCollector.startCollection()
                
                // TODO: Start upload manager.
                // val success = uploadManager.start("server_endpoint")
                
                // Simulate successful start.
                val success = true
                
                if (success) {
                    _collectionStatus.value = "RUNNING"
                    _isCollectionRunning.value = true
                    _isEncryptedUploading.value = true  // After start: uploading encrypted data
                    updateSensorStatus(running = true)
                    _connectionStatus.value = ConnectionStatus.CONNECTED
                    
                    // Start updating session duration.
                    startSessionDurationUpdate()
                } else {
                    _collectionStatus.value = "ERROR"
                    _errorMessage.value = "Failed to start encrypted upload"
                    userIdManager.endSession()
                }
                
                updateDebugInfo()
                
            } catch (e: Exception) {
                _collectionStatus.value = "ERROR"
                _errorMessage.value = "Error starting encrypted upload: ${e.message}"
                userIdManager.endSession()
            }
        }
    }
    
    /**
     * Starts data collection (kept for compatibility).
     */
    fun startCollection() {
        startEncryptedUpload()
    }
    
    /**
     * Records user consent for the privacy agreement.
     */
    fun grantPrivacyConsent() {
        privacyManager.grantConsent()
        checkPrivacyConsent()
    }
    
    /**
     * Initializes upload policy settings.
     */
    private fun initializeUploadPolicy() {
        val prefs = context.getSharedPreferences("upload_policy", Context.MODE_PRIVATE)
        val wifiOnly = prefs.getBoolean("wifi_only", false) // Default: unrestricted
        _uploadPolicyWiFiOnly.value = wifiOnly
        
        // Apply policy to upload manager.
        applyUploadPolicy(wifiOnly)
        
        Log.d(TAG, "Upload policy initialized: ${if (wifiOnly) "Wi-Fi only" else "Unrestricted"}")
    }
    
    /**
     * Sets upload policy.
     */
    fun setUploadPolicyWiFiOnly(wifiOnly: Boolean) {
        viewModelScope.launch {
            try {
                // Persist in SharedPreferences.
                val prefs = context.getSharedPreferences("upload_policy", Context.MODE_PRIVATE)
                prefs.edit().putBoolean("wifi_only", wifiOnly).apply()
                
                // Update LiveData.
                _uploadPolicyWiFiOnly.value = wifiOnly
                
                // Apply policy.
                applyUploadPolicy(wifiOnly)
                
                Log.i(TAG, "Upload policy updated: ${if (wifiOnly) "Wi-Fi only" else "Unrestricted"}")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to set upload policy", e)
                _errorMessage.value = "Failed to set upload policy: ${e.message}"
            }
        }
    }
    
    /**
     * Returns current upload policy.
     */
    fun getUploadPolicyWiFiOnly(): Boolean {
        return _uploadPolicyWiFiOnly.value ?: false
    }
    
    /**
     * Applies upload policy to the upload manager.
     */
    private fun applyUploadPolicy(wifiOnly: Boolean) {
        viewModelScope.launch {
            try {
                // Check current network state.
                val currentNetworkState = _networkState.value ?: NetworkState.UNKNOWN
                
                // If Wi-Fi only is enabled and the current network is not Wi-Fi, pause uploads.
                if (wifiOnly && !isWiFiNetwork(currentNetworkState)) {
                    // If currently uploading, pause uploads.
                    if (_isCollectionRunning.value == true) {
                        Log.i(TAG, "Non-Wi-Fi network; pausing uploads")
                        // TODO: Call uploadManager.pauseUpload()
                    }
                } else {
                    // Resume uploads (if previously paused).
                    if (_isCollectionRunning.value == true) {
                        Log.i(TAG, "Network meets policy; resuming uploads")
                        // TODO: Call uploadManager.resumeUpload()
                    }
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to apply upload policy", e)
            }
        }
    }
    
    /**
     * Returns whether the network state is Wi-Fi.
     */
    private fun isWiFiNetwork(networkState: NetworkState): Boolean {
        return networkState == NetworkState.WIFI_EXCELLENT ||
               networkState == NetworkState.WIFI_GOOD ||
               networkState == NetworkState.WIFI_POOR
    }
    
    /**
     * Stops encrypted data upload.
     */
    fun stopEncryptedUpload() {
        viewModelScope.launch {
            try {
                _collectionStatus.value = "STOPPING"
                
                // SmartTransmissionManager has been removed.
                
                // TODO: Stop sensor collection.
                // sensorCollector.stopCollection()
                
                // TODO: Stop upload manager.
                // uploadManager.stop()
                
                // End session.
                userIdManager.endSession()
                _sessionId.value = null
                _sessionStartTime.value = 0
                _sessionDuration.value = "00:00"
                
                _collectionStatus.value = "STOPPED"
                _isCollectionRunning.value = false
                _isEncryptedUploading.value = false  // After stop: not uploading encrypted data
                updateSensorStatus(running = false)
                _connectionStatus.value = ConnectionStatus.DISCONNECTED
                
                updateDebugInfo()
                
            } catch (e: Exception) {
                _collectionStatus.value = "ERROR"
                _errorMessage.value = "Error stopping encrypted upload: ${e.message}"
            }
        }
    }
    
    /**
     * Stops data collection (kept for compatibility).
     */
    fun stopCollection() {
        stopEncryptedUpload()
    }
    
    /**
     * Updates sensor status.
     */
    private fun updateSensorStatus(running: Boolean) {
        _sensorStatus.value = mapOf(
            "accelerometer" to running,
            "gyroscope" to running,
            "magnetometer" to running
        )
    }
    
    /**
     * Refreshes status information.
     */
    fun refreshStatus() {
        viewModelScope.launch {
            try {
                // Update network state.
                _networkState.value = networkEnvironmentDetector.getCurrentNetworkState()
                
                // Fetch upload status.
                val uploadStatus = uploadManager.getUploadStatus()
                _connectionStatus.value = uploadStatus.connectionStatus
                
                // Update file queue stats.
                uploadStatus.fileQueueStats?.let {
                    _fileQueueStats.value = it
                }
                
                // Update transmission stats.
                updateTransmissionStats()
                
                // Update debug info.
                updateDebugInfo()
                
            } catch (e: Exception) {
                _errorMessage.value = "Failed to refresh status: ${e.message}"
            }
        }
    }
    
    /**
     * Refreshes TLS config information.
     */
    fun refreshTlsConfig() {
        viewModelScope.launch {
            try {
                val configInfo = tlsSecurityManager.getTlsConfigInfo()
                _tlsConfigInfo.value = configInfo
            } catch (e: Exception) {
                // Ignore errors; TLS info is for display only.
            }
        }
    }
    
    /**
     * Updates transmission statistics.
     */
    private fun updateTransmissionStats() {
        // TODO: Fetch stats from real modules.
        val stats = TransmissionStats(
            isFastMode = false, // TODO: Fetch from TransmissionController
            packetsSent = 0L,   // TODO: Fetch from UploadManager
            packetsPending = 0,  // TODO: Fetch from buffer
            lastAckLatency = null // TODO: Fetch from ConnectionStats
        )
        _transmissionStats.value = stats
    }
    
    /**
     * Toggles debug mode.
     */
    fun toggleDebugMode() {
        val currentMode = _debugModeEnabled.value ?: false
        _debugModeEnabled.value = !currentMode
        
        if (!currentMode) {
            updateDebugInfo()
        }
    }
    
    /**
     * Toggles visualization mode.
     */
    fun toggleVisualization() {
        val currentMode = _visualizationEnabled.value ?: false
        _visualizationEnabled.value = !currentMode
    }
    
    /**
     * Starts visualization.
     */
    fun startVisualization() {
        // TODO: Start real-time visualization.
        // chartManager.startVisualization()
    }
    
    /**
     * Updates debug information.
     */
    private fun updateDebugInfo() {
        if (_debugModeEnabled.value != true) return
        
        val debugText = buildString {
            appendLine("=== Transmission ===")
            appendLine("Manager state: ${if (_isCollectionRunning.value == true) "Running" else "Stopped"}")
            appendLine("Batching: 1s windows")
            appendLine("Batch interval: 1000ms")
            appendLine()
            
            appendLine("=== gRPC Connection ===")
            appendLine("Endpoint: grpc://localhost:8080") // TODO: Load from config.
            appendLine("Connection state: ${_connectionStatus.value}")
            val stats = _transmissionStats.value
            stats?.lastAckLatency?.let {
                appendLine("Last ACK latency: ${it}ms")
            }
            appendLine()
            
            appendLine("=== Buffer & Transmission ===")
            val metricsSnapshot = metricsCollector.getSnapshot()
            appendLine("Memory samples: 0") // TODO: Read from the actual buffer.
            appendLine("Disk queue batches: ${metricsSnapshot.counters[com.continuousauth.observability.MetricType.BATCHES_PROCESSED] ?: 0}")
            appendLine("Packets sent: ${metricsSnapshot.counters[com.continuousauth.observability.MetricType.UPLOADS_SUCCESS] ?: 0}")
            appendLine("Upload failures: ${(metricsSnapshot.counters[com.continuousauth.observability.MetricType.UPLOADS_FAILED_NETWORK] ?: 0) + 
                (metricsSnapshot.counters[com.continuousauth.observability.MetricType.UPLOADS_FAILED_SERVER] ?: 0)}")
            appendLine("Packets discarded: 0") // TODO: Add discard counter.
            appendLine("Upload success rate: ${"%.1f".format(metricsSnapshot.summary.currentUploadSuccessRate)}%")
            appendLine()
            
            appendLine("=== Sensors ===")
            val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
            val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
            val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
            val magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
            
            accelerometer?.let {
                appendLine("Accelerometer: ${it.vendor}")
                appendLine("  Max rate: ${1_000_000 / it.minDelay}Hz")
                appendLine("  FIFO max events: ${it.fifoMaxEventCount}")
            }
            gyroscope?.let {
                appendLine("Gyroscope: ${it.vendor}")
                appendLine("  Max rate: ${1_000_000 / it.minDelay}Hz")
                appendLine("  FIFO max events: ${it.fifoMaxEventCount}")
            }
            magnetometer?.let {
                appendLine("Magnetometer: ${it.vendor}")
                appendLine("  Max rate: ${1_000_000 / it.minDelay}Hz")
                appendLine("  FIFO max events: ${it.fifoMaxEventCount}")
            }
            appendLine()
            
            appendLine("=== Device & Keystore ===")
            appendLine("Device model: ${Build.MODEL}")
            appendLine("Android version: ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
            appendLine("Manufacturer: ${Build.MANUFACTURER}")
            appendLine("Product: ${Build.PRODUCT}")
            
            try {
                val keyStore = KeyStore.getInstance("AndroidKeyStore")
                keyStore.load(null)
                appendLine("Keystore Provider: AndroidKeyStore")
                appendLine("StrongBox supported: ${Build.VERSION.SDK_INT >= Build.VERSION_CODES.P}")
                appendLine("Key alias count: ${keyStore.aliases().toList().size}")
            } catch (e: Exception) {
                appendLine("Keystore Provider: unavailable")
                appendLine("StrongBox supported: unknown")
            }
            appendLine()
            
            appendLine("=== Anomaly Detection ===")
            appendLine("Anomaly detection: disabled (removed from spec)")
            appendLine()
            
            appendLine("=== Performance ===")
            val performanceStats = performanceMonitor.getPerformanceStats(60000L)
            appendLine("Samples: ${performanceStats.sampleCount}")
            appendLine("Memory avg: ${performanceStats.memoryUsageAvg.toInt()}MB")
            appendLine("Memory peak: ${performanceStats.memoryUsagePeak}MB")
            appendLine("CPU avg: ${"%.1f".format(performanceStats.cpuUsageAvg)}%")
            appendLine("CPU peak: ${"%.1f".format(performanceStats.cpuUsagePeak)}%")
            appendLine("Heap utilization: ${"%.1f".format(performanceStats.heapUtilization)}%")
            appendLine()
            
            appendLine("=== Memory Monitor ===")
            val memoryStatus = memoryMonitor.getCurrentMemoryStatus()
            val memoryStats = memoryMonitor.getMonitoringStats()
            appendLine("Memory usage: ${(memoryStatus.usedMemoryBytes / (1024 * 1024))}MB / ${(memoryStatus.maxMemoryBytes / (1024 * 1024))}MB")
            appendLine("Utilization: ${"%.1f".format(memoryStatus.usageRatio * 100)}%")
            appendLine("State: ${when {
                memoryStatus.isCritical -> "Critical"
                memoryStatus.isWarning -> "Warning"
                else -> "Normal"
            }}")
            appendLine("Monitoring samples: ${memoryStats.samplesCount}")
            appendLine("Average utilization: ${"%.1f".format(memoryStats.averageUsageRatio * 100)}%")
            appendLine("Peak utilization: ${"%.1f".format(memoryStats.peakUsageRatio * 100)}%")
            appendLine("Warning count: ${memoryStats.warningCount}")
            appendLine("Critical count: ${memoryStats.criticalCount}")
            appendLine()
            
            appendLine("=== Object Pool ===")
            val poolStatus = sensorEventPool.getPoolStatus()
            appendLine("Pool size: ${poolStatus.currentSize}/${poolStatus.maxSize}")
            appendLine("Total acquired: ${poolStatus.totalAcquired}")
            appendLine("Total released: ${poolStatus.totalReleased}")
            appendLine("Total created: ${poolStatus.totalCreated}")
            appendLine("Hit rate: ${"%.1f".format(poolStatus.hitRate * 100)}%")
            appendLine()
            
            appendLine("=== Network ===")
            appendLine("Network type: ${_networkState.value}")
            val networkConfig = networkEnvironmentDetector.getTransmissionConfig()
            appendLine("Max concurrent uploads: ${networkConfig.maxConcurrentUploads}")
            appendLine("Upload timeout: ${networkConfig.uploadTimeoutMs}ms")
            appendLine("Retry delay: ${networkConfig.retryDelayMs}ms")
            appendLine("Max retry attempts: ${networkConfig.maxRetryAttempts}")
            appendLine()
            
            appendLine("=== Metrics Summary ===")
            appendLine("Uptime: ${metricsCollector.getUptime() / 1000}s")
            appendLine("Anomalies detected: ${metricsSnapshot.counters[com.continuousauth.observability.MetricType.ANOMALIES_DETECTED] ?: 0}")
            appendLine("Mode switches (fast→slow): ${metricsSnapshot.counters[com.continuousauth.observability.MetricType.MODE_SWITCHES_TO_SLOW] ?: 0}")
            appendLine("Mode switches (slow→fast): ${metricsSnapshot.counters[com.continuousauth.observability.MetricType.MODE_SWITCHES_TO_FAST] ?: 0}")
            appendLine("Fast mode total time: ${(metricsSnapshot.counters[com.continuousauth.observability.MetricType.FAST_MODE_TOTAL_TIME] ?: 0) / 1000}s")
        }
        
        _debugInfo.value = debugText
    }
    
    /**
     * Checks Usage Stats permission.
     */
    suspend fun hasUsageStatsPermission(): Boolean {
        // TODO: Implement actual permission checks.
        return false
    }
    
    /**
     * Checks Usage Stats permission.
     */
    fun checkUsageStatsPermission() {
        viewModelScope.launch {
            val hasPermission = hasUsageStatsPermission()
            // TODO: Update permission state.
            updateDebugInfo()
        }
    }
    
    /**
     * Tests server connectivity.
     */
    fun testServerConnection(serverIp: String, serverPort: String) {
        viewModelScope.launch {
            try {
                _serverTestResult.value = "Testing server connection..."
                
                val port = serverPort.toIntOrNull() ?: 50051
                val result = serverConnectionTester.testServerConnection(
                    serverIp = serverIp,
                    serverPort = port,
                    testGrpc = true
                )
                
                _serverTestResult.value = serverConnectionTester.getTestResultDescription(result)
                
                // Update connection status based on test result.
                if (result.isReachable) {
                    _connectionStatus.value = ConnectionStatus.CONNECTED
                } else {
                    _connectionStatus.value = ConnectionStatus.DISCONNECTED
                }
                
                // Clear test result after 3 seconds.
                delay(3000)
                _serverTestResult.value = null
                
            } catch (e: Exception) {
                _serverTestResult.value = "Test failed: ${e.message}"
                _connectionStatus.value = ConnectionStatus.DISCONNECTED
                
                delay(3000)
                _serverTestResult.value = null
            }
        }
    }
    
    /**
     * Starts updating session duration.
     */
    private fun startSessionDurationUpdate() {
        viewModelScope.launch {
            while (_isEncryptedUploading.value == true) {
                _sessionDuration.value = userIdManager.getFormattedSessionDuration()
                delay(1000) // Update every second
            }
        }
    }
    
    /**
     * Clears current error message.
     */
    fun clearErrorMessage() {
        _errorMessage.value = null
    }
    
    /**
     * Clears file queue.
     */
    fun clearFileQueue() {
        viewModelScope.launch {
            try {
                fileQueueManager.clearQueue()
                // Refresh file queue stats.
                val stats = fileQueueManager.getQueueStatistics()
                _fileQueueStats.value = QueueStats(
                    totalPackets = stats.totalPackets,
                    pendingPackets = stats.pendingPackets,
                    uploadedPackets = stats.uploadedPackets,
                    corruptedPackets = stats.corruptedPackets,
                    totalSizeBytes = stats.totalSizeBytes
                )
                Log.i(TAG, "File queue cleared")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to clear file queue", e)
                _errorMessage.value = "Failed to clear file queue: ${e.message}"
            }
        }
    }
    
    // ==================== Privacy & Compliance ====================
    
    /**
     * Checks privacy consent state.
     */
    fun checkPrivacyConsent() {
        privacyManager.checkConsentStatus()
        viewModelScope.launch {
            privacyManager.consentState.collect { state ->
                _consentState.value = state
                
                // Stop collection if consent is not granted or has been withdrawn.
                if (state == ConsentState.NOT_GRANTED || state == ConsentState.WITHDRAWN) {
                    stopCollection()
                }
            }
        }
        
        viewModelScope.launch {
            privacyManager.deletionState.collect { state ->
                _deletionState.value = state
            }
        }
    }
    
    /**
     * Withdraws consent and deletes local data.
     */
    fun withdrawConsentAndDeleteData() {
        viewModelScope.launch {
            try {
                // Stop collection first.
                stopCollection()
                
                // Show deletion progress.
                _collectionStatus.value = "Deleting data..."
                
                // Perform consent withdrawal and data deletion.
                val result = privacyManager.withdrawConsentAndDeleteData()
                
                if (result.isSuccess) {
                    _collectionStatus.value = "Data deleted"
                    _errorMessage.value = null
                    
                    // Clear session info.
                    _sessionId.value = null
                    _sessionStartTime.value = 0
                    _sessionDuration.value = "00:00"
                    
                    Log.i(TAG, "Consent withdrawn; all data deleted")
                } else {
                    _errorMessage.value = "Data deletion failed: ${result.exceptionOrNull()?.message}"
                    Log.e(TAG, "Consent withdrawal failed", result.exceptionOrNull())
                }
            } catch (e: Exception) {
                _errorMessage.value = "Consent withdrawal failed: ${e.message}"
                Log.e(TAG, "Error during consent withdrawal", e)
            }
        }
    }
    
    /**
     * Returns data retention days.
     */
    fun getDataRetentionDays(): Int {
        return privacyManager.getDataRetentionDays()
    }
    
    /**
     * Sets data retention days.
     */
    fun setDataRetentionDays(days: Int) {
        privacyManager.setDataRetentionDays(days)
    }
    
    /**
     * Runs data retention cleanup.
     */
    fun performRetentionCleanup() {
        viewModelScope.launch {
            if (privacyManager.shouldPerformRetentionCleanup()) {
                privacyManager.performRetentionCleanup()
                Log.i(TAG, "Retention cleanup executed")
            }
        }
    }
    
    /**
     * Retries pending deletion requests.
     */
    fun retryPendingDeletionRequests() {
        viewModelScope.launch {
            privacyManager.retryPendingDeletionRequests()
        }
    }
    
    override fun onCleared() {
        super.onCleared()
        // Clean up resources.
        networkEnvironmentDetector.cleanup()
        // SmartTransmissionManager has been removed.
        performanceMonitor.cleanup()
        memoryMonitor.cleanup()
        viewModelScope.launch {
            performanceMonitor.stopMonitoring()
            memoryMonitor.stopMonitoring()
        }
    }
}
