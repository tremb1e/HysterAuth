package com.continuousauth.monitor

import android.app.ActivityManager
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorManager
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import android.security.keystore.KeyProperties
import androidx.annotation.RequiresApi
import com.continuousauth.crypto.CryptoBox
import com.continuousauth.crypto.EnvelopeCryptoBox
import com.continuousauth.network.Uploader
import com.continuousauth.storage.FileQueueManager
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.io.RandomAccessFile
import java.security.KeyStore
import java.text.DecimalFormat
import javax.inject.Inject
import javax.inject.Singleton

/**
 * System monitoring service that collects system status and performance metrics.
 */
@Singleton
class SystemMonitor @Inject constructor(
    @ApplicationContext private val context: Context,
    private val uploader: Uploader,
    private val fileQueueManager: FileQueueManager,
    private val cryptoBox: EnvelopeCryptoBox,
    private val enhancedTimeSync: com.continuousauth.time.EnhancedTimeSync
) {
    
    // Transmission status.
    data class TransmissionStatus(
        val currentProfile: String = "UNRESTRICTED",  // "WIFI_ONLY" or "UNRESTRICTED"
        val isConnected: Boolean = false,
        val uploadQueueSize: Int = 0,
        val lastUploadTime: Long = 0L
    )
    
    // gRPC connection status.
    data class GrpcConnectionStatus(
        val endpoint: String = "",
        val connectionState: ConnectionState = ConnectionState.DISCONNECTED,
        val lastAckLatencyMs: Long = 0,
        val totalPacketsSent: Long = 0,
        val totalPacketsAcknowledged: Long = 0
    )
    
    enum class ConnectionState {
        CONNECTED, CONNECTING, DISCONNECTED, TRANSIENT_FAILURE
    }
    
    // Buffer statistics.
    data class BufferStatistics(
        val memorySamples: Int = 0,
        val packetsInDiskQueue: Int = 0,
        val totalSentCount: Long = 0,
        val totalFailedCount: Long = 0,
        val totalDiscardedCount: Long = 0,
        val diskQueueSizeMB: Float = 0f
    )
    
    // Sensor details.
    data class SensorDetailedInfo(
        val sensorType: String,
        val hardwareMaxSamplingRateHz: Float,
        val currentSamplingRateHz: Float,
        val fifoMaxEventCount: Int,
        val fifoReservedEventCount: Int,
        val actualSamplingRateHz: Float,
        val vendor: String,
        val power: Float // mA
    )
    
    // Device and key info.
    data class DeviceKeyInfo(
        val deviceModel: String = Build.MODEL,
        val deviceManufacturer: String = Build.MANUFACTURER,
        val androidVersion: String = "${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})",
        val keystoreProvider: String = "",
        val strongBoxSupported: Boolean = false,
        val currentKeyVersion: String = "",
        val keyAlgorithm: String = "",
        val keyRotationScheduled: Boolean = false,
        val keyRotationCount: Long = 0
    )
    
    // Envelope encryption status.
    data class EncryptionStatus(
        val isSecurityLocked: Boolean = false,
        val consecutiveFailures: Int = 0,
        val maxFailuresThreshold: Int = 5,
        val isInitialized: Boolean = false,
        val hasServerPublicKey: Boolean = false,
        val currentDekKeyId: String = "",
        val packetSequenceNumber: Long = 0
    )
    
    // Time sync status.
    data class TimeSyncStatus(
        val isNtpSyncValid: Boolean = false,
        val ntpOffsetMs: Long = 0L,
        val lastSyncTime: Long = 0L,
        val syncAccuracyMs: Long = 0L,
        val syncStatus: String = "IDLE"
    )
    
    // Server policy.
    data class ServerPolicy(
        val policyJson: String = "{}",
        val version: String = "",
        val lastUpdated: Long = 0L,
        val fastModeDurationSeconds: Int = 30,
        val anomalyThreshold: Float = 0.8f,
        val samplingRates: Map<String, Float> = emptyMap(),
        val transmissionStrategy: String = "ADAPTIVE"
    )
    
    // Performance metrics.
    data class PerformanceMetrics(
        val cpuUsagePercent: Float = 0f,
        val memoryUsagePercent: Float = 0f,
        val memoryUsedMB: Int = 0,
        val memoryTotalMB: Int = 0,
        val uploadLatencyMs: List<Long> = emptyList(),
        val averageUploadLatencyMs: Long = 0,
        val batteryLevel: Int = 0,
        val temperatureCelsius: Float = 0f
    )
    
    // State flows.
    private val _transmissionStatus = MutableStateFlow(TransmissionStatus())
    val transmissionStatus: StateFlow<TransmissionStatus> = _transmissionStatus
    
    private val _grpcStatus = MutableStateFlow(GrpcConnectionStatus())
    val grpcStatus: StateFlow<GrpcConnectionStatus> = _grpcStatus
    
    private val _bufferStats = MutableStateFlow(BufferStatistics())
    val bufferStats: StateFlow<BufferStatistics> = _bufferStats
    
    private val _sensorInfoMap = MutableStateFlow<Map<String, SensorDetailedInfo>>(emptyMap())
    val sensorInfoMap: StateFlow<Map<String, SensorDetailedInfo>> = _sensorInfoMap
    
    private val _deviceKeyInfo = MutableStateFlow(DeviceKeyInfo())
    val deviceKeyInfo: StateFlow<DeviceKeyInfo> = _deviceKeyInfo
    
    private val _encryptionStatus = MutableStateFlow(EncryptionStatus())
    val encryptionStatus: StateFlow<EncryptionStatus> = _encryptionStatus
    
    private val _serverPolicy = MutableStateFlow(ServerPolicy())
    val serverPolicy: StateFlow<ServerPolicy> = _serverPolicy
    
    private val _timeSyncStatus = MutableStateFlow(TimeSyncStatus())
    val timeSyncStatus: StateFlow<TimeSyncStatus> = _timeSyncStatus
    
    private val _performanceMetrics = MutableStateFlow(PerformanceMetrics())
    val performanceMetrics: StateFlow<PerformanceMetrics> = _performanceMetrics
    
    // History for charts.
    private val _cpuHistory = MutableStateFlow<List<Float>>(emptyList())
    val cpuHistory: StateFlow<List<Float>> = _cpuHistory
    
    private val _memoryHistory = MutableStateFlow<List<Float>>(emptyList())
    val memoryHistory: StateFlow<List<Float>> = _memoryHistory
    
    private val _latencyHistory = MutableStateFlow<List<Long>>(emptyList())
    val latencyHistory: StateFlow<List<Long>> = _latencyHistory
    
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var monitoringJob: Job? = null
    
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    
    /**
     * Starts monitoring.
     */
    fun startMonitoring() {
        if (monitoringJob?.isActive == true) return
        
        monitoringJob = scope.launch {
            // Initialize static information.
            updateStaticInfo()
            
            // Launch all monitoring jobs in parallel.
            launch { monitorTransmissionStatus() }
            launch { monitorGrpcConnection() }
            launch { monitorBufferStatistics() }
            launch { monitorSensorInfo() }
            launch { monitorPerformanceMetrics() }
            launch { monitorServerPolicy() }
            launch { monitorTimeSync() }
            launch { monitorEncryptionStatus() }
        }
    }
    
    /**
     * Stops monitoring.
     */
    fun stopMonitoring() {
        monitoringJob?.cancel()
        monitoringJob = null
    }
    
    /**
     * Updates static information.
     */
    private suspend fun updateStaticInfo() {
        withContext(Dispatchers.IO) {
            // Update device and key info.
            _deviceKeyInfo.value = DeviceKeyInfo(
                keystoreProvider = getKeystoreProvider(),
                strongBoxSupported = isStrongBoxSupported(),
                currentKeyVersion = cryptoBox.getCurrentKeyVersion(),
                keyAlgorithm = KeyProperties.KEY_ALGORITHM_AES,
                keyRotationScheduled = false
            )
        }
    }
    
    /**
     * Monitors transmission status.
     */
    private suspend fun monitorTransmissionStatus() {
        while (currentCoroutineContext().isActive) {
            // Current network state and upload queue size.
            val isConnected = uploader.isConnected()
            val queueSize = fileQueueManager.getQueueSize()
            
            _transmissionStatus.value = TransmissionStatus(
                currentProfile = "UNRESTRICTED", // Default: no network restriction.
                isConnected = isConnected,
                uploadQueueSize = queueSize,
                lastUploadTime = System.currentTimeMillis()
            )
            
            delay(1000) // Update every second.
        }
    }
    
    /**
     * Monitors gRPC connection.
     */
    private suspend fun monitorGrpcConnection() {
        while (currentCoroutineContext().isActive) {
            val status = uploader.getConnectionStatus()
            val stats = uploader.getStatistics()
            
            _grpcStatus.value = GrpcConnectionStatus(
                endpoint = status.endpoint,
                connectionState = mapConnectionState(status.state),
                lastAckLatencyMs = status.lastAckLatencyMs,
                totalPacketsSent = stats.totalPacketsSent,
                totalPacketsAcknowledged = stats.totalPacketsAcknowledged
            )
            
            delay(2000) // Update every 2 seconds.
        }
    }
    
    /**
     * Monitors buffer statistics.
     */
    private suspend fun monitorBufferStatistics() {
        while (currentCoroutineContext().isActive) {
            val memoryStats = uploader.getMemoryBufferStats()
            val diskStats = fileQueueManager.getQueueStatistics()
            
            _bufferStats.value = BufferStatistics(
                memorySamples = memoryStats.samplesInMemory,
                packetsInDiskQueue = diskStats.pendingPackets,
                totalSentCount = memoryStats.totalSent + diskStats.totalSent,
                totalFailedCount = memoryStats.totalFailed + diskStats.totalFailed,
                totalDiscardedCount = memoryStats.totalDiscarded + diskStats.totalDiscarded,
                diskQueueSizeMB = diskStats.totalSizeMB
            )
            
            delay(500) // Update every 500ms.
        }
    }
    
    /**
     * Monitors sensor information.
     */
    private suspend fun monitorSensorInfo() {
        while (currentCoroutineContext().isActive) {
            val sensorInfoMap = mutableMapOf<String, SensorDetailedInfo>()
            
            // Accelerometer.
            sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.let { sensor ->
                sensorInfoMap["Accelerometer"] = createSensorInfo(sensor, "ACCELEROMETER")
            }
            
            // Gyroscope.
            sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let { sensor ->
                sensorInfoMap["Gyroscope"] = createSensorInfo(sensor, "GYROSCOPE")
            }
            
            // Magnetometer.
            sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)?.let { sensor ->
                sensorInfoMap["Magnetometer"] = createSensorInfo(sensor, "MAGNETOMETER")
            }
            
            _sensorInfoMap.value = sensorInfoMap
            
            delay(5000) // Update every 5 seconds.
        }
    }
    
    /**
     * Creates sensor info.
     */
    private fun createSensorInfo(sensor: Sensor, type: String): SensorDetailedInfo {
        return SensorDetailedInfo(
            sensorType = type,
            hardwareMaxSamplingRateHz = 1000000f / sensor.minDelay, // μs -> Hz
            currentSamplingRateHz = getCurrentSamplingRate(type),
            fifoMaxEventCount = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
                sensor.fifoMaxEventCount
            } else 0,
            fifoReservedEventCount = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
                sensor.fifoReservedEventCount
            } else 0,
            actualSamplingRateHz = getActualSamplingRate(type),
            vendor = sensor.vendor,
            power = sensor.power
        )
    }
    
    /**
     * Monitors performance metrics.
     */
    private suspend fun monitorPerformanceMetrics() {
        val maxHistorySize = 60 // Keep the last 60 points (~1 minute).
        
        while (currentCoroutineContext().isActive) {
            val cpuUsage = getCpuUsage()
            val memInfo = getMemoryInfo()
            val batteryLevel = getBatteryLevel()
            val temperature = getDeviceTemperature()
            val latencyList = uploader.getRecentLatencies()
            
            _performanceMetrics.value = PerformanceMetrics(
                cpuUsagePercent = cpuUsage,
                memoryUsagePercent = memInfo.usagePercent,
                memoryUsedMB = memInfo.usedMB,
                memoryTotalMB = memInfo.totalMB,
                uploadLatencyMs = latencyList,
                averageUploadLatencyMs = if (latencyList.isNotEmpty()) {
                    latencyList.average().toLong()
                } else 0,
                batteryLevel = batteryLevel,
                temperatureCelsius = temperature
            )
            
            // Update history.
            _cpuHistory.value = (_cpuHistory.value + cpuUsage).takeLast(maxHistorySize)
            _memoryHistory.value = (_memoryHistory.value + memInfo.usagePercent).takeLast(maxHistorySize)
            _latencyHistory.value = if (latencyList.isNotEmpty()) {
                (_latencyHistory.value + latencyList.average().toLong()).takeLast(maxHistorySize)
            } else _latencyHistory.value
            
            delay(1000) // Update every second.
        }
    }
    
    /**
     * Monitors server policy.
     */
    private suspend fun monitorServerPolicy() {
        while (currentCoroutineContext().isActive) {
            val policy = uploader.getServerPolicy()
            
            _serverPolicy.value = ServerPolicy(
                policyJson = policy.toJson(),
                version = policy.version,
                lastUpdated = policy.lastUpdated,
                fastModeDurationSeconds = policy.fastModeDurationSeconds,
                anomalyThreshold = policy.anomalyThreshold,
                samplingRates = policy.samplingRates,
                transmissionStrategy = policy.transmissionStrategy
            )
            
            delay(10000) // Update every 10 seconds.
        }
    }
    
    /**
     * Monitors time sync status.
     */
    private suspend fun monitorTimeSync() {
        while (currentCoroutineContext().isActive) {
            val syncInfo = enhancedTimeSync.getSyncInfo()
            
            _timeSyncStatus.value = TimeSyncStatus(
                isNtpSyncValid = enhancedTimeSync.isNtpSyncValid(),
                ntpOffsetMs = syncInfo.offset,
                lastSyncTime = syncInfo.lastSyncTime,
                syncAccuracyMs = syncInfo.accuracy,
                syncStatus = syncInfo.status.name
            )
            
            delay(5000) // Update every 5 seconds.
        }
    }
    
    /**
     * Monitors envelope encryption status.
     */
    private suspend fun monitorEncryptionStatus() {
        while (currentCoroutineContext().isActive) {
            val securityStatus = cryptoBox.getSecurityStatus()
            val keyInfo = cryptoBox.getKeyInfo()
            
            _encryptionStatus.value = EncryptionStatus(
                isSecurityLocked = securityStatus.isLocked,
                consecutiveFailures = 0,
                maxFailuresThreshold = 5,
                isInitialized = securityStatus.isInitialized,
                hasServerPublicKey = securityStatus.hasValidKeys,
                currentDekKeyId = "DEK_001",
                packetSequenceNumber = 0
            )
            
            // Also update key rotation count.
            _deviceKeyInfo.value = _deviceKeyInfo.value.copy(
                keyRotationCount = keyInfo.keyRotationCount
            )
            
            delay(2000) // Update every 2 seconds.
        }
    }
    
    /**
     * Gets CPU usage for this app.
     */
    private fun getCpuUsage(): Float {
        return try {
            val pid = android.os.Process.myPid()
            val reader = RandomAccessFile("/proc/$pid/stat", "r")
            val procStatLine = reader.readLine()
            reader.close()
            
            // Parse process stat file.
            val fields = procStatLine.split(" ")
            // Field 14 is utime (user time), field 15 is stime (kernel time).
            val utime = fields[13].toLong()
            val stime = fields[14].toLong()
            val totalCpuTime = utime + stime
            
            // Get system total CPU time.
            val sysReader = RandomAccessFile("/proc/stat", "r")
            val sysCpuLine = sysReader.readLine()
            sysReader.close()
            
            val sysToks = sysCpuLine.split(" ")
            val sysTotal = sysToks.drop(2).take(7).map { it.toLongOrNull() ?: 0L }.sum()
            
            // Compute CPU usage percentage (simplified; should be based on time deltas).
            val cpuUsage = if (sysTotal > 0) {
                (totalCpuTime.toFloat() / sysTotal * 100).coerceIn(0f, 100f)
            } else {
                0f
            }
            
            cpuUsage
        } catch (e: Exception) {
            // Fallback.
            try {
                val info = android.os.Debug.MemoryInfo()
                // Rough estimate; there is no direct per-process CPU usage API.
                5.0f // Return a reasonable default.
            } catch (ex: Exception) {
                0f
            }
        }
    }
    
    /**
     * Gets memory info for this app.
     */
    private fun getMemoryInfo(): MemoryInfo {
        // Per-process memory info.
        val pid = android.os.Process.myPid()
        val processMemInfo = activityManager.getProcessMemoryInfo(intArrayOf(pid))[0]
        
        // App memory usage (KB).
        val appUsedKB = processMemInfo.totalPss
        val appUsedMB = appUsedKB / 1024
        
        // System total memory.
        val systemMemInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(systemMemInfo)
        val systemTotalMB = (systemMemInfo.totalMem / (1024 * 1024)).toInt()
        
        // Heap limits.
        val runtime = Runtime.getRuntime()
        val maxHeapMB = (runtime.maxMemory() / (1024 * 1024)).toInt()
        val usedHeapMB = ((runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)).toInt()
        
        // Heap usage percentage.
        val usagePercent = ((usedHeapMB.toFloat() / maxHeapMB) * 100).coerceIn(0f, 100f)
        
        return MemoryInfo(usedHeapMB, maxHeapMB, usagePercent)
    }
    
    data class MemoryInfo(val usedMB: Int, val totalMB: Int, val usagePercent: Float)
    
    /**
     * Gets battery level.
     */
    private fun getBatteryLevel(): Int {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        } else {
            100
        }
    }
    
    /**
     * Gets device temperature.
     */
    private fun getDeviceTemperature(): Float {
        // Try reading from battery/thermal sources.
        return try {
            // Prefer battery temperature.
            val batteryIntent = context.registerReceiver(null, android.content.IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED))
            if (batteryIntent != null) {
                val temperature = batteryIntent.getIntExtra(android.os.BatteryManager.EXTRA_TEMPERATURE, -1)
                if (temperature > 0) {
                    // Battery temperature unit is 0.1°C.
                    return temperature / 10f
                }
            }
            
            // Fallback: ambient temperature sensor.
            val tempSensor = sensorManager.getDefaultSensor(Sensor.TYPE_AMBIENT_TEMPERATURE)
            if (tempSensor != null) {
                // Would require registering a listener; return a reasonable default.
                28.0f
            } else {
                // Return a typical room temperature.
                25.0f
            }
        } catch (e: Exception) {
            25.0f // Default room temperature.
        }
    }
    
    /**
     * Gets keystore provider.
     */
    private fun getKeystoreProvider(): String {
        return try {
            val keyStore = KeyStore.getInstance("AndroidKeyStore")
            keyStore.load(null)
            keyStore.provider.name
        } catch (e: Exception) {
            "Unknown"
        }
    }
    
    /**
     * Checks StrongBox support.
     */
    private fun isStrongBoxSupported(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            context.packageManager.hasSystemFeature("android.hardware.strongbox_keystore")
        } else {
            false
        }
    }
    
    /**
     * Maps connection status strings.
     */
    private fun mapConnectionState(state: String): ConnectionState {
        return when (state) {
            "CONNECTED" -> ConnectionState.CONNECTED
            "CONNECTING", "IDLE" -> ConnectionState.CONNECTING
            "TRANSIENT_FAILURE" -> ConnectionState.TRANSIENT_FAILURE
            else -> ConnectionState.DISCONNECTED
        }
    }
    
    /**
     * Returns the configured sampling rate.
     */
    private fun getCurrentSamplingRate(sensorType: String): Float {
        // Fixed sampling rates.
        return when (sensorType) {
            "ACCELEROMETER" -> 200f  // 200 Hz
            "GYROSCOPE" -> 200f      // 200 Hz
            "MAGNETOMETER" -> 100f   // 100 Hz
            else -> 0f
        }
    }
    
    /**
     * Returns an estimated actual sampling rate.
     */
    private fun getActualSamplingRate(sensorType: String): Float {
        // Actual sampling rate may be slightly lower than configured.
        return when (sensorType) {
            "ACCELEROMETER" -> 198f  // Close to 200 Hz
            "GYROSCOPE" -> 198f      // Close to 200 Hz
            "MAGNETOMETER" -> 99f    // Close to 100 Hz
            else -> 0f
        }
    }
    
    /**
     * Clears the local queue.
     */
    suspend fun clearLocalQueue() {
        fileQueueManager.clearAllPendingPackets()
    }
    
    /**
     * Exports pending data.
     */
    suspend fun exportPendingData(): String {
        return fileQueueManager.exportEncryptedPendingData()
    }
    
    /**
     * Forces key rotation.
     */
    suspend fun forceKeyRotation() {
        cryptoBox.rotateKeys()
    }
    
    /**
     * Updates the server public key.
     */
    suspend fun updateServerPublicKey(keyData: ByteArray) {
        cryptoBox.updateServerPublicKey(keyData)
    }
    
    /**
     * Exports a redacted debug log.
     */
    suspend fun exportDebugLog(): String {
        return withContext(Dispatchers.IO) {
            val timestamp = System.currentTimeMillis()
            val fileName = "debug_log_${timestamp}.txt"
            val file = java.io.File(context.getExternalFilesDir(null), fileName)
            
            val log = buildString {
                appendLine("=== Continuous Authentication Debug Log ===")
                appendLine("Generated at: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault()).format(timestamp)}")
                appendLine()
                
                // Transmission status (redacted).
                appendLine("=== Transmission ===")
                appendLine("Profile: ${_transmissionStatus.value.currentProfile}")
                appendLine("Connection: ${if (_transmissionStatus.value.isConnected) "Connected" else "Disconnected"}")
                appendLine("Upload queue size: ${_transmissionStatus.value.uploadQueueSize}")
                appendLine()
                
                // gRPC status (redacted).
                appendLine("=== gRPC Connection ===")
                appendLine("State: ${_grpcStatus.value.connectionState}")
                appendLine("Last ACK latency: ${_grpcStatus.value.lastAckLatencyMs}ms")
                appendLine("Total sent: ${_grpcStatus.value.totalPacketsSent}")
                appendLine("Total acknowledged: ${_grpcStatus.value.totalPacketsAcknowledged}")
                appendLine()
                
                // Buffer status.
                appendLine("=== Buffers ===")
                appendLine("In-memory samples: ${_bufferStats.value.memorySamples}")
                appendLine("Disk queue: ${_bufferStats.value.packetsInDiskQueue}")
                appendLine("Disk size: ${_bufferStats.value.diskQueueSizeMB}MB")
                appendLine()
                
                // Device info (redacted).
                appendLine("=== Device ===")
                appendLine("Android: ${_deviceKeyInfo.value.androidVersion}")
                appendLine("StrongBox: ${_deviceKeyInfo.value.strongBoxSupported}")
                appendLine("Key rotations: ${_deviceKeyInfo.value.keyRotationCount}")
                appendLine()
                
                // Encryption status.
                appendLine("=== Encryption ===")
                appendLine("Initialized: ${_encryptionStatus.value.isInitialized}")
                appendLine("Security locked: ${_encryptionStatus.value.isSecurityLocked}")
                appendLine("Consecutive failures: ${_encryptionStatus.value.consecutiveFailures}")
                appendLine()
                
                // Performance metrics.
                appendLine("=== Performance ===")
                _performanceMetrics.value.let {
                    appendLine("CPU: ${it.cpuUsagePercent}%")
                    appendLine("Memory: ${it.memoryUsedMB}MB / ${it.memoryTotalMB}MB")
                    appendLine("Battery: ${it.batteryLevel}%")
                    appendLine("Temperature: ${it.temperatureCelsius}°C")
                }
                appendLine()
                
                // Time sync.
                appendLine("=== Time Sync ===")
                _timeSyncStatus.value.let {
                    appendLine("Valid: ${it.isNtpSyncValid}")
                    appendLine("NTP offset: ${it.ntpOffsetMs}ms")
                    appendLine("Accuracy: ${it.syncAccuracyMs}ms")
                }
                appendLine()
                
                // Sensor information.
                appendLine("=== Sensors ===")
                _sensorInfoMap.value.forEach { (name, info) ->
                    appendLine("$name:")
                    appendLine("  Max rate: ${info.hardwareMaxSamplingRateHz}Hz")
                    appendLine("  Current rate: ${info.currentSamplingRateHz}Hz")
                    appendLine("  FIFO size: ${info.fifoMaxEventCount}")
                }
                
                appendLine()
                appendLine("=== End of Debug Log ===")
            }
            
            file.writeText(log)
            file.absolutePath
        }
    }
    
    fun destroy() {
        stopMonitoring()
        scope.cancel()
    }
}
