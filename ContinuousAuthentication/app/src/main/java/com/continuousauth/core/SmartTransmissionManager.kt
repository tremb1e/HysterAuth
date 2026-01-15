package com.continuousauth.core

import android.content.Context
import com.continuousauth.buffer.InMemoryBuffer
import com.continuousauth.chunking.ChunkingManager
import com.continuousauth.compression.CompressionManager
import com.continuousauth.crypto.EnvelopeCryptoBox
import com.continuousauth.data.DataPacketBuilder
import com.continuousauth.database.BatchMetadata
import com.continuousauth.database.BatchStatus
import com.continuousauth.monitor.SystemMonitor
import com.continuousauth.network.UploadManager
import com.continuousauth.observability.MetricsUploader
import com.continuousauth.privacy.PrivacyManager
import com.continuousauth.processing.SensorDataProcessor
import com.continuousauth.proto.DataPacket
import com.continuousauth.proto.SerializedSensorBatch
import com.continuousauth.sensor.SensorCollector
import com.continuousauth.storage.FileQueueManager
import com.continuousauth.time.EnhancedTimeSync
import com.continuousauth.utils.UserIdManager
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Smart transmission manager - central coordinator for the data pipeline.
 *
 * Coordinates the end-to-end flow from sensor collection to server transmission.
 */
@Singleton
class SmartTransmissionManager @Inject constructor(
    @ApplicationContext private val context: Context,
    private val sensorCollector: SensorCollector,
    private val sensorDataProcessor: SensorDataProcessor,
    private val dataPacketBuilder: DataPacketBuilder,
    private val compressionManager: CompressionManager,
    private val cryptoBox: EnvelopeCryptoBox,
    private val inMemoryBuffer: InMemoryBuffer,
    private val fileQueueManager: FileQueueManager,
    private val uploadManager: UploadManager,
    private val chunkingManager: ChunkingManager,
    private val timeSync: EnhancedTimeSync,
    private val systemMonitor: SystemMonitor,
    private val metricsUploader: MetricsUploader,
    private val privacyManager: PrivacyManager,
    private val userIdManager: UserIdManager
) {
    companion object {
        private const val TAG = "SmartTransmissionManager"
        private const val DEFAULT_BATCH_INTERVAL_MS = 1000L // Default 1-second window
        private const val MAX_PAYLOAD_SIZE_BYTES = 10 * 1024 * 1024 // 10MB
        private const val MEMORY_THRESHOLD = 0.8f // Memory usage threshold
        private const val BATTERY_DRAIN_THRESHOLD = 5f // Battery drain threshold
        private const val SESSION_RENEWAL_INTERVAL_MS = 3600000L // 1 hour
    }

    // Manager state
    private var isRunning = AtomicBoolean(false)
    private var isPaused = AtomicBoolean(false)
    private val packetSequenceNumber = AtomicLong(0)
    
    // Config parameters (dynamically adjustable)
    private var batchIntervalMs = DEFAULT_BATCH_INTERVAL_MS
    private var maxPayloadSizeBytes = MAX_PAYLOAD_SIZE_BYTES
    private var compressionEnabled = true
    private var encryptionEnabled = true
    
    // Coroutine management
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var collectionJob: Job? = null
    private var processingJob: Job? = null
    private var uploadJob: Job? = null
    private var monitoringJob: Job? = null
    private var sessionRenewalJob: Job? = null
    
    // Performance stats
    private val processedPackets = AtomicLong(0)
    private val uploadedPackets = AtomicLong(0)
    private val failedPackets = AtomicLong(0)
    
    // Current session info
    private var currentSessionId: String = ""
    private var lastSessionRenewalTime = 0L

    /**
     * Start the smart transmission manager.
     */
    suspend fun start() {
        if (isRunning.getAndSet(true)) {
            Log.w(TAG, "Already running")
            return
        }
        
        Log.i(TAG, "Starting SmartTransmissionManager")
        
        try {
            // 1. Initialize components
            initializeComponents()
            
            // 2. Start periodic time sync
            timeSync.startPeriodicSync()
            
            // 3. Create a new session
            renewSession()
            
            // 4. Start core coroutines
            startCoreCoroutines()
            
            // 5. Start monitoring
            startMonitoring()
            
            Log.i(TAG, "SmartTransmissionManager started successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start SmartTransmissionManager", e)
            isRunning.set(false)
            throw e
        }
    }

    /**
     * Stop the smart transmission manager.
     */
    suspend fun stop() {
        if (!isRunning.getAndSet(false)) {
            Log.w(TAG, "Not running")
            return
        }
        
        Log.i(TAG, "Stopping SmartTransmissionManager")
        
        try {
            // 1. Stop all coroutines
            stopCoreCoroutines()
            
            // 2. Flush buffers
            flushBuffers()
            
            // 3. Wait for pending uploads
            // Wait for pending uploads
            delay(1000)
            
            // 4. Stop components
            sensorCollector.stopCollection()
            timeSync.stopPeriodicSync()
            
            // 5. Upload final metrics
            uploadFinalMetrics()
            
            Log.i(TAG, "SmartTransmissionManager stopped successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error during shutdown", e)
        }
    }

    /**
     * Pause data collection (keep the manager running).
     */
    suspend fun pause() {
        if (!isRunning.get() || isPaused.getAndSet(true)) {
            return
        }
        
        Log.i(TAG, "Pausing data collection")
        // Pause by stopping sensor collection; resume starts it again.
        sensorCollector.stopCollection()
    }

    /**
     * Resume data collection.
     */
    suspend fun resume() {
        if (!isRunning.get() || !isPaused.getAndSet(false)) {
            return
        }
        
        Log.i(TAG, "Resuming data collection")
        // Resume by starting sensor collection again.
        sensorCollector.startCollection()
    }

    private fun getConfiguredServerEndpoint(): String? {
        val prefs = context.getSharedPreferences("server_config", Context.MODE_PRIVATE)
        val ip = prefs.getString("server_ip", "")?.trim().orEmpty()
        val port = prefs.getInt("server_port", 0)
        if (ip.isBlank() || port !in 1..65535) {
            return null
        }
        return "$ip:$port"
    }

    /**
     * Initialize components.
     */
    private suspend fun initializeComponents() {
        // Initialize crypto module
        cryptoBox.initialize()
        
        // Buffer doesn't need initialization - it's ready to use
        
        // FileQueueManager handles directory creation automatically
        
        // UploadManager uses start() method, not initialize()
        // Will be started when needed
    }

    /**
     * Start core coroutines.
     */
    private fun startCoreCoroutines() {
        // 1. Sensor collection coroutine
        collectionJob = scope.launch {
            startDataCollection()
        }
        
        // 2. Data processing coroutine
        processingJob = scope.launch {
            startDataProcessing()
        }
        
        // 3. Upload coroutine
        uploadJob = scope.launch {
            startDataUpload()
        }
        
        // 4. Session renewal coroutine
        sessionRenewalJob = scope.launch {
            startSessionRenewal()
        }
    }

    /**
     * Stop core coroutines.
     */
    private fun stopCoreCoroutines() {
        collectionJob?.cancel()
        processingJob?.cancel()
        uploadJob?.cancel()
        sessionRenewalJob?.cancel()
        monitoringJob?.cancel()
        
        // Wait for coroutines to finish
        runBlocking {
            collectionJob?.join()
            processingJob?.join()
            uploadJob?.join()
            sessionRenewalJob?.join()
            monitoringJob?.join()
        }
    }

    /**
     * Sensor collection coroutine.
     */
    private suspend fun startDataCollection() {
        Log.d(TAG, "Starting data collection coroutine")
        
        try {
            sensorCollector.startCollection()
            
            // Process sensor data from the flow
            sensorCollector.getSensorDataFlow()
                .catch { e ->
                    Log.e(TAG, "Error in sensor collection", e)
                    handleCollectionError(e)
                }
                .collect { sensorSample ->
                    if (!isPaused.get()) {
                        // Process individual sensor sample
                        processSensorSample(sensorSample)
                    }
                }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start sensor collection", e)
            handleCollectionError(e)
        }
    }

    /**
     * Data processing coroutine.
     */
    private suspend fun startDataProcessing() {
        Log.d(TAG, "Starting data processing coroutine")
        
        while (isRunning.get()) {
            try {
                // Dequeue a batch from the in-memory buffer
                val batch = withTimeoutOrNull(batchIntervalMs) {
                    val packets = inMemoryBuffer.dequeue(1)
                    packets.firstOrNull()
                }
                
                if (batch != null) {
                    processDataBatch(batch)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error in data processing", e)
                delay(100) // Brief backoff after errors
            }
        }
    }

    /**
     * Upload coroutine.
     */
    private suspend fun startDataUpload() {
        Log.d(TAG, "Starting data upload coroutine")
        
        try {
            val endpoint = getConfiguredServerEndpoint()
            if (endpoint == null) {
                Log.w(TAG, "Server endpoint not configured; upload manager will not start")
                return
            }

            // Start upload manager.
            val started = uploadManager.start(endpoint)
            if (started) {
                // Monitor upload results if needed
                // Upload results are handled internally by the UploadManager
                Log.d(TAG, "Upload manager started successfully")
            } else {
                Log.e(TAG, "Failed to start upload manager")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error starting upload manager", e)
            handleUploadError(e)
        }
    }

    /**
     * Session renewal coroutine.
     */
    private suspend fun startSessionRenewal() {
        while (isRunning.get()) {
            delay(SESSION_RENEWAL_INTERVAL_MS)
            renewSession()
        }
    }

    /**
     * Process a sensor sample.
     */
    private suspend fun processSensorSample(sensorSample: com.continuousauth.model.SensorSample) {
        try {
            // 1. Check privacy consent
            // Check consent using state flow value
            if (privacyManager.consentState.value != com.continuousauth.privacy.ConsentState.GRANTED) {
                Log.w(TAG, "No user consent, skipping batch")
                return
            }
            
            // 2. Build data packet from sensor sample
            val packet = dataPacketBuilder.buildDataPacket(
                sensorSamples = listOf(sensorSample),
                encryptedPayload = ByteArray(0), // Will be filled later
                userId = userIdManager.getUserId(),
                sessionId = currentSessionId
            )
            
            // 3. Process the packet
            processDataPacket(packet)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing sensor batch", e)
        }
    }

    /**
     * Process a batch packet.
     */
    private suspend fun processDataBatch(batch: DataPacket) {
        try {
            // Packets dequeued from the buffer are already complete and ready for upload.
            Log.d(TAG, "Processing batch packet: ${batch.packetId}")
        } catch (e: Exception) {
            Log.e(TAG, "Error processing data batch", e)
        }
    }

    /**
     * Process a data packet.
     */
    private suspend fun processDataPacket(packet: DataPacket) {
        try {
            // 1. Assign sequence number
            val sequencedPacket = packet.toBuilder()
                .setPacketSeqNo(packetSequenceNumber.incrementAndGet())
                .build()
            
            // 2. Serialize
            val serialized = sequencedPacket.toByteArray()
            
            // 3. Compress (if enabled)
            val compressed = if (compressionEnabled) {
                compressionManager.compress(serialized)
            } else {
                serialized
            }
            
            // 4. Encrypt (if enabled)
            val encrypted = if (encryptionEnabled && compressed != null) {
                // Build AAD from packet metadata
                val aadData = sequencedPacket.packetId.toByteArray() // Use packetId as associated data.
                cryptoBox.encrypt(compressed, aadData)
            } else {
                compressed
            }
            
            // 5. Chunk if oversized
            if (encrypted?.size ?: 0 > maxPayloadSizeBytes) {
                handleLargePayload(encrypted!!, sequencedPacket)
            } else {
                // 6. Put encrypted packet in buffer
                val finalPacket = sequencedPacket.toBuilder()
                    .setEncryptedSensorPayload(com.google.protobuf.ByteString.copyFrom(encrypted ?: ByteArray(0)))
                    .build()
                
                inMemoryBuffer.enqueue(finalPacket)
            }
            
            // 7. Update stats
            processedPackets.incrementAndGet()
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing data packet", e)
            failedPackets.incrementAndGet()
        }
    }

    // processRawData method removed as it's no longer needed

    /**
     * Handle oversized payloads.
     */
    private suspend fun handleLargePayload(data: ByteArray, packet: DataPacket) {
        // Chunking is currently disabled; enqueue the full payload as-is.
        Log.w(TAG, "Payload exceeds max size (${data.size} > $maxPayloadSizeBytes); enqueuing full payload")
        val packetWithData = packet.toBuilder()
            .setEncryptedSensorPayload(com.google.protobuf.ByteString.copyFrom(data))
            .build()
        inMemoryBuffer.enqueue(packetWithData)
    }

    // queueForUpload method removed as it's replaced with direct buffer usage

    /**
     * Flush buffers.
     */
    private suspend fun flushBuffers() {
        Log.d(TAG, "Flushing buffers")
        
        // 1. Flush in-memory buffer to disk
        while (!inMemoryBuffer.isEmpty()) {
            val packets = inMemoryBuffer.dequeue()
            packets.forEach { packet ->
                try {
                    val data = packet.toByteArray()
                    val metadata = BatchMetadata(
                        packetId = packet.packetId,
                        filePath = "",
                        status = BatchStatus.PENDING,
                        createdTime = System.currentTimeMillis(),
                        fileSize = 0L,
                        sampleCount = 0,
                        transmissionMode = packet.metadata.transmissionProfile.ifBlank { "UNRESTRICTED" },
                        ntpOffset = packet.ntpOffsetMs.takeIf { it != 0L },
                        baseWallMs = packet.baseWallMs,
                        deviceUptimeNs = packet.deviceUptimeNs,
                        sequenceNumber = packet.packetSeqNo,
                        sessionId = currentSessionId,
                        deviceId = packet.deviceIdHash
                    )
                    val result = fileQueueManager.saveDataPacket(packet.packetId, data, metadata)
                    if (result.isSuccess) {
                        Log.d(TAG, "Flushed packet to disk queue: ${packet.packetId}")
                    } else {
                        Log.e(TAG, "Failed to flush packet to disk queue: ${packet.packetId}", result.exceptionOrNull())
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error handling packet for file queue", e)
                }
            }
        }
        
        // 2. Upload manager handles pending uploads internally
    }

    /**
     * Renew the current session.
     */
    private suspend fun renewSession() {
        val now = System.currentTimeMillis()
        if (now - lastSessionRenewalTime < SESSION_RENEWAL_INTERVAL_MS / 2) {
            // Avoid frequent renewals
            return
        }
        
        Log.d(TAG, "Renewing session")
        
        // 1. Generate a new session ID
        currentSessionId = generateSessionId()
        
        // 2. Rotate the session key (if needed)
        cryptoBox.rotateSessionKey()
        
        // 3. Update timestamps
        lastSessionRenewalTime = now
        
        Log.i(TAG, "Session renewed: $currentSessionId")
    }

    /**
     * Generate a session ID.
     */
    private fun generateSessionId(): String {
        return "session_${System.currentTimeMillis()}_${(Math.random() * 10000).toInt()}"
    }

    /**
     * Start monitoring.
     */
    private fun startMonitoring() {
        monitoringJob = scope.launch {
            while (isRunning.get()) {
                delay(10000) // Sample every 10 seconds
                
                val stats = TransmissionStats(
                    processedPackets = processedPackets.get(),
                    uploadedPackets = uploadedPackets.get(),
                    failedPackets = failedPackets.get(),
                    memoryUsage = 0.5f, // TODO: Replace with real memory usage.
                    cpuUsage = 0.1f, // TODO: Replace with real CPU usage.
                    batteryLevel = 80, // TODO: Replace with real battery level.
                    isRunning = isRunning.get(),
                    isPaused = isPaused.get()
                )
                
                Log.d(TAG, "Transmission stats: $stats")
                
                // Dynamic tuning
                optimizeBasedOnPerformance(stats)
            }
        }
    }

    /**
     * Tune settings based on performance signals.
     */
    private fun optimizeBasedOnPerformance(stats: TransmissionStats) {
        // Memory pressure handling
        if (stats.memoryUsage > MEMORY_THRESHOLD) {
            Log.w(TAG, "High memory usage: ${stats.memoryUsage}")
            // Reduce batch interval
            batchIntervalMs = (batchIntervalMs * 0.8).toLong()
            // Enable compression
            compressionEnabled = true
        }
        
        // Battery optimizations
        if (stats.batteryLevel < 20) {
            Log.w(TAG, "Low battery: ${stats.batteryLevel}%")
            // Increase batch interval
            batchIntervalMs = (batchIntervalMs * 1.5).toLong()
        }
        
        // Failure rate handling
        val failureRate = if (stats.processedPackets > 0) {
            stats.failedPackets.toFloat() / stats.processedPackets
        } else 0f
        
        if (failureRate > 0.1f) {
            Log.w(TAG, "High failure rate: ${failureRate * 100}%")
            // Restart UploadManager to recover from transient failures.
            scope.launch {
                try {
                    uploadManager.stop()
                    delay(1000) // Brief delay
                    val endpoint = getConfiguredServerEndpoint()
                    if (endpoint == null) {
                        Log.w(TAG, "Server endpoint not configured; cannot restart UploadManager")
                        return@launch
                    }
                    uploadManager.start(endpoint)
                } catch (e: Exception) {
                    Log.e(TAG, "Error during reconnect", e)
                }
            }
        }
    }

    /**
     * Upload final metrics.
     */
    private suspend fun uploadFinalMetrics() {
        val metrics = mapOf(
            "total_processed" to processedPackets.get(),
            "total_uploaded" to uploadedPackets.get(),
            "total_failed" to failedPackets.get(),
            "session_id" to currentSessionId
        )
        
        Log.d(TAG, "Final metrics snapshot: $metrics")
    }

    /**
     * Handle collection errors.
     */
    private suspend fun handleCollectionError(error: Throwable) {
        Log.e(TAG, "Collection error", error)
        
        // Retry policy based on error type
        when (error) {
            is SecurityException -> {
                // Permission issue; pause collection
                pause()
            }
            is OutOfMemoryError -> {
                // Out of memory; flush buffers
                flushBuffers()
            }
            else -> {
                // Other errors; backoff and retry
                delay(5000)
                if (isRunning.get()) {
                    collectionJob = scope.launch {
                        startDataCollection()
                    }
                }
            }
        }
    }

    /**
     * Handle upload errors.
     */
    private suspend fun handleUploadError(error: Throwable) {
        Log.e(TAG, "Upload error", error)
        
        // Restart UploadManager to recover from errors.
        scope.launch {
            try {
                uploadManager.stop()
                delay(1000)
                val endpoint = getConfiguredServerEndpoint()
                if (endpoint == null) {
                    Log.w(TAG, "Server endpoint not configured; cannot restart UploadManager")
                    return@launch
                }
                uploadManager.start(endpoint)
            } catch (e: Exception) {
                Log.e(TAG, "Error during upload manager restart", e)
            }
        }
    }

    /**
     * Update transmission policy.
     */
    fun updatePolicy(
        batchInterval: Long? = null,
        maxPayloadSize: Int? = null,
        compressionEnabled: Boolean? = null,
        encryptionEnabled: Boolean? = null
    ) {
        batchInterval?.let { 
            this.batchIntervalMs = it
            Log.i(TAG, "Updated batch interval: $it ms")
        }
        
        maxPayloadSize?.let { 
            this.maxPayloadSizeBytes = it
            Log.i(TAG, "Updated max payload size: $it bytes")
        }
        
        compressionEnabled?.let { 
            this.compressionEnabled = it
            Log.i(TAG, "Compression ${if (it) "enabled" else "disabled"}")
        }
        
        encryptionEnabled?.let { 
            this.encryptionEnabled = it
            Log.i(TAG, "Encryption ${if (it) "enabled" else "disabled"}")
        }
    }

    /**
     * Get transmission stats.
     */
    fun getStats(): TransmissionStats {
        return TransmissionStats(
            processedPackets = processedPackets.get(),
            uploadedPackets = uploadedPackets.get(),
            failedPackets = failedPackets.get(),
            memoryUsage = 0.5f, // TODO: Replace with real memory usage.
            cpuUsage = 0.1f, // TODO: Replace with real CPU usage.
            batteryLevel = 80, // TODO: Replace with real battery level.
            isRunning = isRunning.get(),
            isPaused = isPaused.get()
        )
    }

    /**
     * Transmission statistics model.
     */
    data class TransmissionStats(
        val processedPackets: Long,
        val uploadedPackets: Long,
        val failedPackets: Long,
        val memoryUsage: Float,
        val cpuUsage: Float,
        val batteryLevel: Int,
        val isRunning: Boolean,
        val isPaused: Boolean
    )
}
