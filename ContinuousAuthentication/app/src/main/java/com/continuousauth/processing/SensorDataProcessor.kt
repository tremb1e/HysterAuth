package com.continuousauth.processing

import com.continuousauth.chunking.ChunkingManager
import com.continuousauth.compression.CompressionManager
import com.continuousauth.crypto.AADBuilder
import com.continuousauth.crypto.CryptoBox
import com.continuousauth.data.DataPacketBuilder
import com.continuousauth.model.SensorSample
import com.continuousauth.proto.DataPacket
import com.google.protobuf.InvalidProtocolBufferException
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.*
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Sensor data processor.
 *
 * Consumes the sensor stream and produces encrypted data packets:
 * serialize -> compress -> encrypt -> chunk (if needed).
 */
@Singleton
class SensorDataProcessor @Inject constructor(
    private val cryptoBox: CryptoBox,
    private val envelopeCryptoBox: com.continuousauth.crypto.EnvelopeCryptoBox,
    private val aadBuilder: AADBuilder,
    private val dataPacketBuilder: DataPacketBuilder,
    private val compressionManager: CompressionManager,
    private val chunkingManager: ChunkingManager
) {
    
    companion object {
        private const val TAG = "SensorDataProcessor"
    }
    
    // Internal state
    private val isProcessing = AtomicBoolean(false)
    private val processedCount = AtomicLong(0L)
    private val encryptedPacketChannel = Channel<DataPacket>(Channel.UNLIMITED)
    
    // Coroutine scope
    private val processingScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var processingJob: Job? = null
    
    // Current session info
    private var currentUserId = ""
    private var currentSessionId = ""
    private var currentTransmissionProfile = "UNRESTRICTED"
    
    /**
     * Start processing the sensor data stream.
     */
    suspend fun startProcessing(
        sensorDataFlow: Flow<SensorSample>,
        userId: String,
        sessionId: String = UUID.randomUUID().toString()
    ): Boolean {
        if (isProcessing.get()) {
            android.util.Log.w(TAG, "Sensor processing is already running")
            return true
        }
        
        // Initialize crypto
        if (!cryptoBox.initialize()) {
            android.util.Log.e(TAG, "Failed to initialize crypto")
            return false
        }
        
        currentUserId = userId
        currentSessionId = sessionId
        
        // Start processing coroutine
        processingJob = processingScope.launch {
            processSensorDataFlow(sensorDataFlow)
        }
        
        isProcessing.set(true)
        android.util.Log.i(TAG, "Sensor processing started - sessionId: $sessionId")
        
        return true
    }
    
    /**
     * Stop processing.
     */
    suspend fun stopProcessing() {
        if (!isProcessing.get()) {
            return
        }
        
        processingJob?.cancel()
        processingJob?.join()
        
        isProcessing.set(false)
        
        android.util.Log.i(TAG, "Sensor processing stopped - total processed: ${processedCount.get()} samples")
    }
    
    /**
     * Get the encrypted packet flow.
     */
    fun getEncryptedPacketFlow(): Flow<DataPacket> {
        return encryptedPacketChannel.receiveAsFlow()
    }
    
    /**
     * Set the transmission profile.
     */
    fun setTransmissionProfile(profile: String) {
        currentTransmissionProfile = profile
        android.util.Log.d(TAG, "Transmission profile updated: $profile")
    }
    
    /**
     * Get processing status.
     */
    fun getProcessingStatus(): ProcessingStatus {
        return ProcessingStatus(
            isProcessing = isProcessing.get(),
            processedSampleCount = processedCount.get(),
            currentSessionId = currentSessionId,
            currentTransmissionProfile = currentTransmissionProfile
        )
    }
    
    /**
     * Core processing loop for the sensor data flow.
     */
    private suspend fun processSensorDataFlow(sensorDataFlow: Flow<SensorSample>) {
        try {
            // Collect sensor samples into batches
            sensorDataFlow
                .buffer() // Buffer to reduce backpressure
                .chunked() // Chunk into time windows
                .collect { sampleBatch ->
                    if (sampleBatch.isNotEmpty()) {
                        processSampleBatch(sampleBatch)
                        processedCount.addAndGet(sampleBatch.size.toLong())
                    }
                }
        } catch (e: CancellationException) {
            android.util.Log.i(TAG, "Sensor processing cancelled")
            throw e
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Sensor processing error", e)
        }
    }
    
    /**
     * Process a batch of sensor samples.
     */
    private suspend fun processSampleBatch(samples: List<SensorSample>) {
        try {
            // 1. Build plaintext sensor batch and serialize
            val sensorBatch = dataPacketBuilder.buildSensorBatch(
                sensorSamples = samples,
                userId = currentUserId,
                sessionId = currentSessionId
            )
            
            val sensorBatchBytes = sensorBatch.toByteArray()
            
            // 2. Compress
            val compressionType = CompressionManager.CompressionType.GZIP // Use GZIP by default
            
            val compressedData = compressionManager.compress(sensorBatchBytes, compressionType)
            if (compressedData == null) {
                android.util.Log.e(TAG, "Compression failed; skipping batch")
                return
            }
            
            // 3. Generate packet ID and build AAD
            val packetId = UUID.randomUUID().toString()
            val appVersion = getAppVersion()
            val packetSeqNo = envelopeCryptoBox.getNextPacketSeqNo()
            val dekKeyId = envelopeCryptoBox.getDekKeyId()
            
            val aad = aadBuilder.buildAAD(
                packetId = packetId,
                packetSeqNo = packetSeqNo,
                dekKeyId = dekKeyId,
                transmissionProfile = currentTransmissionProfile,
                appVersion = appVersion,
                sampleCount = samples.size
            )
            
            // 4. Encrypt the compressed payload
            val encryptedPayload = cryptoBox.encrypt(compressedData, aad)
            
            if (encryptedPayload == null) {
                android.util.Log.e(TAG, "Encryption failed - packetId: $packetId")
                return
            }
            
            // 5. Get encrypted DEK (envelope encryption)
            val encryptedDek = envelopeCryptoBox.getEncryptedDEK()
            
            // 6. Compute SHA-256 checksum
            val sha256 = calculateSha256(encryptedPayload)
            
            // 7. Build final packet (includes compression metadata)
            val dataPacket = dataPacketBuilder.buildDataPacket(
                sensorSamples = samples,
                encryptedPayload = encryptedPayload,
                transmissionProfile = currentTransmissionProfile,
                userId = currentUserId,
                sessionId = currentSessionId,
                encryptedDek = encryptedDek,
                dekKeyId = dekKeyId,
                sha256 = sha256,
                compressionType = compressionManager.getCompressionTypeString(compressionType)
            )
            
            // 8. Chunk if needed
            val packets = if (chunkingManager.needsChunking(encryptedPayload)) {
                val chunkedPackets = chunkingManager.processPacketChunking(dataPacket)
                android.util.Log.i(TAG, "Chunking required; split into ${chunkedPackets.size} chunks")
                chunkedPackets
            } else {
                listOf(dataPacket)
            }
            
            // 9. Emit packets (may contain multiple chunks)
            packets.forEach { packet ->
                encryptedPacketChannel.trySend(packet)
                val chunkInfo = chunkingManager.getChunkInfo(packet)
                android.util.Log.v(TAG, "Packet emitted - $chunkInfo")
            }
            
            android.util.Log.d(
                TAG,
                "Batch processed - packetId: $packetId, samples: ${samples.size}, " +
                    "compression: ${compressionManager.getCompressionTypeString(compressionType)}, " +
                    "chunks: ${packets.size}"
            )
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to process batch", e)
        }
    }
    
    /**
     * Chunk the flow into time windows (default: 1s).
     */
    private fun <T> Flow<T>.chunked(): Flow<List<T>> {
        // Default 1-second window
        return this.chunked(1000) // 1s
    }
    
    /**
     * Chunk the flow into fixed time windows.
     */
    private fun <T> Flow<T>.chunked(windowMillis: Long): Flow<List<T>> = flow {
        val buffer = mutableListOf<T>()
        var lastEmitTime = System.currentTimeMillis()
        
        collect { item ->
            buffer.add(item)
            val currentTime = System.currentTimeMillis()
            
            if (currentTime - lastEmitTime >= windowMillis) {
                if (buffer.isNotEmpty()) {
                    emit(buffer.toList())
                    buffer.clear()
                }
                lastEmitTime = currentTime
            }
        }
        
        // Emit remaining items
        if (buffer.isNotEmpty()) {
            emit(buffer.toList())
        }
    }
    
    /**
     * Returns the app version.
     */
    private fun getAppVersion(): String {
        return "1.0.0" // TODO: Retrieve from PackageManager
    }
    
    /**
     * Returns a device identifier.
     */
    private fun getDeviceId(): String {
        return "device_${System.currentTimeMillis()}" // TODO: Use a stable device identifier
    }
    
    /**
     * Compute SHA-256 hash for the given data.
     */
    private fun calculateSha256(data: ByteArray): ByteArray {
        return try {
            val digest = java.security.MessageDigest.getInstance("SHA-256")
            digest.digest(data)
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to compute SHA-256", e)
            ByteArray(0)
        }
    }
}

/**
 * Processing status model.
 */
data class ProcessingStatus(
    val isProcessing: Boolean,
    val processedSampleCount: Long,
    val currentSessionId: String,
    val currentTransmissionProfile: String
)
