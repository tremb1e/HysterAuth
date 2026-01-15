package com.continuousauth.network

import android.util.Log
import com.continuousauth.buffer.InMemoryBuffer
import com.continuousauth.policy.PolicyManager
import com.continuousauth.storage.FileQueueManager
import com.continuousauth.proto.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Upload manager.
 *
 * Coordinates packet upload, ACK handling, and policy updates.
 */
@Singleton
class UploadManager @Inject constructor(
    private val uploader: Uploader,
    private val inMemoryBuffer: InMemoryBuffer,
    private val policyManager: PolicyManager,
    private val fileQueueManager: FileQueueManager,
    private val networkEnvironmentDetector: NetworkEnvironmentDetector
) {
    
    companion object {
        private const val TAG = "UploadManager"
        private const val UPLOAD_BATCH_SIZE = 50
        private const val UPLOAD_INTERVAL_MS = 1000L // Upload interval (1s)
    }
    
    // State.
    private val isRunning = AtomicBoolean(false)
    private val uploadedPackets = AtomicLong(0L)
    
    // Coroutine scope.
    private val managerScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var uploadJob: Job? = null
    private var directiveJob: Job? = null
    
    // Policy.
    private var currentPolicy: PolicyUpdate? = null
    private var policyUpdateCallback: ((PolicyUpdate) -> Unit)? = null
    
    // Network mode.
    private var isWifiOnlyMode = false
    
    /**
     * Starts the upload manager.
     */
    suspend fun start(serverEndpoint: String): Boolean {
        if (isRunning.get()) {
            Log.w(TAG, "UploadManager is already running")
            return true
        }
        
        // Connect to server.
        if (!uploader.connect(serverEndpoint)) {
            Log.e(TAG, "Failed to connect to server: $serverEndpoint")
            return false
        }
        
        isRunning.set(true)
        
        // Start server directive processing.
        startDirectiveProcessing()
        
        // Start upload loop.
        startUploadLoop()
        
        Log.i(TAG, "UploadManager started")
        return true
    }
    
    /**
     * Stops the upload manager.
     */
    suspend fun stop() {
        if (!isRunning.get()) {
            return
        }
        
        isRunning.set(false)
        
        // Cancel jobs.
        uploadJob?.cancel()
        directiveJob?.cancel()
        
        // Disconnect.
        uploader.disconnect()
        
        Log.i(TAG, "UploadManager stopped")
    }
    
    /**
     * Sends a single packet manually.
     */
    suspend fun sendDataPacket(dataPacket: DataPacket): Boolean {
        if (!isRunning.get()) {
            Log.w(TAG, "UploadManager is not running; cannot send data packet")
            return false
        }
        
        return uploader.sendDataPacket(dataPacket)
    }
    
    /**
     * Sets a policy update callback.
     */
    fun setPolicyUpdateCallback(callback: (PolicyUpdate) -> Unit) {
        policyUpdateCallback = callback
    }
    
    /**
     * Returns current policy.
     */
    fun getCurrentPolicy(): PolicyUpdate? = currentPolicy
    
    /**
     * Returns upload status.
     */
    fun getUploadStatus(): UploadStatus {
        val statusDetail = uploader.getConnectionStatus()
        val status = ConnectionStatus.values().find { it.name == statusDetail.state } 
            ?: ConnectionStatus.DISCONNECTED
        
        val queueStats = runBlocking { 
            val stats = fileQueueManager.getQueueStatistics()
            com.continuousauth.storage.QueueStats(
                totalPackets = stats.totalPackets,
                pendingPackets = stats.pendingPackets, 
                uploadedPackets = stats.uploadedPackets,
                corruptedPackets = stats.corruptedPackets,
                totalSizeBytes = stats.totalSizeBytes
            )
        }
        
        return UploadStatus(
            isRunning = isRunning.get(),
            connectionStatus = status,
            uploadedPackets = uploadedPackets.get(),
            bufferedPackets = inMemoryBuffer.getSize(),
            connectionStats = uploader.getConnectionStats(),
            fileQueueStats = queueStats
        )
    }
    
    /**
     * Starts server directive processing.
     */
    private fun startDirectiveProcessing() {
        directiveJob = managerScope.launch {
            try {
                uploader.getServerDirectiveFlow()
                    .catch { e ->
                        Log.e(TAG, "Error while collecting server directives", e)
                    }
                    .collect { directive ->
                        processServerDirective(directive)
                    }
            } catch (e: CancellationException) {
                Log.i(TAG, "Server directive processing cancelled")
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Server directive processing error", e)
            }
        }
    }
    
    /**
     * Starts the upload loop.
     */
    private fun startUploadLoop() {
        uploadJob = managerScope.launch {
            while (isRunning.get() && isActive) {
                try {
                    uploadBatchFromBuffer()
                    delay(UPLOAD_INTERVAL_MS)
                } catch (e: CancellationException) {
                    Log.i(TAG, "Upload loop cancelled")
                    throw e
                } catch (e: Exception) {
                    Log.e(TAG, "Upload loop error", e)
                    delay(5000L) // Retry after delay
                }
            }
        }
    }
    
    /**
     * Uploads a batch from buffer.
     */
    private suspend fun uploadBatchFromBuffer() {
        // WiFi-only mode check.
        if (isWifiOnlyMode && !networkEnvironmentDetector.isWifiConnected()) {
            Log.v(TAG, "WiFi-only mode enabled; not on WiFi, skipping upload")
            return
        }
        
        // Prefer in-memory buffer.
        val packets = if (!inMemoryBuffer.isEmpty()) {
            inMemoryBuffer.dequeue(UPLOAD_BATCH_SIZE)
        } else {
            // If memory is empty, do not pull from disk queue (simplified).
            emptyList()
        }
        
        if (packets.isEmpty()) {
            return
        }
        
        try {
            Log.d(TAG, "Uploading batch - count: ${packets.size}")
            
            // Send packets one by one.
            var successCount = 0
            
            for (packet in packets) {
                if (uploader.sendDataPacket(packet)) {
                    successCount++
                    // If present in disk queue, delete it.
                    fileQueueManager.deleteDataPacket(packet.packetId)
                } else {
                    Log.w(TAG, "Packet send failed: ${packet.packetId}")
                    // Keep the packet in the queue for future retries.
                }
            }
            
            uploadedPackets.addAndGet(successCount.toLong())
            Log.d(TAG, "Batch upload complete - success: $successCount/${packets.size}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Batch upload error", e)
        }
    }
    
    /**
     * Processes a server directive.
     */
    private suspend fun processServerDirective(directive: ServerDirective) {
        try {
            when {
                directive.hasAck() -> {
                    processAck(directive.ack)
                }
                directive.hasPolicy() -> {
                    processPolicyUpdate(directive.policy)
                }
                else -> {
                    Log.w(TAG, "Unknown server directive type received")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing server directive", e)
        }
    }
    
    /**
     * Processes ACK.
     */
    private suspend fun processAck(ack: Ack) {
        Log.d(TAG, "Processing ACK - packetId: ${ack.packetId}, success: ${ack.success}")
        
        if (ack.success) {
            // Update ack status and record server timestamp.
            fileQueueManager.updateAckStatus(
                packetId = ack.packetId,
                serverTimestamp = ack.creationServerTs
            )
            // Then delete local file.
            fileQueueManager.deleteDataPacket(ack.packetId)
            Log.d(TAG, "ACK succeeded; status updated and deleted: ${ack.packetId}")
        } else {
            // Failure handling.
            when (ack.errorCode) {
                "DUPLICATE" -> {
                    // Duplicate packets: update status and delete.
                    fileQueueManager.updateAckStatus(
                        packetId = ack.packetId,
                        serverTimestamp = ack.creationServerTs
                    )
                    fileQueueManager.deleteDataPacket(ack.packetId)
                    Log.w(TAG, "Duplicate ACK; status updated and deleted: ${ack.packetId}")
                }
                "INVALID_FORMAT" -> {
                    // Invalid format: mark failed and delete.
                    fileQueueManager.updateFailedStatus(
                        packetId = ack.packetId,
                        error = ack.errorCode
                    )
                    fileQueueManager.deleteDataPacket(ack.packetId)
                    Log.e(TAG, "Invalid packet format; marked failed and deleted: ${ack.packetId}")
                }
                "DECRYPTION_FAILED" -> {
                    // Decryption failed: mark failed and delete.
                    fileQueueManager.updateFailedStatus(
                        packetId = ack.packetId,
                        error = ack.errorCode
                    )
                    fileQueueManager.deleteDataPacket(ack.packetId)
                    Log.e(TAG, "Packet decryption failed; marked failed and deleted: ${ack.packetId}")
                }
                "SERVER_ERROR" -> {
                    // Server error: update retry info and keep the packet for retry.
                    fileQueueManager.updateRetryInfo(
                        packetId = ack.packetId,
                        error = ack.errorCode
                    )
                    // Use retry-after if provided.
                    if (ack.retryAfterMs > 0) {
                        Log.e(TAG, "Server error; retry after ${ack.retryAfterMs}ms: ${ack.packetId}")
                    } else {
                        Log.e(TAG, "Server error; retry info updated: ${ack.packetId}")
                    }
                }
                else -> {
                    Log.w(TAG, "Unknown error code: ${ack.errorCode}, packetId: ${ack.packetId}")
                }
            }
        }
    }
    
    /**
     * Processes policy updates.
     */
    private suspend fun processPolicyUpdate(policyUpdate: PolicyUpdate) {
        Log.i(TAG, "Policy update received - policyId: ${policyUpdate.policyId}")
        
        try {
            // Apply to PolicyManager.
            policyManager.applyPolicyUpdate(policyUpdate)
            
            currentPolicy = policyUpdate
            
            // Notify app layer.
            policyUpdateCallback?.invoke(policyUpdate)
            
            // Log policy details.
            Log.d(TAG, "Policy update details:")
            Log.d(TAG, "  Version: ${policyUpdate.policyVersion}")
            if (policyUpdate.batchIntervalMs > 0) {
                Log.d(TAG, "  Batch interval: ${policyUpdate.batchIntervalMs}ms")
            }
            if (policyUpdate.anomalyConfig != null) {
                val ac = policyUpdate.anomalyConfig
                Log.d(TAG, "  Anomaly detection - enabled: ${ac.enabled}, multiplier: ${ac.thresholdMultiplier}")
            }
            if (policyUpdate.transmissionProfile.isNotEmpty()) {
                Log.d(TAG, "  Transmission profile: ${policyUpdate.transmissionProfile}")
            }
            
            Log.i(TAG, "Policy update processed")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process policy update", e)
        }
    }
    
    /**
     * Sets offline mode.
     */
    fun setOfflineMode(enabled: Boolean) {
        Log.i(TAG, "Set offline mode: $enabled")
        // TODO: Implement offline mode.
    }
    
    /**
     * Reduces upload rate.
     */
    fun reduceUploadRate() {
        Log.i(TAG, "Reducing upload rate")
        // TODO: Implement rate limiting.
    }
    
    /**
     * Resumes normal upload rate.
     */
    fun resumeNormalRate() {
        Log.i(TAG, "Resuming normal upload rate")
        // TODO: Implement rate restoration.
    }
    
    /**
     * Retries a packet.
     */
    suspend fun retryPacket(packet: DataPacket) {
        Log.i(TAG, "Retrying packet: ${packet.packetId}")
        // TODO: Implement retry logic.
    }
    
    /**
     * Enables/disables WiFi-only mode.
     */
    fun setWifiOnlyMode(enabled: Boolean) {
        Log.i(TAG, "Set WiFi-only mode: $enabled")
        isWifiOnlyMode = enabled
        
        // If WiFi-only mode is enabled and we are not on WiFi, pause uploads.
        if (enabled && !networkEnvironmentDetector.isWifiConnected()) {
            Log.w(TAG, "WiFi-only mode enabled but not on WiFi; pausing uploads")
            // Temporarily stop upload loop.
            uploadJob?.cancel()
        } else if (!enabled || networkEnvironmentDetector.isWifiConnected()) {
            // If WiFi-only is disabled or we are on WiFi, resume uploads.
            if (isRunning.get() && uploadJob?.isActive != true) {
                startUploadLoop()
            }
        }
    }
    
    /**
     * Enables/disables local caching.
     */
    fun setLocalCachingEnabled(enabled: Boolean) {
        Log.i(TAG, "Set local caching: $enabled")
        // TODO: Implement local caching controls.
    }
    
    /**
     * Returns the number of pending packets.
     */
    fun getPendingPacketsCount(): Int {
        // Simplified: return in-memory buffer size.
        return inMemoryBuffer.getSize()
    }
    
    /**
     * Returns the last packet sequence number.
     */
    fun getLastPacketSeqNo(): Long {
        return uploadedPackets.get()
    }
}

/**
 * Upload status.
 */
data class UploadStatus(
    val isRunning: Boolean,
    val connectionStatus: ConnectionStatus,
    val uploadedPackets: Long,
    val bufferedPackets: Int,
    val connectionStats: ConnectionStats,
    val fileQueueStats: com.continuousauth.storage.QueueStats? = null
)
