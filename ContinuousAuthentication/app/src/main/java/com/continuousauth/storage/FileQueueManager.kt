package com.continuousauth.storage

import android.content.Context
import android.util.Log
import com.continuousauth.database.BatchMetadata
import com.continuousauth.database.BatchMetadataDao
import com.continuousauth.database.BatchStatus
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * File queue manager.
 *
 * Manages on-disk storage and queue operations for encrypted packets.
 */
@Singleton
class FileQueueManager @Inject constructor(
    @ApplicationContext private val context: Context,
    private val batchMetadataDao: BatchMetadataDao
) {
    
    // Coroutine scope
    private val scope = CoroutineScope(Dispatchers.IO)
    
    companion object {
        private const val TAG = "FileQueueManager"
        private const val QUEUE_DIR = "data_queue"
        private const val MAX_QUEUE_SIZE_MB = 200 // Max queue size: 200MB
        private const val MAX_QUEUE_SIZE_BYTES = MAX_QUEUE_SIZE_MB * 1024 * 1024L
        private const val MIN_FREE_SPACE_MB = 50 // Min free space: 50MB
        private const val MIN_FREE_SPACE_BYTES = MIN_FREE_SPACE_MB * 1024 * 1024L
        private const val CLEANUP_THRESHOLD = 0.9f // Cleanup threshold: 90%
    }
    
    // Queue directory
    private val queueDir: File by lazy {
        File(context.cacheDir, QUEUE_DIR).apply {
            if (!exists()) {
                mkdirs()
            }
        }
    }
    
    // Queue stats
    private val _queueStats = MutableStateFlow(QueueStats())
    val queueStats: StateFlow<QueueStats> = _queueStats.asStateFlow()
    
    // Cleanup in progress flag
    private var isCleaningUp = false
    
    init {
        // Update stats on startup
        scope.launch {
            updateQueueStats()
        }
    }
    
    /**
     * Save an encrypted packet to the file queue.
     *
     * Uses an atomic write: tmp -> fsync -> rename.
     */
    suspend fun saveDataPacket(
        packetId: String,
        encryptedData: ByteArray,
        metadata: BatchMetadata
    ): Result<File> = withContext(Dispatchers.IO) {
        try {
            // Check disk space
            if (!hasEnoughSpace(encryptedData.size.toLong())) {
                // Try to free space
                performCleanup()
                
                // Re-check
                if (!hasEnoughSpace(encryptedData.size.toLong())) {
                    return@withContext Result.failure(
                        InsufficientStorageException("Insufficient disk space")
                    )
                }
            }
            
            // Compute SHA-256
            val sha256 = calculateSHA256(encryptedData)
            
            // Create temp and final files
            val tmpFile = File(queueDir, "$packetId.tmp")
            val finalFile = File(queueDir, "$packetId.dat")
            
            // Atomic write: write to temp file first
            try {
                FileOutputStream(tmpFile).use { fos ->
                    fos.write(encryptedData)
                    // fsync to ensure data is persisted
                    fos.fd.sync()
                }
                
                // Rename to final file (atomic)
                if (!tmpFile.renameTo(finalFile)) {
                    throw Exception("Failed to rename temp file to final file")
                }
            } catch (e: Exception) {
                // Cleanup temp file
                tmpFile.delete()
                throw e
            }
            
            // Update metadata (include SHA-256)
            val updatedMetadata = metadata.copy(
                filePath = finalFile.absolutePath,
                fileSize = finalFile.length(),
                sha256 = sha256
            )
            
            // Persist to database
            batchMetadataDao.insert(updatedMetadata)
            
            // Update stats
            updateQueueStats()
            
            Log.i(TAG, "Packet saved: $packetId, size: ${finalFile.length()} bytes, SHA256: $sha256")
            
            Result.success(finalFile)
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save packet: $packetId", e)
            Result.failure(e)
        }
    }
    
    /**
     * Read an encrypted packet.
     *
     * Validates the SHA-256 checksum if present.
     */
    suspend fun readDataPacket(packetId: String): Result<ByteArray> = withContext(Dispatchers.IO) {
        try {
            // Load metadata from database
            val metadata = batchMetadataDao.getById(packetId)
                ?: return@withContext Result.failure(
                    FileNotFoundException("Packet not found: $packetId")
                )
            
            // Check whether packet is marked corrupt
            if (metadata.status == BatchStatus.CORRUPT) {
                return@withContext Result.failure(
                    Exception("Packet marked corrupt: $packetId")
                )
            }
            
            // Read file
            val file = File(metadata.filePath)
            if (!file.exists()) {
                return@withContext Result.failure(
                    FileNotFoundException("File not found: ${metadata.filePath}")
                )
            }
            
            val data = file.readBytes()
            
            // Validate SHA-256
            if (metadata.sha256 != null) {
                val calculatedSha256 = calculateSHA256(data)
                if (calculatedSha256 != metadata.sha256) {
                    // Mark as corrupt
                    batchMetadataDao.updateStatus(packetId, BatchStatus.CORRUPT)
                    Log.e(
                        TAG,
                        "SHA-256 validation failed: $packetId, expected: ${metadata.sha256}, actual: $calculatedSha256"
                    )
                    return@withContext Result.failure(
                        Exception("SHA-256 validation failed; packet is corrupt")
                    )
                }
            }
            
            Result.success(data)
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read packet: $packetId", e)
            Result.failure(e)
        }
    }
    
    /**
     * Delete a packet.
     */
    suspend fun deleteDataPacket(packetId: String): Boolean = withContext(Dispatchers.IO) {
        try {
            // Load metadata from database
            val metadata = batchMetadataDao.getById(packetId) ?: return@withContext false
            
            // Delete file
            val file = File(metadata.filePath)
            if (file.exists()) {
                file.delete()
            }
            
            // Delete database record
            batchMetadataDao.deleteById(packetId)
            
            // Update stats
            updateQueueStats()
            
            Log.i(TAG, "Packet deleted: $packetId")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to delete packet: $packetId", e)
            false
        }
    }
    
    /**
     * Update packet ACK status and record server timestamp.
     */
    suspend fun updateAckStatus(
        packetId: String, 
        serverTimestamp: Long
    ) = withContext(Dispatchers.IO) {
        try {
            // Mark as ACKNOWLEDGED and persist server timestamp.
            batchMetadataDao.updateAckStatus(
                packetId = packetId,
                status = BatchStatus.ACKNOWLEDGED,
                ackTime = serverTimestamp
            )
            Log.d(TAG, "Packet ACK updated: $packetId, serverTimestamp: $serverTimestamp")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to update ACK status: $packetId", e)
        }
    }
    
    /**
     * Mark packet as failed.
     */
    suspend fun updateFailedStatus(
        packetId: String,
        error: String
    ) = withContext(Dispatchers.IO) {
        try {
            // Mark as FAILED
            batchMetadataDao.updateStatus(packetId, BatchStatus.FAILED)
            // Update retry info
            batchMetadataDao.updateRetryInfo(packetId, error)
            Log.d(TAG, "Packet marked failed: $packetId, error: $error")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to update failed status: $packetId", e)
        }
    }
    
    /**
     * Update retry info.
     */
    suspend fun updateRetryInfo(
        packetId: String,
        error: String
    ) = withContext(Dispatchers.IO) {
        try {
            batchMetadataDao.updateRetryInfo(packetId, error)
            Log.d(TAG, "Packet retry info updated: $packetId, error: $error")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to update retry info: $packetId", e)
        }
    }
    
    /**
     * Bulk delete acknowledged packets.
     */
    suspend fun deleteAcknowledgedPackets() = withContext(Dispatchers.IO) {
        try {
            // Load acknowledged batches
            val acknowledgedBatches = batchMetadataDao.getPendingBatches(
                listOf(BatchStatus.ACKNOWLEDGED)
            )
            
            var deletedCount = 0
            acknowledgedBatches.forEach { batch ->
                val file = File(batch.filePath)
                if (file.exists()) {
                    file.delete()
                    deletedCount++
                }
            }
            
            // Bulk delete database records
            batchMetadataDao.deleteByStatus(BatchStatus.ACKNOWLEDGED)
            
            // Update stats
            updateQueueStats()
            
            Log.i(TAG, "Deleted $deletedCount acknowledged packets")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to delete acknowledged packets", e)
        }
    }
    
    /**
     * Get packets pending upload.
     */
    suspend fun getPendingPackets(): List<BatchMetadata> = withContext(Dispatchers.IO) {
        try {
            batchMetadataDao.getPendingBatches(
                listOf(BatchStatus.PENDING, BatchStatus.FAILED)
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get pending packets", e)
            emptyList()
        }
    }
    
    /**
     * Perform queue cleanup.
     */
    private suspend fun performCleanup() = withContext(Dispatchers.IO) {
        if (isCleaningUp) {
            return@withContext
        }
        
        isCleaningUp = true
        
        try {
            Log.i(TAG, "Starting queue cleanup")
            
            // Get current queue size
            val currentSize = getQueueSizeBytes()
            
            // Compute target size (clean down to 70%)
            val targetSize = (MAX_QUEUE_SIZE_BYTES * 0.7).toLong()
            val needToFree = currentSize - targetSize
            
            if (needToFree <= 0) {
                return@withContext
            }
            
            // Load all batches and sort by creation time
            val allBatches = batchMetadataDao.getPendingBatches(
                BatchStatus.values().toList()
            ).sortedBy { it.createdTime }
            
            var freedSize = 0L
            val toDelete = mutableListOf<BatchMetadata>()
            
            // Delete in priority order: acknowledged -> discarded -> failed -> pending
            val priorityOrder = listOf(
                BatchStatus.ACKNOWLEDGED,
                BatchStatus.DISCARDED,
                BatchStatus.FAILED,
                BatchStatus.PENDING
            )
            
            for (status in priorityOrder) {
                val batchesOfStatus = allBatches.filter { it.status == status }
                
                for (batch in batchesOfStatus) {
                    if (freedSize >= needToFree) {
                        break
                    }
                    
                    toDelete.add(batch)
                    freedSize += batch.fileSize
                }
                
                if (freedSize >= needToFree) {
                    break
                }
            }
            
            // Perform deletion
            toDelete.forEach { batch ->
                val file = File(batch.filePath)
                if (file.exists()) {
                    file.delete()
                }
                batchMetadataDao.updateStatus(batch.packetId, BatchStatus.DISCARDED)
            }
            
            Log.i(TAG, "Queue cleanup complete: deleted ${toDelete.size} packets, freed $freedSize bytes")
            
            // Update stats
            updateQueueStats()
            
        } catch (e: Exception) {
            Log.e(TAG, "Queue cleanup failed", e)
        } finally {
            isCleaningUp = false
        }
    }
    
    /**
     * Check whether there is enough storage space.
     */
    private fun hasEnoughSpace(dataSize: Long): Boolean {
        val currentQueueSize = getQueueSizeBytes()
        val freeSpace = queueDir.freeSpace
        
        // Check queue size limit
        if (currentQueueSize + dataSize > MAX_QUEUE_SIZE_BYTES) {
            return false
        }
        
        // Check free disk space
        if (freeSpace - dataSize < MIN_FREE_SPACE_BYTES) {
            return false
        }
        
        return true
    }
    
    /**
     * Get queue size (bytes).
     */
    private fun getQueueSizeBytes(): Long {
        return queueDir.listFiles()?.sumOf { it.length() } ?: 0L
    }
    
    /**
     * Get the number of pending packets in the queue.
     */
    fun getQueueSize(): Int {
        return runBlocking {
            try {
                batchMetadataDao.getPendingBatches(
                    listOf(BatchStatus.PENDING, BatchStatus.UPLOADING)
                ).size
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get queue size", e)
                0
            }
        }
    }
    
    /**
     * Update queue statistics.
     */
    private suspend fun updateQueueStats() = withContext(Dispatchers.IO) {
        try {
            val totalSize = getQueueSizeBytes()
            val fileCount = queueDir.listFiles()?.size ?: 0
            val statusCounts = batchMetadataDao.getStatusCounts()
            
            val pending = statusCounts.find { it.status == BatchStatus.PENDING }?.count ?: 0
            val uploading = statusCounts.find { it.status == BatchStatus.UPLOADING }?.count ?: 0
            val uploaded = statusCounts.find { it.status == BatchStatus.ACKNOWLEDGED }?.count ?: 0
            val failed = statusCounts.find { it.status == BatchStatus.FAILED }?.count ?: 0
            val corrupted = statusCounts.find { it.status == BatchStatus.CORRUPT }?.count ?: 0
            val total = pending + uploading + uploaded + failed + corrupted
            
            _queueStats.value = QueueStats(
                totalSizeBytes = totalSize,
                fileCount = fileCount,
                pendingCount = pending,
                uploadingCount = uploading,
                failedCount = failed,
                acknowledgedCount = uploaded,
                queueUsagePercent = (totalSize.toFloat() / MAX_QUEUE_SIZE_BYTES * 100).coerceIn(0f, 100f),
                totalPackets = total,
                pendingPackets = pending,
                uploadedPackets = uploaded,
                corruptedPackets = corrupted
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to update queue stats", e)
        }
    }
    
    /**
     * Clear the entire queue.
     */
    suspend fun clearQueue() = withContext(Dispatchers.IO) {
        try {
            // Delete all files
            queueDir.listFiles()?.forEach { it.delete() }
            
            // Clear database
            batchMetadataDao.clearAll()
            
            // Update stats
            updateQueueStats()
            
            Log.i(TAG, "Queue cleared")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to clear queue", e)
        }
    }
    
    /**
     * Get detailed queue statistics.
     */
    suspend fun getQueueStatistics(): QueueStatisticsDetail = withContext(Dispatchers.IO) {
        try {
            val totalSize = getQueueSizeBytes()
            val fileCount = queueDir.listFiles()?.size ?: 0
            val statusCounts = batchMetadataDao.getStatusCounts()
            
            val pending = statusCounts.find { it.status == BatchStatus.PENDING }?.count ?: 0
            val uploaded = statusCounts.find { it.status == BatchStatus.ACKNOWLEDGED }?.count ?: 0
            val failed = statusCounts.find { it.status == BatchStatus.FAILED }?.count ?: 0
            val corrupted = statusCounts.find { it.status == BatchStatus.CORRUPT }?.count ?: 0
            val total = pending + uploaded + failed + corrupted
            
            QueueStatisticsDetail(
                pendingPackets = pending,
                totalSent = uploaded.toLong(),
                totalFailed = failed.toLong(),
                totalDiscarded = statusCounts.find { it.status == BatchStatus.DISCARDED }?.count?.toLong() ?: 0L,
                totalSizeMB = totalSize / (1024f * 1024f),
                totalPackets = total,
                uploadedPackets = uploaded,
                corruptedPackets = corrupted,
                totalSizeBytes = totalSize
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get queue statistics", e)
            QueueStatisticsDetail()
        }
    }
    
    /**
     * Clear all pending packets.
     */
    suspend fun clearAllPendingPackets() = withContext(Dispatchers.IO) {
        try {
            // Load pending batches
            val pendingBatches = batchMetadataDao.getPendingBatches(
                listOf(BatchStatus.PENDING, BatchStatus.FAILED)
            )
            
            // Delete files
            pendingBatches.forEach { batch ->
                val file = File(batch.filePath)
                if (file.exists()) {
                    file.delete()
                }
            }
            
            // Update database status
            pendingBatches.forEach { batch ->
                batchMetadataDao.updateStatus(batch.packetId, BatchStatus.DISCARDED)
            }
            
            // Update stats
            updateQueueStats()
            
            Log.i(TAG, "Cleared ${pendingBatches.size} pending packets")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to clear pending packets", e)
        }
    }
    
    /**
     * Export encrypted pending data.
     */
    suspend fun exportEncryptedPendingData(): String = withContext(Dispatchers.IO) {
        try {
            val pendingBatches = batchMetadataDao.getPendingBatches(
                listOf(BatchStatus.PENDING, BatchStatus.FAILED)
            )
            
            if (pendingBatches.isEmpty()) {
                return@withContext "No pending packets"
            }
            
            // Create export directory
            val exportDir = File(context.cacheDir, "export_${System.currentTimeMillis()}")
            exportDir.mkdirs()
            
            var exportedCount = 0
            pendingBatches.forEach { batch ->
                val sourceFile = File(batch.filePath)
                if (sourceFile.exists()) {
                    val destFile = File(exportDir, "${batch.packetId}.dat")
                    sourceFile.copyTo(destFile, overwrite = true)
                    exportedCount++
                }
            }
            
            Log.i(TAG, "Exported $exportedCount encrypted packets to: ${exportDir.absolutePath}")
            exportDir.absolutePath
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to export encrypted data", e)
            "Export failed: ${e.message}"
        }
    }
    
    /**
     * Cleanup resources.
     */
    fun cleanup() {
        scope.cancel()
    }
    
    /**
     * Calculate SHA-256 checksum.
     */
    private fun calculateSHA256(data: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val hashBytes = digest.digest(data)
        return hashBytes.joinToString("") { "%02x".format(it) }
    }
    
    /**
     * Enable persistence.
     */
    fun enablePersistence() {
        Log.i(TAG, "Enabling persistence")
        // TODO: Implement persistence logic
    }
    
    /**
     * Clean up old files.
     */
    suspend fun cleanupOldFiles() {
        Log.i(TAG, "Cleaning up old files")
        // TODO: Implement cleanup logic
    }
    
    /**
     * Compact database.
     */
    suspend fun compactDatabase() {
        Log.i(TAG, "Compacting database")
        // TODO: Implement database compaction
    }
    
    /**
     * Clear pending queue.
     */
    suspend fun clearPendingQueue() {
        Log.i(TAG, "Clearing pending queue")
        // TODO: Implement pending queue clearing
    }
}

/**
 * Queue statistics detail.
 */
data class QueueStatisticsDetail(
    val pendingPackets: Int = 0,
    val totalSent: Long = 0L,
    val totalFailed: Long = 0L,
    val totalDiscarded: Long = 0L,
    val totalSizeMB: Float = 0f,
    val totalPackets: Int = 0,
    val uploadedPackets: Int = 0,
    val corruptedPackets: Int = 0,
    val totalSizeBytes: Long = 0L
)

/**
 * Queue statistics summary.
 */
data class QueueStats(
    val totalSizeBytes: Long = 0, // Total size (bytes)
    val fileCount: Int = 0, // File count
    val pendingCount: Int = 0, // Pending count
    val uploadingCount: Int = 0, // Uploading count
    val failedCount: Int = 0, // Failed count
    val acknowledgedCount: Int = 0, // Acknowledged count
    val queueUsagePercent: Float = 0f, // Queue usage (%)
    val totalPackets: Int = 0, // Total packets
    val pendingPackets: Int = 0, // Pending packets
    val uploadedPackets: Int = 0, // Uploaded packets
    val corruptedPackets: Int = 0 // Corrupted packets
)

/**
 * Raised when storage space is insufficient.
 */
class InsufficientStorageException(message: String) : Exception(message)

/**
 * Raised when a queued file/packet cannot be found.
 */
class FileNotFoundException(message: String) : Exception(message)
