package com.continuousauth.database

import androidx.room.*
import kotlinx.coroutines.flow.Flow

/**
 * DAO for batch metadata.
 *
 * Provides database access methods for batch bookkeeping.
 */
@Dao
interface BatchMetadataDao {
    
    /**
     * Insert a new batch metadata entry.
     */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(batch: BatchMetadata)
    
    /**
     * Insert multiple batch metadata entries.
     */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(vararg batches: BatchMetadata)
    
    /**
     * Update a batch metadata entry.
     */
    @Update
    suspend fun update(batch: BatchMetadata)
    
    /**
     * Update batch status.
     */
    @Query("UPDATE batch_metadata SET status = :status WHERE packetId = :packetId")
    suspend fun updateStatus(packetId: String, status: BatchStatus)
    
    /**
     * Update batch status and upload time.
     */
    @Query("UPDATE batch_metadata SET status = :status, uploadTime = :uploadTime WHERE packetId = :packetId")
    suspend fun updateUploadStatus(packetId: String, status: BatchStatus, uploadTime: Long)
    
    /**
     * Update ACK status.
     */
    @Query("UPDATE batch_metadata SET status = :status, ackTime = :ackTime WHERE packetId = :packetId")
    suspend fun updateAckStatus(packetId: String, status: BatchStatus = BatchStatus.ACKNOWLEDGED, ackTime: Long)
    
    /**
     * Update retry info.
     */
    @Query("UPDATE batch_metadata SET retryCount = retryCount + 1, lastError = :error WHERE packetId = :packetId")
    suspend fun updateRetryInfo(packetId: String, error: String)
    
    /**
     * Delete a batch metadata entry.
     */
    @Delete
    suspend fun delete(batch: BatchMetadata)
    
    /**
     * Delete a batch by packet id.
     */
    @Query("DELETE FROM batch_metadata WHERE packetId = :packetId")
    suspend fun deleteById(packetId: String)
    
    /**
     * Delete batches by status.
     */
    @Query("DELETE FROM batch_metadata WHERE status = :status")
    suspend fun deleteByStatus(status: BatchStatus = BatchStatus.ACKNOWLEDGED)
    
    /**
     * Delete the oldest N batches (capacity management).
     */
    @Query("DELETE FROM batch_metadata WHERE packetId IN (SELECT packetId FROM batch_metadata ORDER BY createdTime ASC LIMIT :count)")
    suspend fun deleteOldest(count: Int)
    
    /**
     * Get all pending batches (e.g. PENDING or FAILED).
     */
    @Query("SELECT * FROM batch_metadata WHERE status IN (:statuses) ORDER BY createdTime ASC")
    suspend fun getPendingBatches(statuses: List<BatchStatus> = listOf(BatchStatus.PENDING, BatchStatus.FAILED)): List<BatchMetadata>
    
    /**
     * Get a single batch metadata entry.
     */
    @Query("SELECT * FROM batch_metadata WHERE packetId = :packetId")
    suspend fun getById(packetId: String): BatchMetadata?
    
    /**
     * Get all batches (newest first).
     */
    @Query("SELECT * FROM batch_metadata ORDER BY createdTime DESC")
    fun getAllBatches(): Flow<List<BatchMetadata>>
    
    /**
     * Get total batch count.
     */
    @Query("SELECT COUNT(*) FROM batch_metadata")
    suspend fun getTotalCount(): Int
    
    /**
     * Get batch counts by status.
     */
    @Query("SELECT status, COUNT(*) as count FROM batch_metadata GROUP BY status")
    suspend fun getStatusCounts(): List<StatusCount>
    
    /**
     * Get total file size (excluding a status).
     */
    @Query("SELECT SUM(fileSize) FROM batch_metadata WHERE status != :excludeStatus")
    suspend fun getTotalFileSize(excludeStatus: BatchStatus = BatchStatus.ACKNOWLEDGED): Long?
    
    /**
     * Get the oldest batch.
     */
    @Query("SELECT * FROM batch_metadata ORDER BY createdTime ASC LIMIT 1")
    suspend fun getOldestBatch(): BatchMetadata?
    
    /**
     * Clear the table.
     */
    @Query("DELETE FROM batch_metadata")
    suspend fun clearAll()
}

/**
 * Status count projection.
 */
data class StatusCount(
    val status: BatchStatus,
    val count: Int
)
