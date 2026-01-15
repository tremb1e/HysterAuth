package com.continuousauth.database

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Batch metadata entity.
 *
 * Stores per-packet state and bookkeeping information.
 */
@Entity(tableName = "batch_metadata")
data class BatchMetadata(
    @PrimaryKey
    val packetId: String,                  // Packet identifier (UUID)
    val filePath: String,                  // Encrypted payload file path
    val status: BatchStatus,               // Batch status
    val createdTime: Long,                 // Creation timestamp (ms)
    val uploadTime: Long? = null,          // Upload timestamp (ms)
    val ackTime: Long? = null,             // ACK timestamp (ms)
    val fileSize: Long,                    // File size (bytes)
    val sampleCount: Int,                  // Sample count
    val transmissionMode: String,          // Transmission mode (SLOW/FAST)
    val ntpOffset: Long? = null,           // NTP offset (ms)
    val baseWallMs: Long,                  // Base UTC time (NTP-adjusted)
    val deviceUptimeNs: Long,              // Device uptime (ns)
    val retryCount: Int = 0,               // Retry count
    val lastError: String? = null,         // Last error message
    val sequenceNumber: Long? = null,      // Sequence number (ordering)
    val userId: String? = null,            // User identifier
    val sessionId: String? = null,         // Session identifier
    val deviceId: String,                  // Device identifier
    val sha256: String? = null             // SHA-256 checksum
)

/**
 * Batch status enum.
 */
enum class BatchStatus {
    PENDING,        // Pending upload
    UPLOADING,      // Uploading
    UPLOADED,       // Uploaded (awaiting ACK)
    ACKNOWLEDGED,   // Acknowledged (ACK received)
    FAILED,         // Upload failed
    DISCARDED,      // Discarded (capacity exceeded)
    CORRUPT         // Corrupt (SHA-256 verification failed)
}
