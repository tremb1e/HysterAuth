package com.continuousauth.buffer

import com.continuousauth.proto.DataPacket
import kotlinx.coroutines.flow.Flow

/**
 * In-memory buffer interface.
 *
 * Thread-safe queue for encrypted sensor data packets.
 */
interface InMemoryBuffer {
    
    /**
     * Enqueue an encrypted data packet.
     */
    suspend fun enqueue(dataPacket: DataPacket): Boolean
    
    /**
     * Dequeue a batch of packets.
     *
     * @param maxCount Maximum count; -1 means dequeue all.
     */
    suspend fun dequeue(maxCount: Int = -1): List<DataPacket>
    
    /**
     * Get current buffer size.
     */
    fun getSize(): Int
    
    /**
     * Check whether buffer is empty.
     */
    fun isEmpty(): Boolean
    
    /**
     * Stream packets as they arrive.
     */
    fun getDataFlow(): Flow<DataPacket>
    
    /**
     * Clear the buffer.
     */
    suspend fun clear()
    
    /**
     * Get buffer status.
     */
    fun getBufferStatus(): BufferStatus
    
    /**
     * Adjust buffer size.
     */
    fun adjustBufferSize(multiplier: Float)
}

/**
 * Buffer status.
 */
data class BufferStatus(
    val currentSize: Int,           // Current packet count
    val maxCapacity: Int,           // Max capacity
    val totalEnqueued: Long,        // Total enqueued
    val totalDequeued: Long,        // Total dequeued
    val memoryUsageBytes: Long,     // Estimated memory usage (bytes)
    val isOverflow: Boolean         // Near overflow
)
