package com.continuousauth.buffer

import com.continuousauth.proto.DataPacket
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import javax.inject.Inject
import javax.inject.Singleton

/**
 * In-memory buffer implementation.
 *
 * Uses a Channel to queue encrypted sensor packets and supports an optional ring buffer for fast mode.
 */
@Singleton
class InMemoryBufferImpl @Inject constructor() : InMemoryBuffer {
    
    companion object {
        private const val DEFAULT_MAX_CAPACITY = 10000 // Default max capacity
        private const val OVERFLOW_THRESHOLD = 0.9 // Overflow threshold (90%)
        private const val ESTIMATED_PACKET_SIZE = 1024 // Estimated packet size (bytes)
        private const val TAG = "InMemoryBuffer"
        private const val RING_BUFFER_CAPACITY = 2 * 1024 * 1024 // 2MB ring buffer
    }
    
    // Channel-based thread-safe queue
    private val dataChannel = Channel<DataPacket>(capacity = DEFAULT_MAX_CAPACITY)
    
    // Optional ring buffer for fast mode
    private val ringBuffer = java.nio.ByteBuffer.allocateDirect(RING_BUFFER_CAPACITY)
    private var ringBufferWritePos = 0
    private var ringBufferReadPos = 0
    private var ringBufferSize = 0
    private val ringBufferLock = kotlinx.coroutines.sync.Mutex()
    
    // Transmission mode flag
    private var isFastMode = false
    
    // Cache for batch reads
    private val batchCache = mutableListOf<DataPacket>()
    private val batchCacheMutex = Mutex()
    
    // Stats
    private val currentSize = AtomicInteger(0)
    private val totalEnqueued = AtomicLong(0L)
    private val totalDequeued = AtomicLong(0L)
    
    // Configuration
    private val maxCapacity = DEFAULT_MAX_CAPACITY
    
    override suspend fun enqueue(dataPacket: DataPacket): Boolean {
        try {
            // Capacity check
            if (currentSize.get() >= maxCapacity) {
                android.util.Log.w(TAG, "Buffer full; dropping packet - current size: ${currentSize.get()}")
                return false
            }
            
            // Non-blocking enqueue
            val result = dataChannel.trySend(dataPacket)
            
            if (result.isSuccess) {
                currentSize.incrementAndGet()
                totalEnqueued.incrementAndGet()
                android.util.Log.v(TAG, "Packet enqueued - current size: ${currentSize.get()}")
                return true
            } else {
                android.util.Log.w(TAG, "Channel full; cannot enqueue packet")
                return false
            }
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to enqueue packet", e)
            return false
        }
    }
    
    override suspend fun dequeue(maxCount: Int): List<DataPacket> = batchCacheMutex.withLock {
        val result = mutableListOf<DataPacket>()
        
        try {
            val actualMaxCount = if (maxCount <= 0) currentSize.get() else minOf(maxCount, currentSize.get())
            
            // Dequeue packets in a batch
            var count = 0
            while (count < actualMaxCount) {
                val packet = dataChannel.tryReceive()
                if (packet.isSuccess) {
                    result.add(packet.getOrThrow())
                    currentSize.decrementAndGet()
                    totalDequeued.incrementAndGet()
                    count++
                } else {
                    // Channel is empty
                    break
                }
            }
            
            if (result.isNotEmpty()) {
                android.util.Log.v(TAG, "Packets dequeued - count: ${result.size}, remaining: ${currentSize.get()}")
            }
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to dequeue packets", e)
        }
        
        return result
    }
    
    override fun getSize(): Int {
        return currentSize.get()
    }
    
    override fun isEmpty(): Boolean {
        return currentSize.get() == 0
    }
    
    override fun getDataFlow(): Flow<DataPacket> {
        return dataChannel.receiveAsFlow()
    }
    
    override suspend fun clear() {
        try {
            // Drain the channel
            while (!dataChannel.isEmpty) {
                dataChannel.tryReceive()
            }
            currentSize.set(0)
            android.util.Log.i(TAG, "Buffer cleared")
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to clear buffer", e)
        }
    }
    
    override fun getBufferStatus(): BufferStatus {
        val size = currentSize.get()
        val memoryUsage = size * ESTIMATED_PACKET_SIZE.toLong()
        val isOverflow = size.toDouble() / maxCapacity > OVERFLOW_THRESHOLD
        
        return BufferStatus(
            currentSize = size,
            maxCapacity = maxCapacity,
            totalEnqueued = totalEnqueued.get(),
            totalDequeued = totalDequeued.get(),
            memoryUsageBytes = memoryUsage,
            isOverflow = isOverflow
        )
    }
    
    /**
     * Set transmission mode.
     *
     * When fast mode is enabled, a ring buffer can be used for optimization.
     */
    fun setTransmissionMode(fastMode: Boolean) {
        isFastMode = fastMode
        if (fastMode) {
            android.util.Log.i(TAG, "Switched to FAST_MODE - using Ring Buffer optimization")
        } else {
            android.util.Log.i(TAG, "Switched to SLOW_MODE - using standard Channel")
        }
    }
    
    /**
     * Ring buffer write (fast mode).
     */
    private suspend fun writeToRingBuffer(data: ByteArray): Boolean = ringBufferLock.withLock {
        if (data.size > RING_BUFFER_CAPACITY) {
            android.util.Log.e(TAG, "Data size exceeds ring buffer capacity")
            return false
        }
        
        // Check available space
        val availableSpace = RING_BUFFER_CAPACITY - ringBufferSize
        if (data.size > availableSpace) {
            // Buffer full: discard oldest data
            val toDiscard = data.size - availableSpace
            ringBufferReadPos = (ringBufferReadPos + toDiscard) % RING_BUFFER_CAPACITY
            ringBufferSize -= toDiscard
            android.util.Log.w(TAG, "Ring buffer overflow; discarded $toDiscard bytes")
        }
        
        // Write data
        val endPos = (ringBufferWritePos + data.size) % RING_BUFFER_CAPACITY
        if (endPos > ringBufferWritePos) {
            // Contiguous write
            ringBuffer.position(ringBufferWritePos)
            ringBuffer.put(data)
        } else {
            // Split write (wrap-around)
            val firstPartSize = RING_BUFFER_CAPACITY - ringBufferWritePos
            ringBuffer.position(ringBufferWritePos)
            ringBuffer.put(data, 0, firstPartSize)
            
            ringBuffer.position(0)
            ringBuffer.put(data, firstPartSize, data.size - firstPartSize)
        }
        
        ringBufferWritePos = endPos
        ringBufferSize += data.size
        
        return true
    }
    
    /**
     * Ring buffer read (fast mode).
     */
    private suspend fun readFromRingBuffer(maxBytes: Int): ByteArray? = ringBufferLock.withLock {
        if (ringBufferSize == 0) {
            return null
        }
        
        val bytesToRead = minOf(maxBytes, ringBufferSize)
        val data = ByteArray(bytesToRead)
        
        val endPos = (ringBufferReadPos + bytesToRead) % RING_BUFFER_CAPACITY
        if (endPos > ringBufferReadPos) {
            // Contiguous read
            ringBuffer.position(ringBufferReadPos)
            ringBuffer.get(data)
        } else {
            // Split read (wrap-around)
            val firstPartSize = RING_BUFFER_CAPACITY - ringBufferReadPos
            ringBuffer.position(ringBufferReadPos)
            ringBuffer.get(data, 0, firstPartSize)
            
            ringBuffer.position(0)
            ringBuffer.get(data, firstPartSize, bytesToRead - firstPartSize)
        }
        
        ringBufferReadPos = endPos
        ringBufferSize -= bytesToRead
        
        return data
    }
    
    /**
     * Adjust buffer size.
     */
    override fun adjustBufferSize(multiplier: Float) {
        android.util.Log.i(TAG, "Adjust buffer size multiplier: $multiplier")
        // TODO: Implement buffer size adjustment logic
    }
}
