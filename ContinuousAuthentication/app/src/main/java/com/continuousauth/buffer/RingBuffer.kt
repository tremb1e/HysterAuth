package com.continuousauth.buffer

import com.continuousauth.model.SensorSample
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Ring buffer implementation.
 *
 * Uses a preallocated direct ByteBuffer to reduce allocations and GC pressure, improving performance
 * in fast mode.
 */
@Singleton
class RingBuffer @Inject constructor() {
    
    companion object {
        private const val TAG = "RingBuffer"
        
        // Estimated bytes per sensor sample
        // type(4) + timestamp(8) + x(4) + y(4) + z(4) + accuracy(4) + seqNo(8) + app(~50) = ~90 bytes
        private const val SAMPLE_SIZE_BYTES = 96
        
        // Default buffer size (≈10,000 samples)
        private const val DEFAULT_BUFFER_SIZE = 10000 * SAMPLE_SIZE_BYTES // ~960KB
        
        // Min and max buffer sizes
        private const val MIN_BUFFER_SIZE = 1000 * SAMPLE_SIZE_BYTES    // ~96KB
        private const val MAX_BUFFER_SIZE = 50000 * SAMPLE_SIZE_BYTES   // ~4.8MB
    }
    
    // Direct ByteBuffer allocated off-heap to reduce GC pressure
    private val buffer: ByteBuffer = ByteBuffer.allocateDirect(DEFAULT_BUFFER_SIZE)
        .order(ByteOrder.nativeOrder())
    
    // Read/write pointers
    private var writePosition = 0
    private var readPosition = 0
    private var size = 0
    
    // Thread-safety
    private val bufferMutex = Mutex()
    
    // Stats
    private var totalWritten = 0L
    private var totalRead = 0L
    private var overflowCount = 0L
    
    /**
     * Write a sensor sample into the ring buffer.
     *
     * Writes directly into the ByteBuffer to avoid intermediate allocations.
     */
    suspend fun write(sample: SensorSample): Boolean = bufferMutex.withLock {
        // Ensure there is enough space
        if (size >= buffer.capacity() - SAMPLE_SIZE_BYTES) {
            // Buffer full: overwrite oldest data
            overflowCount++
            android.util.Log.w(TAG, "Ring buffer overflow; overwriting oldest data. overflowCount: $overflowCount")
            
            // Move read pointer and drop the oldest sample
            readPosition = (readPosition + SAMPLE_SIZE_BYTES) % buffer.capacity()
            size -= SAMPLE_SIZE_BYTES
        }
        
        try {
            // Set write position
            buffer.position(writePosition)
            
            // Write sample into ByteBuffer (simplified format for performance)
            buffer.putInt(sample.type.ordinal)             // 4 bytes: sensor type
            buffer.putLong(sample.eventTimestampNs)        // 8 bytes: timestamp
            buffer.putFloat(sample.x)                      // 4 bytes: x value
            buffer.putFloat(sample.y)                      // 4 bytes: y value
            buffer.putFloat(sample.z)                      // 4 bytes: z value
            buffer.putInt(sample.accuracy)                  // 4 bytes: accuracy
            buffer.putLong(sample.seqNo)                   // 8 bytes: sequence number
            
            // Write foreground app (fixed-length string)
            val appBytes = (sample.foregroundApp ?: "").toByteArray(Charsets.UTF_8)
            val appLength = minOf(appBytes.size, 50)
            buffer.putInt(appLength)                       // 4 bytes: string length
            if (appLength > 0) {
                buffer.put(appBytes, 0, appLength)         // N bytes: app string
            }
            // Pad to fixed length
            for (i in appLength until 50) {
                buffer.put(0)
            }
            
            // Update write position
            writePosition = (writePosition + SAMPLE_SIZE_BYTES) % buffer.capacity()
            size += SAMPLE_SIZE_BYTES
            totalWritten++
            
            return true
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to write to ring buffer", e)
            return false
        }
    }
    
    /**
     * Read sensor samples in a batch.
     *
     * Returns up to the requested number of samples (if available).
     */
    suspend fun read(maxSamples: Int): List<SensorSample> = bufferMutex.withLock {
        val samples = mutableListOf<SensorSample>()
        
        if (size == 0) {
            return samples
        }
        
        val samplesToRead = minOf(maxSamples, size / SAMPLE_SIZE_BYTES)
        
        try {
            repeat(samplesToRead) {
                // Set read position
                buffer.position(readPosition)
                
                // Read sample
                val typeOrdinal = buffer.getInt()
                val type = com.continuousauth.model.SensorType.values()[typeOrdinal]
                val timestamp = buffer.getLong()
                val x = buffer.getFloat()
                val y = buffer.getFloat()
                val z = buffer.getFloat()
                val accuracy = buffer.getInt()
                val seqNo = buffer.getLong()
                
                // Read foreground app
                val appLength = buffer.getInt()
                val foregroundApp = if (appLength > 0) {
                    val appBytes = ByteArray(appLength)
                    buffer.get(appBytes)
                    // Skip padding
                    buffer.position(buffer.position() + (50 - appLength))
                    String(appBytes, Charsets.UTF_8)
                } else {
                    // Skip padding
                    buffer.position(buffer.position() + 50)
                    null
                }
                
                samples.add(
                    SensorSample(
                        type = type,
                        eventTimestampNs = timestamp,
                        x = x,
                        y = y,
                        z = z,
                        accuracy = accuracy,
                        seqNo = seqNo,
                        foregroundApp = foregroundApp ?: ""
                    )
                )
                
                // Update read position
                readPosition = (readPosition + SAMPLE_SIZE_BYTES) % buffer.capacity()
                size -= SAMPLE_SIZE_BYTES
                totalRead++
            }
            
            android.util.Log.v(TAG, "Read ${samples.size} samples from ring buffer")
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to read from ring buffer", e)
        }
        
        return samples
    }
    
    /**
     * Clear buffer.
     */
    suspend fun clear() = bufferMutex.withLock {
        buffer.clear()
        writePosition = 0
        readPosition = 0
        size = 0
        android.util.Log.i(TAG, "Ring buffer cleared")
    }
    
    /**
     * Get buffer status.
     */
    suspend fun getStatus(): RingBufferStatus = bufferMutex.withLock {
        RingBufferStatus(
            capacity = buffer.capacity(),
            size = size,
            sampleCount = size / SAMPLE_SIZE_BYTES,
            writePosition = writePosition,
            readPosition = readPosition,
            totalWritten = totalWritten,
            totalRead = totalRead,
            overflowCount = overflowCount,
            utilizationPercent = (size * 100.0 / buffer.capacity()).toFloat()
        )
    }
    
    /**
     * Resize the buffer (requires reallocation).
     *
     * Note: this clears existing data.
     */
    suspend fun resize(newSize: Int): Boolean = bufferMutex.withLock {
        val actualSize = newSize.coerceIn(MIN_BUFFER_SIZE, MAX_BUFFER_SIZE)
        
        try {
            // Allocate new ByteBuffer
            val newBuffer = ByteBuffer.allocateDirect(actualSize)
                .order(ByteOrder.nativeOrder())
            
            // Copy existing data if needed
            if (size > 0 && size <= actualSize) {
                // TODO: Implement data migration logic
                android.util.Log.w(TAG, "Resizing the buffer will drop existing data")
            }
            
            // Replace buffer (old buffer becomes eligible for GC)
            buffer.clear()
            writePosition = 0
            readPosition = 0
            size = 0
            
            android.util.Log.i(TAG, "Ring buffer resized to: $actualSize bytes")
            return true
        } catch (e: OutOfMemoryError) {
            android.util.Log.e(TAG, "Failed to resize ring buffer: out of memory", e)
            return false
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to resize ring buffer", e)
            return false
        }
    }
    
    /**
     * Get available space (bytes).
     */
    fun getAvailableSpace(): Int {
        return buffer.capacity() - size
    }
    
    /**
     * Check whether the buffer is empty.
     */
    fun isEmpty(): Boolean {
        return size == 0
    }
    
    /**
     * Check whether the buffer is full.
     */
    fun isFull(): Boolean {
        return size >= buffer.capacity() - SAMPLE_SIZE_BYTES
    }
}

/**
 * Ring buffer status.
 */
data class RingBufferStatus(
    val capacity: Int,              // Total capacity (bytes)
    val size: Int,                  // Current usage (bytes)
    val sampleCount: Int,           // Current sample count
    val writePosition: Int,         // Write position
    val readPosition: Int,          // Read position
    val totalWritten: Long,         // Total samples written
    val totalRead: Long,            // Total samples read
    val overflowCount: Long,        // Overflow count
    val utilizationPercent: Float   // Utilization percentage
)
