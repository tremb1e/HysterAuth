package com.continuousauth.chunking

import android.util.Log
import com.continuousauth.proto.DataPacket
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.ceil

/**
 * Chunking manager.
 *
 * Note: chunking is currently not required. This class is kept for backward compatibility and the
 * default behavior is to return the original data unchanged.
 */
@Singleton
class ChunkingManager @Inject constructor() {
    
    companion object {
        private const val TAG = "ChunkingManager"
        
        // Default max packet size: 10MB
        const val DEFAULT_MAX_PACKET_SIZE = 10 * 1024 * 1024
        
        // Minimum packet size: 256KB
        const val MIN_PACKET_SIZE = 256 * 1024
        
        // Maximum packet size: 10MB
        const val MAX_PACKET_SIZE = 10 * 1024 * 1024
        
        // Chunking buffer size
        private const val CHUNK_BUFFER_SIZE = 8192
    }
    
    // Current max packet size
    private var maxPacketSize = DEFAULT_MAX_PACKET_SIZE
    
    /**
     * Set maximum packet size.
     *
     * @param size Size in bytes; must be within [MIN_PACKET_SIZE, MAX_PACKET_SIZE].
     */
    fun setMaxPacketSize(size: Int) {
        if (size in MIN_PACKET_SIZE..MAX_PACKET_SIZE) {
            maxPacketSize = size
            Log.i(TAG, "Max packet size set to: ${size / 1024}KB")
        } else {
            Log.w(TAG, "Invalid packet size: $size; keeping current: ${maxPacketSize / 1024}KB")
        }
    }
    
    /**
     * Check whether data needs chunking.
     *
     * @param data Payload bytes.
     * @return Always returns false (chunking currently disabled).
     */
    fun needsChunking(data: ByteArray): Boolean {
        return false // Chunking currently disabled
    }
    
    /**
     * Chunk data.
     *
     * @param data Original payload.
     * @return List of chunks.
     */
    fun chunkData(data: ByteArray): List<ByteArray> {
        if (!needsChunking(data)) {
            return listOf(data)
        }
        
        val chunks = mutableListOf<ByteArray>()
        val chunkCount = ceil(data.size.toDouble() / maxPacketSize).toInt()
        
        Log.d(TAG, "Payload size ${data.size} bytes; splitting into $chunkCount chunks")
        
        for (i in 0 until chunkCount) {
            val start = i * maxPacketSize
            val end = minOf(start + maxPacketSize, data.size)
            val chunk = data.sliceArray(start until end)
            chunks.add(chunk)
            
            Log.v(TAG, "Chunk ${i + 1}/$chunkCount: ${chunk.size} bytes")
        }
        
        return chunks
    }
    
    /**
     * Process chunking for a DataPacket.
     *
     * @return Chunked packets list (currently returns the original packet only).
     */
    fun processPacketChunking(originalPacket: DataPacket): List<DataPacket> {
        // Chunking disabled; return original packet.
        return listOf(originalPacket)
    }
    
    /**
     * Reassemble chunked packets.
     *
     * @return Reassembled packet, or null on failure.
     */
    fun reassembleChunks(chunks: List<DataPacket>): DataPacket? {
        if (chunks.isEmpty()) {
            Log.e(TAG, "Chunk list is empty; cannot reassemble")
            return null
        }
        
        // Chunking disabled; return first packet.
        return chunks.first()
    }
    
    /**
     * Get chunk info description.
     */
    fun getChunkInfo(packet: DataPacket): String {
        return "full packet" // Chunking disabled
    }
    
    /**
     * Calculate how many chunks are needed for a given size.
     */
    fun calculateChunkCount(dataSize: Int): Int {
        return if (dataSize <= maxPacketSize) {
            1
        } else {
            ceil(dataSize.toDouble() / maxPacketSize).toInt()
        }
    }
}
