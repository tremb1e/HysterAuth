package com.continuousauth.compression

import android.util.Log
import net.jpountz.lz4.LZ4Factory
import net.jpountz.lz4.LZ4FastDecompressor
import net.jpountz.lz4.LZ4SafeDecompressor
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Data compression manager.
 *
 * Intended pipeline: serialize -> compress -> encrypt.
 *
 * Supports LZ4 and GZIP.
 */
@Singleton
class CompressionManager @Inject constructor() {
    
    companion object {
        private const val TAG = "CompressionManager"
        
        // Compression threshold: skip compression under this size (bytes)
        private const val MIN_COMPRESS_SIZE = 256
        
        // Default I/O buffer size
        private const val BUFFER_SIZE = 8192
        
        // LZ4 factory instance
        private val lz4Factory = LZ4Factory.fastestInstance()
    }
    
    /**
     * Compression algorithm enum.
     */
    enum class CompressionType {
        NONE,
        GZIP,
        LZ4,
        SNAPPY  // Not implemented; falls back to LZ4
    }
    
    /**
     * Compress data.
     *
     * @param data Original payload.
     * @param type Compression algorithm.
     * @return Compressed bytes, or null on failure.
     */
    fun compress(data: ByteArray, type: CompressionType = CompressionType.GZIP): ByteArray? {
        try {
            // Skip compression for small payloads
            if (data.size < MIN_COMPRESS_SIZE) {
                Log.v(TAG, "Payload size ${data.size} bytes is below threshold; skipping compression")
                return data
            }
            
            return when (type) {
                CompressionType.NONE -> data
                CompressionType.GZIP -> compressGzip(data)
                CompressionType.LZ4 -> compressLz4(data)
                CompressionType.SNAPPY -> {
                    Log.w(TAG, "Snappy compression not implemented; falling back to LZ4")
                    compressLz4(data)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to compress data", e)
            return null
        }
    }
    
    /**
     * Decompress data.
     *
     * @param data Compressed payload.
     * @param type Compression algorithm.
     * @return Decompressed bytes, or null on failure.
     */
    fun decompress(data: ByteArray, type: CompressionType = CompressionType.GZIP): ByteArray? {
        try {
            return when (type) {
                CompressionType.NONE -> data
                CompressionType.GZIP -> decompressGzip(data)
                CompressionType.LZ4 -> decompressLz4(data)
                CompressionType.SNAPPY -> {
                    Log.w(TAG, "Snappy decompression not implemented; falling back to LZ4")
                    decompressLz4(data)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to decompress data", e)
            return null
        }
    }
    
    /**
     * GZIP compression.
     */
    private fun compressGzip(data: ByteArray): ByteArray {
        val startTime = System.currentTimeMillis()
        val originalSize = data.size
        
        val outputStream = ByteArrayOutputStream()
        GZIPOutputStream(outputStream).use { gzipStream ->
            gzipStream.write(data)
            gzipStream.finish()
        }
        
        val compressedData = outputStream.toByteArray()
        val compressedSize = compressedData.size
        val compressionRatio = (1 - compressedSize.toDouble() / originalSize) * 100
        val duration = System.currentTimeMillis() - startTime
        
        Log.d(
            TAG,
            "GZIP compression complete: $originalSize -> $compressedSize bytes " +
                "(ratio: %.1f%%, duration: ${duration}ms)".format(compressionRatio)
        )
        
        return compressedData
    }
    
    /**
     * GZIP decompression.
     */
    private fun decompressGzip(data: ByteArray): ByteArray {
        val startTime = System.currentTimeMillis()
        val compressedSize = data.size
        
        val inputStream = ByteArrayInputStream(data)
        val outputStream = ByteArrayOutputStream()
        
        GZIPInputStream(inputStream).use { gzipStream ->
            val buffer = ByteArray(BUFFER_SIZE)
            var bytesRead: Int
            while (gzipStream.read(buffer).also { bytesRead = it } != -1) {
                outputStream.write(buffer, 0, bytesRead)
            }
        }
        
        val decompressedData = outputStream.toByteArray()
        val decompressedSize = decompressedData.size
        val duration = System.currentTimeMillis() - startTime
        
        Log.d(
            TAG,
            "GZIP decompression complete: $compressedSize -> $decompressedSize bytes " +
                "(duration: ${duration}ms)"
        )
        
        return decompressedData
    }
    
    /**
     * Convert compression type to a string.
     */
    fun getCompressionTypeString(type: CompressionType): String {
        return when (type) {
            CompressionType.NONE -> "none"
            CompressionType.GZIP -> "gzip"
            CompressionType.LZ4 -> "lz4"
            CompressionType.SNAPPY -> "snappy"
        }
    }
    
    /**
     * Parse compression type from a string.
     */
    fun parseCompressionType(typeString: String): CompressionType {
        return when (typeString.lowercase()) {
            "none" -> CompressionType.NONE
            "gzip" -> CompressionType.GZIP
            "lz4" -> CompressionType.LZ4
            "snappy" -> CompressionType.SNAPPY
            else -> CompressionType.GZIP // Default: GZIP
        }
    }
    
    /**
     * Calculate compression ratio.
     */
    fun calculateCompressionRatio(originalSize: Int, compressedSize: Int): Double {
        if (originalSize == 0) return 0.0
        return (1 - compressedSize.toDouble() / originalSize) * 100
    }
    
    /**
     * LZ4 compression.
     *
     * Uses LZ4 for fast, low-latency compression.
     */
    private fun compressLz4(data: ByteArray): ByteArray {
        val startTime = System.currentTimeMillis()
        val originalSize = data.size
        
        // Use high-compression mode
        val compressor = lz4Factory.highCompressor()
        val maxCompressedLength = compressor.maxCompressedLength(originalSize)
        
        // Output buffer: includes 4 bytes of original size
        val compressed = ByteArray(maxCompressedLength + 4)
        
        // Write original size (used for decompression)
        ByteBuffer.wrap(compressed, 0, 4).putInt(originalSize)
        
        // Compress
        val compressedLength = compressor.compress(
            data, 0, originalSize,
            compressed, 4, maxCompressedLength
        )
        
        // Copy exact-sized result
        val result = ByteArray(compressedLength + 4)
        System.arraycopy(compressed, 0, result, 0, compressedLength + 4)
        
        val compressionRatio = (1 - result.size.toDouble() / originalSize) * 100
        val duration = System.currentTimeMillis() - startTime
        
        Log.d(
            TAG,
            "LZ4 compression complete: $originalSize -> ${result.size} bytes " +
                "(ratio: %.1f%%, duration: ${duration}ms)".format(compressionRatio)
        )
        
        return result
    }
    
    /**
     * LZ4 decompression.
     */
    private fun decompressLz4(data: ByteArray): ByteArray {
        val startTime = System.currentTimeMillis()
        val compressedSize = data.size
        
        // Read original size
        val originalSize = ByteBuffer.wrap(data, 0, 4).int
        
        // Create decompressor
        val decompressor = lz4Factory.safeDecompressor()
        
        // Decompress
        val decompressed = ByteArray(originalSize)
        decompressor.decompress(
            data, 4, compressedSize - 4,
            decompressed, 0, originalSize
        )
        
        val duration = System.currentTimeMillis() - startTime
        
        Log.d(
            TAG,
            "LZ4 decompression complete: $compressedSize -> $originalSize bytes " +
                "(duration: ${duration}ms)"
        )
        
        return decompressed
    }
}
