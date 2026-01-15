package com.continuousauth.network

import com.continuousauth.proto.DataPacket
import com.continuousauth.proto.ServerDirective
import kotlinx.coroutines.flow.Flow

/**
 * gRPC bidirectional-stream uploader interface.
 *
 * Responsible for bidirectional streaming communication with the server.
 */
interface Uploader {
    
    /**
     * Starts a bidirectional connection.
     *
     * @param serverEndpoint Server endpoint.
     * @return True if connection is established.
     */
    suspend fun connect(serverEndpoint: String): Boolean
    
    /**
     * Sends a data packet to the server.
     *
     * @return True if the packet was sent.
     */
    suspend fun sendDataPacket(dataPacket: DataPacket): Boolean
    
    /**
     * Returns the server directive flow.
     */
    fun getServerDirectiveFlow(): Flow<ServerDirective>
    
    /**
     * Disconnects.
     */
    suspend fun disconnect()
    
    /**
     * Returns true if connected.
     */
    fun isConnected(): Boolean
    
    /**
     * Returns connection stats.
     */
    fun getConnectionStats(): ConnectionStats
    
    /**
     * Returns transmission stats.
     */
    fun getTransmissionStats(): TransmissionStats
    
    /**
     * Returns gRPC status.
     */
    fun getGrpcStatus(): GrpcStatus
    
    /**
     * Returns buffer stats.
     */
    fun getBufferStats(): BufferStats
    
    /**
     * Returns server endpoint.
     */
    fun getServerEndpoint(): String
    
    /**
     * Returns connection status detail.
     */
    fun getConnectionStatus(): ConnectionStatusDetail
    
    /**
     * Returns transmission statistics detail.
     */
    fun getStatistics(): Statistics
    
    /**
     * Returns memory buffer stats.
     */
    fun getMemoryBufferStats(): MemoryBufferStats
    
    /**
     * Returns recent latencies.
     */
    fun getRecentLatencies(): List<Long>
    
    /**
     * Returns server policy.
     */
    suspend fun getServerPolicy(): ServerPolicy
}

/**
 * Connection status detail.
 */
data class ConnectionStatusDetail(
    val state: String,
    val endpoint: String,
    val lastAckLatencyMs: Long
)

/**
 * Statistics.
 */
data class Statistics(
    val totalPacketsSent: Long,
    val totalPacketsAcknowledged: Long,
    val totalFailures: Long
)

/**
 * Memory buffer stats.
 */
data class MemoryBufferStats(
    val samplesInMemory: Int,
    val totalSent: Long,
    val totalFailed: Long,
    val totalDiscarded: Long
)

/**
 * Server policy.
 */
data class ServerPolicy(
    val version: String = "1.0",
    val lastUpdated: Long = System.currentTimeMillis(),
    val fastModeDurationSeconds: Int = 30,
    val anomalyThreshold: Float = 0.8f,
    val samplingRates: Map<String, Float> = mapOf(
        "ACCELEROMETER" to 100f,
        "GYROSCOPE" to 100f,
        "MAGNETOMETER" to 50f
    ),
    val transmissionStrategy: String = "ADAPTIVE"
) {
    fun toJson(): String {
        return """
            {
                "version": "$version",
                "lastUpdated": $lastUpdated,
                "fastModeDurationSeconds": $fastModeDurationSeconds,
                "anomalyThreshold": $anomalyThreshold,
                "samplingRates": {
                    ${samplingRates.entries.joinToString(",\n                    ") { 
                        "\"${it.key}\": ${it.value}"
                    }}
                },
                "transmissionStrategy": "$transmissionStrategy"
            }
        """.trimIndent()
    }
}

/**
 * Connection status.
 */
enum class ConnectionStatus {
    DISCONNECTED,       // Disconnected
    CONNECTING,         // Connecting
    CONNECTED,          // Connected
    RECONNECTING,       // Reconnecting
    ERROR               // Error
}

/**
 * Connection statistics.
 */
data class ConnectionStats(
    val currentStatus: ConnectionStatus,
    val connectedSince: Long? = null,           // Connection established timestamp
    val totalPacketsSent: Long = 0L,            // Total packets sent
    val totalAcksReceived: Long = 0L,           // Total ACKs received
    val totalPolicyUpdates: Long = 0L,          // Total policy updates received
    val lastAckLatency: Long? = null,           // Last ACK latency (ms)
    val averageAckLatency: Double = 0.0,        // Average ACK latency (ms)
    val connectionErrors: Int = 0,              // Connection error count
    val lastErrorTimestamp: Long? = null,       // Last error timestamp
    val lastErrorMessage: String? = null        // Last error message
)

/**
 * Transmission statistics.
 */
data class TransmissionStats(
    val isFastMode: Boolean = false,            // Whether fast mode is active
    val fastModeRemainingSeconds: Int = 0,      // Remaining fast mode seconds
    val lastTriggerType: String? = null,        // Last trigger type
    val successCount: Long = 0L,                // Successful uploads
    val failedCount: Long = 0L,                 // Failed uploads
    val averageLatency: Long = 0L               // Average latency (ms)
)

/**
 * gRPC status.
 */
data class GrpcStatus(
    val connectionState: ConnectionStatus = ConnectionStatus.DISCONNECTED,
    val lastAckLatency: Long = 0L               // Last ACK latency (ms)
)

/**
 * Buffer statistics.
 */
data class BufferStats(
    val memorySamples: Int = 0,                 // Samples in memory
    val diskBatches: Int = 0,                   // Batches on disk
    val sentCount: Long = 0L,                   // Sent count
    val discardedCount: Long = 0L               // Discarded count
)
