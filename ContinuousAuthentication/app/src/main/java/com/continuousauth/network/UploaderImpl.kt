package com.continuousauth.network

import android.util.Log
import com.continuousauth.proto.*
import com.continuousauth.policy.PolicyManager
import io.grpc.*
import io.grpc.okhttp.OkHttpChannelBuilder
import io.grpc.stub.StreamObserver
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Packet awaiting ACK.
 */
data class PendingAck(
    val packetId: String,
    val sentTimestamp: Long,
    val dataPacket: DataPacket
)

/**
 * gRPC bidirectional-stream uploader implementation.
 *
 * Uses a bidirectional streaming RPC for data upload and receiving server directives.
 */
@Singleton
class UploaderImpl @Inject constructor(
    private val tlsSecurityManager: TlsSecurityManager,
    private val policyManager: PolicyManager
) : Uploader {

    companion object {
        private const val TAG = "Uploader"
        private const val CONNECTION_TIMEOUT_SECONDS = 30L
        private const val KEEPALIVE_TIME_SECONDS = 30L
        private const val KEEPALIVE_TIMEOUT_SECONDS = 5L
        private const val MAX_RETRY_ATTEMPTS = 3
        private const val ACK_TIMEOUT_MS = 10000L // ACK timeout (10s)
    }

    // gRPC.
    private var channel: ManagedChannel? = null
    private var stub: SensorDataServiceGrpc.SensorDataServiceStub? = null
    private var requestObserver: StreamObserver<DataPacket>? = null

    // State.
    private val connectionStatus = AtomicReference(ConnectionStatus.DISCONNECTED)
    private var connectedSince: Long? = null

    // Stream for directives.
    private val serverDirectiveChannel = Channel<ServerDirective>(Channel.UNLIMITED)

    // Coroutine scope.
    private val uploaderScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    // Statistics.
    private val totalPacketsSent = AtomicLong(0L)
    private val totalAcksReceived = AtomicLong(0L)
    private val totalPolicyUpdates = AtomicLong(0L)
    private var connectionErrors = 0
    private var lastErrorTimestamp: Long? = null
    private var lastErrorMessage: String? = null

    // ACK tracking.
    private val pendingAcks = ConcurrentHashMap<String, PendingAck>()
    private var lastAckLatency: Long? = null
    private val ackLatencies = mutableListOf<Long>()

    // Reconnect logic.
    private var reconnectJob: Job? = null
    private var currentEndpoint: String = ""

    override suspend fun connect(serverEndpoint: String): Boolean {
        if (connectionStatus.get() == ConnectionStatus.CONNECTED) {
            Log.w(TAG, "Already connected to server")
            return true
        }

        currentEndpoint = serverEndpoint
        connectionStatus.set(ConnectionStatus.CONNECTING)

        return try {
            // Parse endpoint.
            val parts = serverEndpoint.split(":")
            val host = parts[0]
            val port = if (parts.size > 1) parts[1].toInt() else 443

            // Create gRPC channel.
            val baseBuilder = OkHttpChannelBuilder
                .forAddress(host, port)
                .keepAliveTime(KEEPALIVE_TIME_SECONDS, TimeUnit.SECONDS)
                .keepAliveTimeout(KEEPALIVE_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                .keepAliveWithoutCalls(true)
                .maxInboundMessageSize(4 * 1024 * 1024) // 4MB

            // Certificate pinning from policy.
            val policyConfig = policyManager.getCurrentPolicyConfiguration()
            val pinnedCertificates = policyConfig.securityConfig.pinnedCertificates

            // Apply TLS configuration and SPKI pinning.
            channel = tlsSecurityManager.configureTlsForChannelBuilder(
                baseBuilder,
                pinnedCertificates,
                host
            ).build()
            stub = SensorDataServiceGrpc.newStub(channel)

            // Establish bidirectional stream.
            if (establishBidirectionalStream()) {
                connectionStatus.set(ConnectionStatus.CONNECTED)
                connectedSince = System.currentTimeMillis()

                Log.i(TAG, "Connected to server: $serverEndpoint")
                true
            } else {
                handleConnectionError("Failed to establish bidirectional stream")
                false
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to connect to server: $serverEndpoint", e)
            handleConnectionError("Connection error: ${e.message}")
            false
        }
    }

    override suspend fun sendDataPacket(dataPacket: DataPacket): Boolean {
        val observer = requestObserver
        if (observer == null || connectionStatus.get() != ConnectionStatus.CONNECTED) {
            Log.w(TAG, "Not connected; cannot send data packet")
            return false
        }

        return try {
            // Track pending ACK.
            val pendingAck = PendingAck(
                packetId = dataPacket.packetId,
                sentTimestamp = System.currentTimeMillis(),
                dataPacket = dataPacket
            )
            pendingAcks[dataPacket.packetId] = pendingAck

            // Send packet.
            observer.onNext(dataPacket)
            totalPacketsSent.incrementAndGet()

            Log.d(TAG, "Packet sent: ${dataPacket.packetId}")

            // ACK timeout watchdog.
            uploaderScope.launch {
                delay(ACK_TIMEOUT_MS)
                checkAckTimeout(dataPacket.packetId)
            }

            true

        } catch (e: Exception) {
            Log.e(TAG, "Failed to send packet: ${dataPacket.packetId}", e)
            pendingAcks.remove(dataPacket.packetId)
            false
        }
    }

    override fun getServerDirectiveFlow(): Flow<ServerDirective> {
        return serverDirectiveChannel.receiveAsFlow()
    }

    override suspend fun disconnect() {
        connectionStatus.set(ConnectionStatus.DISCONNECTED)
        connectedSince = null

        // Cancel reconnect job.
        reconnectJob?.cancel()

        // Close request stream.
        try {
            requestObserver?.onCompleted()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing request stream", e)
        }

        // Close gRPC channel.
        channel?.let { ch ->
            try {
                ch.shutdown().awaitTermination(5, TimeUnit.SECONDS)
                if (!ch.isTerminated) {
                    ch.shutdownNow()
                }
            } catch (e: Exception) {
                Log.w(TAG, "Error closing gRPC channel", e)
            }
            Unit // Explicit return value for the lambda.
        }

        // Clear state.
        requestObserver = null
        stub = null
        channel = null
        pendingAcks.clear()

        Log.i(TAG, "Disconnected from server")
    }

    override fun isConnected(): Boolean {
        return connectionStatus.get() == ConnectionStatus.CONNECTED
    }

    override fun getConnectionStatus(): ConnectionStatusDetail {
        return ConnectionStatusDetail(
            state = connectionStatus.get().name,
            endpoint = currentEndpoint,
            lastAckLatencyMs = lastAckLatency ?: 0L
        )
    }

    override fun getConnectionStats(): ConnectionStats {
        return ConnectionStats(
            currentStatus = connectionStatus.get(),
            connectedSince = connectedSince,
            totalPacketsSent = totalPacketsSent.get(),
            totalAcksReceived = totalAcksReceived.get(),
            totalPolicyUpdates = totalPolicyUpdates.get(),
            lastAckLatency = lastAckLatency,
            averageAckLatency = if (ackLatencies.isEmpty()) 0.0 else ackLatencies.average(),
            connectionErrors = connectionErrors,
            lastErrorTimestamp = lastErrorTimestamp,
            lastErrorMessage = lastErrorMessage
        )
    }

    /**
     * Establishes bidirectional streaming.
     */
    private fun establishBidirectionalStream(): Boolean {
        return try {
            val responseObserver = object : StreamObserver<ServerDirective> {
                override fun onNext(directive: ServerDirective) {
                    handleServerDirective(directive)
                }

                override fun onError(t: Throwable) {
                    Log.e(TAG, "Server response stream error", t)
                    handleConnectionError("Server response stream error: ${t.message}")
                    // Trigger reconnect.
                    startReconnect()
                }

                override fun onCompleted() {
                    Log.i(TAG, "Server response stream completed")
                    connectionStatus.set(ConnectionStatus.DISCONNECTED)
                    startReconnect()
                }
            }

            // Create request observer.
            requestObserver = stub?.streamSensorData(responseObserver)
            requestObserver != null

        } catch (e: Exception) {
            Log.e(TAG, "Failed to establish bidirectional stream", e)
            false
        }
    }

    /**
     * Handles server directives.
     */
    private fun handleServerDirective(directive: ServerDirective) {
        uploaderScope.launch {
            try {
                when {
                    directive.hasAck() -> handleAck(directive.ack)
                    directive.hasPolicy() -> handlePolicyUpdate(directive.policy)
                    directive.hasKeyRotation() -> handleKeyRotation(directive.keyRotation)
                    directive.hasEmergency() -> handleEmergencyStop(directive.emergency)
                }

                // Emit to directive stream.
                serverDirectiveChannel.trySend(directive)

            } catch (e: Exception) {
                Log.e(TAG, "Failed to handle server directive", e)
            }
        }
    }

    /**
     * Handles ACK.
     */
    private fun handleAck(ack: Ack) {
        val pendingAck = pendingAcks.remove(ack.packetId)
        if (pendingAck != null) {
            // Compute ACK latency.
            val latency = System.currentTimeMillis() - pendingAck.sentTimestamp
            lastAckLatency = latency
            
            synchronized(ackLatencies) {
                ackLatencies.add(latency)
                // Keep the last 100 samples.
                if (ackLatencies.size > 100) {
                    ackLatencies.removeAt(0)
                }
            }

            totalAcksReceived.incrementAndGet()

            Log.d(TAG, "ACK received: ${ack.packetId}, latency: ${latency}ms, success: ${ack.success}")
        } else {
            Log.w(TAG, "ACK for unknown packet: ${ack.packetId}")
        }
    }

    /**
     * Handles policy updates.
     */
    private fun handlePolicyUpdate(policyUpdate: PolicyUpdate) {
        totalPolicyUpdates.incrementAndGet()
        Log.i(TAG, "Policy update received: ${policyUpdate.policyId}")
        
        // Apply via policy manager.
        uploaderScope.launch {
            policyManager.updatePolicy(policyUpdate)
        }
    }

    /**
     * Checks for ACK timeout.
     */
    private fun checkAckTimeout(packetId: String) {
        val pendingAck = pendingAcks.remove(packetId)
        if (pendingAck != null) {
            Log.w(TAG, "ACK timed out for packet: $packetId")
            // TODO: Add re-send logic if needed.
        }
    }

    /**
     * Handles key rotation notice.
     */
    private fun handleKeyRotation(keyRotation: KeyRotationNotice) {
        Log.i(TAG, "Key rotation notice received: ${keyRotation.newKeyId}")
        // TODO: Implement key rotation.
    }
    
    /**
     * Handles emergency stop directive.
     */
    private fun handleEmergencyStop(emergency: EmergencyStop) {
        Log.w(TAG, "Emergency stop received: ${emergency.reason}")
        // TODO: Implement emergency stop handling.
    }

    /**
     * Handles connection errors.
     */
    private fun handleConnectionError(errorMessage: String) {
        connectionErrors++
        lastErrorTimestamp = System.currentTimeMillis()
        lastErrorMessage = errorMessage
        connectionStatus.set(ConnectionStatus.ERROR)
        
        Log.e(TAG, "Connection error: $errorMessage")
    }

    /**
     * Starts reconnect attempts.
     */
    private fun startReconnect() {
        if (currentEndpoint.isEmpty()) return
        
        reconnectJob?.cancel()
        reconnectJob = uploaderScope.launch {
            connectionStatus.set(ConnectionStatus.RECONNECTING)
            var retryCount = 0
            
            while (retryCount < MAX_RETRY_ATTEMPTS && connectionStatus.get() != ConnectionStatus.CONNECTED) {
                delay(kotlin.math.min(1000 * (1 shl retryCount), 30000).toLong()) // Exponential backoff
                
                Log.i(TAG, "Reconnect attempt (${retryCount + 1}/$MAX_RETRY_ATTEMPTS): $currentEndpoint")
                
                if (connect(currentEndpoint)) {
                    Log.i(TAG, "Reconnect succeeded")
                    return@launch
                }
                
                retryCount++
            }
            
            if (connectionStatus.get() != ConnectionStatus.CONNECTED) {
                Log.e(TAG, "Reconnect failed: max retries reached")
                connectionStatus.set(ConnectionStatus.ERROR)
            }
        }
    }

    /**
     * Returns transmission stats.
     */
    override fun getTransmissionStats(): TransmissionStats {
        return TransmissionStats(
            isFastMode = false,  // Default value; update with real logic.
            fastModeRemainingSeconds = 0,
            lastTriggerType = null,
            successCount = totalAcksReceived.get(),
            failedCount = totalPacketsSent.get() - totalAcksReceived.get(),
            averageLatency = if (ackLatencies.isEmpty()) 0L else ackLatencies.average().toLong()
        )
    }

    /**
     * Returns gRPC status.
     */
    override fun getGrpcStatus(): GrpcStatus {
        return GrpcStatus(
            connectionState = connectionStatus.get(),
            lastAckLatency = lastAckLatency ?: 0L
        )
    }

    /**
     * Returns buffer stats.
     */
    override fun getBufferStats(): BufferStats {
        return BufferStats(
            memorySamples = 0,  // Default value; update with real buffer logic.
            diskBatches = 0,
            sentCount = totalPacketsSent.get(),
            discardedCount = 0L
        )
    }

    /**
     * Returns server endpoint.
     */
    override fun getServerEndpoint(): String {
        return currentEndpoint
    }

    /**
     * Returns basic statistics.
     */
    override fun getStatistics(): Statistics {
        return Statistics(
            totalPacketsSent = totalPacketsSent.get(),
            totalPacketsAcknowledged = totalAcksReceived.get(),
            totalFailures = totalPacketsSent.get() - totalAcksReceived.get()
        )
    }

    /**
     * Returns memory buffer stats.
     */
    override fun getMemoryBufferStats(): MemoryBufferStats {
        return MemoryBufferStats(
            samplesInMemory = 0,  // Default value; update with real logic.
            totalSent = totalPacketsSent.get(),
            totalFailed = totalPacketsSent.get() - totalAcksReceived.get(),
            totalDiscarded = 0L
        )
    }

    /**
     * Returns recent latencies.
     */
    override fun getRecentLatencies(): List<Long> {
        return ackLatencies.toList()
    }

    /**
     * Returns server policy.
     */
    override suspend fun getServerPolicy(): ServerPolicy {
        return ServerPolicy(
            version = "1.0",
            lastUpdated = System.currentTimeMillis(),
            fastModeDurationSeconds = 30,  // Default value.
            anomalyThreshold = 0.8f,       // Default value.
            samplingRates = mapOf(
                "ACCELEROMETER" to 200f,    // 200 Hz
                "GYROSCOPE" to 200f,        // 200 Hz
                "MAGNETOMETER" to 100f      // 100 Hz
            ),
            transmissionStrategy = "ADAPTIVE"
        )
    }
}
