package com.continuousauth.network

import android.util.Log
import com.continuousauth.proto.DataPacket
import com.continuousauth.proto.Heartbeat
import com.continuousauth.proto.HeartbeatAck
import com.continuousauth.proto.ServerDirective
import io.grpc.stub.StreamObserver
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.min

/**
 * Connection manager.
 *
 * Manages network connectivity, reconnection, and heartbeat.
 */
@Singleton
class ConnectionManager @Inject constructor(
    private val grpcManager: GrpcManager,
    private val uploadManager: UploadManager
) {
    
    companion object {
        private const val TAG = "ConnectionManager"
        
        // Retry settings
        private const val MAX_RETRIES = 10
        private const val BASE_DELAY = 1000L // 1 second
        private const val MAX_DELAY = 60_000L // Max 60 seconds
        
        // Heartbeat settings
        private const val HEARTBEAT_INTERVAL = 30_000L // 30 seconds
        private const val HEARTBEAT_TIMEOUT = 5_000L // 5 seconds timeout
        
        // Connection check interval
        private const val CONNECTION_CHECK_INTERVAL = 10_000L // 10 seconds
    }
    
    /**
     * Connection status.
     */
    enum class ConnectionStatus {
        DISCONNECTED,
        CONNECTING,
        CONNECTED,
        RECONNECTING,
        ERROR,
        SUSPENDED  // Server requested suspension
    }
    
    /**
     * Connection events.
     */
    sealed class ConnectionEvent {
        object Connected : ConnectionEvent()
        object Disconnected : ConnectionEvent()
        data class Error(val message: String, val throwable: Throwable? = null) : ConnectionEvent()
        data class Reconnecting(val attempt: Int, val maxAttempts: Int) : ConnectionEvent()
        data class ServerDirectiveReceived(val directive: ServerDirective) : ConnectionEvent()
    }
    
    // State management
    private val _connectionStatus = MutableStateFlow(ConnectionStatus.DISCONNECTED)
    val connectionStatus: StateFlow<ConnectionStatus> = _connectionStatus.asStateFlow()
    
    private val _connectionEvents = MutableStateFlow<ConnectionEvent?>(null)
    val connectionEvents: StateFlow<ConnectionEvent?> = _connectionEvents.asStateFlow()
    
    // Coroutine management
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var heartbeatJob: Job? = null
    private var connectionMonitorJob: Job? = null
    private var reconnectJob: Job? = null
    
    // Connection state
    private var retryCount = 0
    private var lastConnectionTime = 0L
    private var lastHeartbeatTime = 0L
    private var currentStreamObserver: StreamObserver<DataPacket>? = null
    
    // Callback handler
    private var serverDirectiveHandler: ((ServerDirective) -> Unit)? = null
    
    /**
     * Initialize connection manager.
     */
    fun initialize(serverHost: String, serverPort: Int) {
        Log.i(TAG, "Initializing connection manager: $serverHost:$serverPort")
        grpcManager.configureServer(serverHost, serverPort)
    }
    
    /**
     * Connect with retry.
     */
    suspend fun connectWithRetry(): Result<Unit> {
        return withContext(Dispatchers.IO) {
            Log.i(TAG, "Starting connection (with retries)")
            _connectionStatus.value = ConnectionStatus.CONNECTING
            retryCount = 0
            
            while (retryCount < MAX_RETRIES) {
                try {
                    // Attempt connection
                    val result = grpcManager.connect()
                    
                    if (result.isSuccess) {
                        onConnectionEstablished()
                        return@withContext Result.success(Unit)
                    }
                    
                    // Connection failed; prepare to retry
                    handleConnectionFailure(result.exceptionOrNull())
                    
                } catch (e: Exception) {
                    handleConnectionFailure(e)
                }
                
                // Compute backoff delay
                val delay = calculateBackoff(retryCount)
                Log.i(TAG, "Connection attempt ${retryCount + 1}/$MAX_RETRIES failed; retrying in ${delay}ms")
                
                _connectionStatus.value = ConnectionStatus.RECONNECTING
                _connectionEvents.value = ConnectionEvent.Reconnecting(retryCount + 1, MAX_RETRIES)
                
                delay(delay)
                retryCount++
            }
            
            // Exceeded max retries
            Log.e(TAG, "Connection failed: max retries reached")
            _connectionStatus.value = ConnectionStatus.ERROR
            _connectionEvents.value = ConnectionEvent.Error("Connection failed: max retries reached")
            Result.failure(Exception("Connection failed: max retries reached"))
        }
    }
    
    /**
     * Disconnect.
     */
    suspend fun disconnect() {
        Log.i(TAG, "Disconnecting")
        
        // Cancel coroutine jobs
        heartbeatJob?.cancel()
        connectionMonitorJob?.cancel()
        reconnectJob?.cancel()
        
        // Close stream
        currentStreamObserver?.onCompleted()
        currentStreamObserver = null
        
        // Disconnect gRPC channel
        grpcManager.disconnect()
        
        _connectionStatus.value = ConnectionStatus.DISCONNECTED
        _connectionEvents.value = ConnectionEvent.Disconnected
    }
    
    /**
     * Establish bidirectional stream.
     */
    fun establishBidirectionalStream(): Boolean {
        Log.i(TAG, "Establishing bidirectional stream")
        
        val responseObserver = object : StreamObserver<ServerDirective> {
            override fun onNext(directive: ServerDirective) {
                Log.d(TAG, "Received server directive: ${directive.directiveCase}")
                handleServerDirective(directive)
            }
            
            override fun onError(t: Throwable) {
                Log.e(TAG, "Stream error", t)
                handleStreamError(t)
            }
            
            override fun onCompleted() {
                Log.i(TAG, "Stream completed")
                handleStreamCompleted()
            }
        }
        
        currentStreamObserver = grpcManager.establishBidirectionalStream(responseObserver)
        return currentStreamObserver != null
    }
    
    /**
     * Send a data packet.
     */
    fun sendDataPacket(packet: DataPacket): Boolean {
        if (_connectionStatus.value != ConnectionStatus.CONNECTED) {
            Log.w(TAG, "Not connected; cannot send data packet")
            return false
        }
        
        return try {
            currentStreamObserver?.onNext(packet)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send data packet", e)
            // May require reconnection
            handleSendError(e)
            false
        }
    }
    
    /**
     * Set server directive handler.
     */
    fun setServerDirectiveHandler(handler: (ServerDirective) -> Unit) {
        serverDirectiveHandler = handler
    }
    
    /**
     * Called when connection is established.
     */
    private fun onConnectionEstablished() {
        Log.i(TAG, "Connection established")
        
        retryCount = 0
        lastConnectionTime = System.currentTimeMillis()
        _connectionStatus.value = ConnectionStatus.CONNECTED
        _connectionEvents.value = ConnectionEvent.Connected
        
        // Establish bidirectional stream
        if (!establishBidirectionalStream()) {
            Log.e(TAG, "Failed to establish bidirectional stream")
            return
        }
        
        // Start heartbeat
        startHeartbeat()
        
        // Start connection monitoring
        startConnectionMonitor()
    }
    
    /**
     * Start heartbeat loop.
     */
    private fun startHeartbeat() {
        heartbeatJob?.cancel()
        heartbeatJob = scope.launch {
            Log.i(TAG, "Starting heartbeat")
            
            while (isActive && _connectionStatus.value == ConnectionStatus.CONNECTED) {
                delay(HEARTBEAT_INTERVAL)
                
                try {
                    sendHeartbeat()
                } catch (e: Exception) {
                    Log.e(TAG, "Heartbeat send failed", e)
                    // Heartbeat failure may require reconnection
                    triggerReconnect()
                    break
                }
            }
        }
    }
    
    /**
     * Send heartbeat.
     */
    private suspend fun sendHeartbeat() {
        Log.v(TAG, "Sending heartbeat")
        
        try {
            // Collect pending packet count and last sequence number
            val pendingPackets = uploadManager.getPendingPacketsCount()
            val lastSeqNo = uploadManager.getLastPacketSeqNo()
            
            val heartbeat = Heartbeat.newBuilder()
                .setClientTimestamp(System.currentTimeMillis())
                .setPendingPackets(pendingPackets)
                .setLastPacketSeqNo(lastSeqNo)
                .build()
            
            // Send heartbeat via gRPC and wait for response
            val heartbeatResponse = withTimeoutOrNull(HEARTBEAT_TIMEOUT) {
                grpcManager.sendHeartbeat(heartbeat)
            }
            
            if (heartbeatResponse != null) {
                // Handle heartbeat response
                val latency = System.currentTimeMillis() - heartbeatResponse.clientTimestampEcho
                Log.d(TAG, "Heartbeat acknowledged, latency: ${latency}ms")
                
                // Update connection status
                if (_connectionStatus.value != ConnectionStatus.CONNECTED) {
                    _connectionStatus.value = ConnectionStatus.CONNECTED
                }
                lastHeartbeatTime = System.currentTimeMillis()
            } else {
                Log.w(TAG, "Heartbeat timed out")
                throw Exception("Heartbeat timeout")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send heartbeat", e)
            throw e
        }
    }
    
    /**
     * Start connection monitoring.
     */
    private fun startConnectionMonitor() {
        connectionMonitorJob?.cancel()
        connectionMonitorJob = scope.launch {
            Log.i(TAG, "Starting connection monitor")
            
            while (isActive) {
                delay(CONNECTION_CHECK_INTERVAL)
                
                if (!grpcManager.isConnected()) {
                    Log.w(TAG, "Connection lost")
                    triggerReconnect()
                    break
                }
                
                // Record connection state
                val channelState = grpcManager.getChannelState()
                Log.v(TAG, "Channel state: $channelState")
            }
        }
    }
    
    /**
     * Trigger reconnect.
     */
    private fun triggerReconnect() {
        if (reconnectJob?.isActive == true) {
            Log.d(TAG, "Reconnect already in progress")
            return
        }
        
        reconnectJob = scope.launch {
            Log.i(TAG, "Triggering reconnect")
            
            // Disconnect current connection first
            disconnect()
            
            // Wait a bit
            delay(1000)
            
            // Attempt reconnect
            connectWithRetry()
        }
    }
    
    /**
     * Calculate exponential backoff delay.
     */
    private fun calculateBackoff(attempt: Int): Long {
        // Exponential backoff: base * 2^attempt, capped at MAX_DELAY.
        val delay = BASE_DELAY * (1 shl attempt)
        return min(delay, MAX_DELAY)
    }
    
    /**
     * Handle connection failure.
     */
    private fun handleConnectionFailure(error: Throwable?) {
        Log.e(TAG, "Connection failed", error)
        _connectionEvents.value = ConnectionEvent.Error(
            error?.message ?: "Unknown error",
            error
        )
    }
    
    /**
     * Handle server directives.
     */
    private fun handleServerDirective(directive: ServerDirective) {
        // Emit event
        _connectionEvents.value = ConnectionEvent.ServerDirectiveReceived(directive)
        
        // Invoke handler
        serverDirectiveHandler?.invoke(directive)
        
        // Special directives
        when (directive.directiveCase) {
            ServerDirective.DirectiveCase.EMERGENCY -> {
                handleEmergencyStop(directive.emergency)
            }
            ServerDirective.DirectiveCase.KEY_ROTATION -> {
                Log.i(TAG, "Received key rotation notice")
            }
            else -> {
                // Other directives are handled by the handler
            }
        }
    }
    
    /**
     * Handle emergency stop.
     */
    private fun handleEmergencyStop(emergency: com.continuousauth.proto.EmergencyStop) {
        Log.w(TAG, "Emergency stop directive received: ${emergency.reason}")
        _connectionStatus.value = ConnectionStatus.SUSPENDED
        
        scope.launch {
            // Stop all activities
            heartbeatJob?.cancel()
            connectionMonitorJob?.cancel()
            
            // Clear local cache if requested
            if (emergency.clearLocalCache) {
                Log.i(TAG, "Clearing local cache")
                // TODO: Call cache clearing implementation
            }
            
            // Reconnect after the requested delay
            if (emergency.stopDurationSec > 0) {
                delay(emergency.stopDurationSec * 1000L)
                connectWithRetry()
            }
        }
    }
    
    /**
     * Handle stream error.
     */
    private fun handleStreamError(error: Throwable) {
        Log.e(TAG, "Stream error", error)
        _connectionEvents.value = ConnectionEvent.Error("Stream error", error)
        
        // Trigger reconnect
        triggerReconnect()
    }
    
    /**
     * Handle stream completion.
     */
    private fun handleStreamCompleted() {
        Log.i(TAG, "Stream completed")
        currentStreamObserver = null
        
        // If still connected, re-establish the stream
        if (_connectionStatus.value == ConnectionStatus.CONNECTED) {
            establishBidirectionalStream()
        }
    }
    
    /**
     * Handle send error.
     */
    private fun handleSendError(error: Throwable) {
        Log.e(TAG, "Send error", error)
        
        // Reconnect if needed
        if (!grpcManager.isConnected()) {
            triggerReconnect()
        }
    }
    
    /**
     * Cleanup resources.
     */
    fun cleanup() {
        Log.i(TAG, "Cleaning up connection manager")
        scope.cancel()
    }
}
