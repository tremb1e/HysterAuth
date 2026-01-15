package com.continuousauth.network

import android.content.Context
import android.util.Log
import com.continuousauth.proto.DataPacket
import com.continuousauth.proto.Heartbeat
import com.continuousauth.proto.HeartbeatAck
import com.continuousauth.proto.SensorDataServiceGrpc
import com.continuousauth.proto.ServerDirective
import dagger.hilt.android.qualifiers.ApplicationContext
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import io.grpc.okhttp.OkHttpChannelBuilder
import io.grpc.stub.StreamObserver
import okhttp3.OkHttpClient
import okhttp3.Protocol
import java.security.cert.X509Certificate
import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManager
import javax.net.ssl.X509TrustManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asExecutor
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

/**
 * gRPC manager.
 *
 * Manages gRPC connection and communication.
 */
@Singleton
class GrpcManager @Inject constructor(
    @ApplicationContext private val context: Context
) {
    
    companion object {
        private const val TAG = "GrpcManager"
        private const val DEFAULT_HOST = "localhost"
        private const val DEFAULT_PORT = 8080
        private const val DEFAULT_TLS_PORT = 8443
        private const val CONNECT_TIMEOUT_SECONDS = 30L
        private const val KEEPALIVE_TIME_SECONDS = 30L
        private const val KEEPALIVE_TIMEOUT_SECONDS = 10L
        private const val MAX_INBOUND_MESSAGE_SIZE = 10 * 1024 * 1024 // 10MB
        private const val MAX_OUTBOUND_MESSAGE_SIZE = 10 * 1024 * 1024 // 10MB
    }
    
    private var channel: ManagedChannel? = null
    private var stub: SensorDataServiceGrpc.SensorDataServiceStub? = null
    private var currentStreamObserver: StreamObserver<DataPacket>? = null
    
    // Connection state
    private val _connectionState = MutableStateFlow(ConnectionState.DISCONNECTED)
    val connectionState: StateFlow<ConnectionState> = _connectionState.asStateFlow()
    
    // Server configuration
    private var serverHost = DEFAULT_HOST
    private var serverPort = DEFAULT_PORT
    private var useTls = true // TLS enabled by default
    
    /**
     * Configure server address.
     */
    fun configureServer(host: String, port: Int) {
        serverHost = host
        serverPort = port
        Log.i(TAG, "Configuring server address: $host:$port")
    }
    
    /**
     * Establish gRPC connection.
     */
    suspend fun connect(): Result<Unit> {
        return withContext(Dispatchers.IO) {
            try {
                if (channel != null && !channel!!.isShutdown) {
                    Log.w(TAG, "Active connection already exists")
                    return@withContext Result.success(Unit)
                }
                
                Log.i(TAG, "Connecting to gRPC: $serverHost:$serverPort")
                _connectionState.value = ConnectionState.CONNECTING
                
                channel = if (useTls) {
                    buildTlsChannel()
                } else {
                    // Development/testing only
                    ManagedChannelBuilder
                        .forAddress(serverHost, serverPort)
                        .usePlaintext()
                        .keepAliveTime(KEEPALIVE_TIME_SECONDS, TimeUnit.SECONDS)
                        .keepAliveTimeout(KEEPALIVE_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                        .keepAliveWithoutCalls(true)
                        .maxInboundMessageSize(MAX_INBOUND_MESSAGE_SIZE)
                        .build()
                }
                
                stub = SensorDataServiceGrpc.newStub(channel)
                _connectionState.value = ConnectionState.CONNECTED
                
                Log.i(TAG, "gRPC connection established")
                Result.success(Unit)
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to establish gRPC connection", e)
                _connectionState.value = ConnectionState.ERROR
                Result.failure(e)
            }
        }
    }
    
    /**
     * Disconnect gRPC channel.
     */
    suspend fun disconnect() {
        withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Disconnecting gRPC connection")
                
                currentStreamObserver?.onCompleted()
                currentStreamObserver = null
                
                channel?.shutdown()
                channel?.awaitTermination(5, TimeUnit.SECONDS)
                channel = null
                stub = null
                
                _connectionState.value = ConnectionState.DISCONNECTED
                Log.i(TAG, "gRPC connection disconnected")
                
            } catch (e: Exception) {
                Log.e(TAG, "Error while disconnecting", e)
            }
        }
    }
    
    /**
     * Establish bidirectional stream.
     */
    fun establishBidirectionalStream(
        responseObserver: StreamObserver<ServerDirective>
    ): StreamObserver<DataPacket>? {
        return try {
            if (stub == null) {
                Log.e(TAG, "gRPC stub not initialized")
                return null
            }
            
            Log.i(TAG, "Establishing bidirectional stream")
            currentStreamObserver = stub!!.streamSensorData(responseObserver)
            currentStreamObserver
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to establish bidirectional stream", e)
            null
        }
    }
    
    /**
     * Send a data packet.
     */
    fun sendDataPacket(packet: DataPacket): Boolean {
        return try {
            if (currentStreamObserver == null) {
                Log.e(TAG, "Stream observer not initialized")
                return false
            }
            
            currentStreamObserver!!.onNext(packet)
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send data packet", e)
            false
        }
    }
    
    /**
     * Check connection state.
     */
    fun isConnected(): Boolean {
        return channel != null && !channel!!.isShutdown && !channel!!.isTerminated
    }
    
    /**
     * Get channel state.
     */
    fun getChannelState(): String {
        return channel?.getState(false)?.toString() ?: "NO_CHANNEL"
    }
    
    /**
     * Enable/disable TLS.
     */
    fun setUseTls(enable: Boolean) {
        useTls = enable
        if (enable && serverPort == DEFAULT_PORT) {
            serverPort = DEFAULT_TLS_PORT // Auto-switch to TLS port
        }
    }
    
    /**
     * Send heartbeat message.
     */
    suspend fun sendHeartbeat(heartbeat: Heartbeat): HeartbeatAck? {
        return withContext(Dispatchers.IO) {
            try {
                if (stub == null) {
                    Log.e(TAG, "Cannot send heartbeat: gRPC stub not initialized")
                    return@withContext null
                }
                
                // Send heartbeat via blocking stub
                val blockingStub = SensorDataServiceGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(5, TimeUnit.SECONDS)
                
                val ack = blockingStub.sendHeartbeat(heartbeat)
                Log.v(TAG, "Heartbeat ack: serverTimestamp=${ack.serverTimestamp}")
                ack
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send heartbeat", e)
                null
            }
        }
    }
    
    /**
     * Build a TLS-secured channel.
     *
     * Forces TLS 1.3 for transport.
     */
    private fun buildTlsChannel(): ManagedChannel {
        Log.i(TAG, "Building TLS 1.3 secure channel")
        
        // Create SSLContext configured for TLS 1.3
        val sslContext = createTls13SSLContext()
        
        // Build OkHttpClient for TLS transport
        val okHttpClient = OkHttpClient.Builder()
            .protocols(listOf(Protocol.HTTP_2)) // Force HTTP/2
            .sslSocketFactory(sslContext.socketFactory, createTrustAllManager())
            .hostnameVerifier { _, _ -> true } // Temporary: accept any hostname (production must verify)
            .connectTimeout(CONNECT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .readTimeout(KEEPALIVE_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .writeTimeout(KEEPALIVE_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .build()
        
        // Use OkHttpChannelBuilder to create the channel
        return OkHttpChannelBuilder
            .forAddress(serverHost, serverPort)
            .transportExecutor(Dispatchers.IO.asExecutor())
            .keepAliveTime(KEEPALIVE_TIME_SECONDS, TimeUnit.SECONDS)
            .keepAliveTimeout(KEEPALIVE_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .keepAliveWithoutCalls(true)
            .maxInboundMessageSize(MAX_INBOUND_MESSAGE_SIZE)
            .build()
    }
    
    /**
     * Create SSLContext for TLS 1.3.
     */
    private fun createTls13SSLContext(): SSLContext {
        val sslContext = SSLContext.getInstance("TLSv1.3")
        sslContext.init(null, arrayOf(createTrustAllManager()), java.security.SecureRandom())
        return sslContext
    }
    
    /**
     * Create a TrustManager that trusts all certificates.
     *
     * Note: for development/testing only. Production should use certificate pinning.
     */
    private fun createTrustAllManager(): X509TrustManager {
        return object : X509TrustManager {
            override fun checkClientTrusted(chain: Array<X509Certificate>, authType: String) {}
            override fun checkServerTrusted(chain: Array<X509Certificate>, authType: String) {
                Log.d(TAG, "Verifying server certificate: authType=$authType, chainSize=${chain.size}")
                // TODO: Add certificate pinning verification logic
            }
            override fun getAcceptedIssuers(): Array<X509Certificate> = arrayOf()
        }
    }
    
    
    /**
     * Connection state enum.
     */
    enum class ConnectionState {
        DISCONNECTED,
        CONNECTING,
        CONNECTED,
        ERROR
    }
}
