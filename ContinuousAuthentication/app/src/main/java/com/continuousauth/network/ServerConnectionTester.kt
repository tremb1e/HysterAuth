package com.continuousauth.network

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import okhttp3.OkHttpClient
import okhttp3.Request
import java.net.InetSocketAddress
import java.net.Socket
import java.net.SocketTimeoutException
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton
import io.grpc.ManagedChannelBuilder
import io.grpc.ConnectivityState
import com.continuousauth.proto.SensorDataServiceGrpc
import kotlinx.coroutines.TimeoutCancellationException

/**
 * Server connection tester.
 *
 * Tests server reachability and latency.
 */
@Singleton
class ServerConnectionTester @Inject constructor(
    private val grpcManager: GrpcManager
) {
    
    companion object {
        private const val TAG = "ServerConnectionTester"
        private const val DEFAULT_TIMEOUT_MS = 5000L
        private const val SOCKET_CONNECT_TIMEOUT_MS = 3000
    }
    
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(DEFAULT_TIMEOUT_MS, TimeUnit.MILLISECONDS)
        .readTimeout(DEFAULT_TIMEOUT_MS, TimeUnit.MILLISECONDS)
        .build()
    
    /**
     * Server connection test result.
     */
    data class TestResult(
        val isReachable: Boolean,
        val latencyMs: Long? = null,
        val statusCode: Int? = null,
        val errorMessage: String? = null,
        val testType: TestType,
        val details: Map<String, String> = emptyMap()
    )
    
    /**
     * Test type.
     */
    enum class TestType {
        SOCKET_TEST,    // TCP socket test
        HTTP_TEST,      // HTTP request test
        GRPC_TEST      // gRPC connectivity test
    }
    
    /**
     * Runs an end-to-end server connectivity test.
     */
    suspend fun testServerConnection(
        serverIp: String,
        serverPort: Int,
        useHttps: Boolean = false,
        testGrpc: Boolean = true
    ): TestResult = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Testing server connection: $serverIp:$serverPort")
            
            // First: TCP socket connectivity test.
            val socketResult = testSocketConnection(serverIp, serverPort)
            if (!socketResult.isReachable) {
                Log.w(TAG, "Socket test failed: ${socketResult.errorMessage}")
                return@withContext socketResult
            }
            
            // Optional: gRPC test.
            if (testGrpc) {
                val grpcResult = testGrpcConnection(serverIp, serverPort)
                if (grpcResult.isReachable) {
                    return@withContext grpcResult
                }
            }
            
            // Fallback: HTTP/HTTPS test.
            val httpResult = testHttpConnection(serverIp, serverPort, useHttps)
            return@withContext httpResult
            
        } catch (e: Exception) {
            Log.e(TAG, "Server connection test error", e)
            TestResult(
                isReachable = false,
                errorMessage = "Test error: ${e.message}",
                testType = TestType.SOCKET_TEST
            )
        }
    }
    
    /**
     * Tests TCP socket connectivity.
     */
    private suspend fun testSocketConnection(
        serverIp: String,
        serverPort: Int
    ): TestResult = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()
        
        try {
            withTimeout(SOCKET_CONNECT_TIMEOUT_MS.toLong()) {
                Socket().use { socket ->
                    socket.connect(
                        InetSocketAddress(serverIp, serverPort),
                        SOCKET_CONNECT_TIMEOUT_MS
                    )
                    
                    val latency = System.currentTimeMillis() - startTime
                    
                    Log.i(TAG, "Socket connected: $serverIp:$serverPort, latency: ${latency}ms")
                    
                    TestResult(
                        isReachable = true,
                        latencyMs = latency,
                        testType = TestType.SOCKET_TEST,
                        details = mapOf(
                            "local_address" to socket.localAddress.toString(),
                            "remote_address" to socket.remoteSocketAddress.toString()
                        )
                    )
                }
            }
        } catch (e: SocketTimeoutException) {
            TestResult(
                isReachable = false,
                errorMessage = "Connection timed out",
                testType = TestType.SOCKET_TEST
            )
        } catch (e: Exception) {
            TestResult(
                isReachable = false,
                errorMessage = "Socket error: ${e.message}",
                testType = TestType.SOCKET_TEST
            )
        }
    }
    
    /**
     * Tests HTTP/HTTPS connectivity.
     */
    private suspend fun testHttpConnection(
        serverIp: String,
        serverPort: Int,
        useHttps: Boolean
    ): TestResult = withContext(Dispatchers.IO) {
        val protocol = if (useHttps) "https" else "http"
        val url = "$protocol://$serverIp:$serverPort/health"
        val startTime = System.currentTimeMillis()
        
        try {
            val request = Request.Builder()
                .url(url)
                .head() // Use HEAD to minimize payload.
                .build()
            
            val response = httpClient.newCall(request).execute()
            val latency = System.currentTimeMillis() - startTime
            
            response.use {
                Log.i(TAG, "HTTP test completed: $url, code: ${response.code}, latency: ${latency}ms")
                
                TestResult(
                    isReachable = response.isSuccessful,
                    latencyMs = latency,
                    statusCode = response.code,
                    testType = TestType.HTTP_TEST,
                    details = mapOf(
                        "protocol" to response.protocol.toString(),
                        "message" to response.message
                    )
                )
            }
        } catch (e: Exception) {
            TestResult(
                isReachable = false,
                errorMessage = "HTTP error: ${e.message}",
                testType = TestType.HTTP_TEST
            )
        }
    }
    
    /**
     * Tests gRPC connectivity.
     *
     * Verifies that the gRPC service is reachable.
     */
    private suspend fun testGrpcConnection(
        serverIp: String,
        serverPort: Int
    ): TestResult = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()
        var testChannel: io.grpc.ManagedChannel? = null
        
        try {
            // Create a temporary gRPC channel for testing.
            testChannel = ManagedChannelBuilder
                .forAddress(serverIp, serverPort)
                .usePlaintext() // Development/testing only
                .build()
            
            // Wait for channel readiness.
            withTimeout(DEFAULT_TIMEOUT_MS) {
                var attempts = 0
                while (testChannel.getState(true) != ConnectivityState.READY && attempts < 50) {
                    delay(100) // 100ms
                    attempts++
                }
                
                if (testChannel.getState(false) != ConnectivityState.READY) {
                    throw Exception("Failed to establish gRPC connection")
                }
            }
            
            val latency = System.currentTimeMillis() - startTime
            
            // Create a stub to validate the service symbol exists.
            val stub = SensorDataServiceGrpc.newBlockingStub(testChannel)
                .withDeadlineAfter(2, TimeUnit.SECONDS)
            
            // Channel is ready; treat as reachable.
            Log.i(TAG, "gRPC test succeeded: $serverIp:$serverPort, latency: ${latency}ms")
            
            TestResult(
                isReachable = true,
                latencyMs = latency,
                testType = TestType.GRPC_TEST,
                details = mapOf(
                    "state" to testChannel.getState(false).toString(),
                    "service" to "SensorDataService"
                )
            )
            
        } catch (e: TimeoutCancellationException) {
            Log.e(TAG, "gRPC connection timed out")
            TestResult(
                isReachable = false,
                errorMessage = "gRPC connection timed out",
                testType = TestType.GRPC_TEST
            )
        } catch (e: Exception) {
            Log.e(TAG, "gRPC connection failed", e)
            TestResult(
                isReachable = false,
                errorMessage = "gRPC unavailable: ${e.message}",
                testType = TestType.GRPC_TEST
            )
        } finally {
            // Clean up test channel.
            try {
                testChannel?.shutdown()
                testChannel?.awaitTermination(1, TimeUnit.SECONDS)
            } catch (e: Exception) {
                Log.w(TAG, "Error while closing test channel", e)
            }
        }
    }
    
    /**
     * Tests multiple servers.
     */
    suspend fun testMultipleServers(
        servers: List<Pair<String, Int>>
    ): List<Pair<Pair<String, Int>, TestResult>> = withContext(Dispatchers.IO) {
        servers.map { server ->
            server to testServerConnection(server.first, server.second)
        }
    }
    
    /**
     * Returns a human-readable description of the result.
     */
    fun getTestResultDescription(result: TestResult): String {
        return buildString {
            if (result.isReachable) {
                append("✓ Server reachable")
                result.latencyMs?.let { append(" (latency: ${it}ms)") }
                result.statusCode?.let { append(" [HTTP: $it]") }
            } else {
                append("✗ Server unreachable")
                result.errorMessage?.let { append(" - $it") }
            }
            append(" [${result.testType}]")
        }
    }
}
