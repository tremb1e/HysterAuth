package com.continuousauth.network

import android.util.Log
import io.grpc.Status
import io.grpc.StatusException
import io.grpc.StatusRuntimeException
import kotlinx.coroutines.delay
import java.io.IOException
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import java.security.cert.CertificateException
import javax.inject.Inject
import javax.inject.Singleton
import javax.net.ssl.SSLException
import kotlin.math.min
import kotlin.math.pow
import kotlin.random.Random

/**
 * Structured error handler.
 *
 * Determines retry strategy based on error type and network conditions.
 */
@Singleton
class ErrorHandler @Inject constructor() {
    
    companion object {
        private const val TAG = "ErrorHandler"
        private const val DEFAULT_BASE_DELAY_MS = 1000L
        private const val DEFAULT_MAX_DELAY_MS = 30000L
        private const val DEFAULT_BACKOFF_MULTIPLIER = 2.0
        private const val DEFAULT_JITTER_FACTOR = 0.1
    }
    
    /**
     * Handles an error and decides the retry strategy.
     */
    suspend fun handleError(
        throwable: Throwable,
        attemptNumber: Int,
        networkState: NetworkState = NetworkState.UNKNOWN
    ): ErrorHandlingResult {
        
        val errorType = classifyError(throwable)
        val retryStrategy = determineRetryStrategy(errorType, networkState, attemptNumber)
        
        Log.w(TAG, "Error handling - type: $errorType, attempt: $attemptNumber, action: ${retryStrategy.action}")
        Log.d(TAG, "Error details", throwable)
        
        return when (retryStrategy.action) {
            RetryAction.IMMEDIATE_RETRY -> {
                ErrorHandlingResult.Retry(
                    delayMs = 0,
                    nextAttempt = attemptNumber + 1,
                    errorType = errorType,
                    message = "Retry immediately"
                )
            }
            
            RetryAction.EXPONENTIAL_BACKOFF -> {
                val delayMs = calculateExponentialBackoffDelay(
                    attemptNumber = attemptNumber,
                    baseDelayMs = retryStrategy.baseDelayMs,
                    maxDelayMs = retryStrategy.maxDelayMs,
                    multiplier = retryStrategy.backoffMultiplier
                )
                
                ErrorHandlingResult.Retry(
                    delayMs = delayMs,
                    nextAttempt = attemptNumber + 1,
                    errorType = errorType,
                    message = "Retry with exponential backoff (delay=${delayMs}ms)"
                )
            }
            
            RetryAction.FIXED_DELAY -> {
                ErrorHandlingResult.Retry(
                    delayMs = retryStrategy.baseDelayMs,
                    nextAttempt = attemptNumber + 1,
                    errorType = errorType,
                    message = "Retry with fixed delay (delay=${retryStrategy.baseDelayMs}ms)"
                )
            }
            
            RetryAction.GIVE_UP -> {
                ErrorHandlingResult.GiveUp(
                    errorType = errorType,
                    message = "Giving up: ${retryStrategy.reason}",
                    originalError = throwable
                )
            }
        }
    }
    
    /**
     * Classifies the error.
     */
    private fun classifyError(throwable: Throwable): ErrorType {
        return when (throwable) {
            // gRPC errors
            is StatusException -> classifyGrpcError(throwable.status)
            is StatusRuntimeException -> classifyGrpcError(throwable.status)
            
            // Network errors
            is ConnectException -> ErrorType.NETWORK_INTERRUPTION
            is SocketTimeoutException -> ErrorType.NETWORK_TIMEOUT
            is UnknownHostException -> ErrorType.DNS_RESOLUTION_FAILED
            
            // SSL/TLS errors
            is SSLException -> when {
                throwable.message?.contains("certificate", ignoreCase = true) == true -> 
                    ErrorType.CERTIFICATE_ERROR
                throwable.message?.contains("handshake", ignoreCase = true) == true -> 
                    ErrorType.TLS_HANDSHAKE_FAILED
                else -> ErrorType.AUTHENTICATION_FAILURE
            }
            
            is CertificateException -> ErrorType.CERTIFICATE_ERROR
            
            // IO errors
            is IOException -> when {
                throwable.message?.contains("timeout", ignoreCase = true) == true -> 
                    ErrorType.NETWORK_TIMEOUT
                throwable.message?.contains("connection", ignoreCase = true) == true -> 
                    ErrorType.NETWORK_INTERRUPTION
                else -> ErrorType.IO_ERROR
            }
            
            // Other errors
            is OutOfMemoryError -> ErrorType.RESOURCE_EXHAUSTION
            is SecurityException -> ErrorType.PERMISSION_DENIED
            
            else -> ErrorType.UNKNOWN_ERROR
        }
    }
    
    /**
     * Classifies gRPC errors.
     */
    private fun classifyGrpcError(status: Status): ErrorType {
        return when (status.code) {
            Status.Code.OK -> ErrorType.NO_ERROR
            
            // Authentication/authorization errors
            Status.Code.UNAUTHENTICATED -> ErrorType.AUTHENTICATION_FAILURE
            Status.Code.PERMISSION_DENIED -> ErrorType.PERMISSION_DENIED
            
            // Network/connection errors
            Status.Code.UNAVAILABLE -> ErrorType.SERVER_UNAVAILABLE
            Status.Code.DEADLINE_EXCEEDED -> ErrorType.NETWORK_TIMEOUT
            Status.Code.CANCELLED -> ErrorType.OPERATION_CANCELLED
            
            // Client errors
            Status.Code.INVALID_ARGUMENT -> ErrorType.INVALID_REQUEST
            Status.Code.NOT_FOUND -> ErrorType.ENDPOINT_NOT_FOUND
            Status.Code.ALREADY_EXISTS -> ErrorType.RESOURCE_CONFLICT
            Status.Code.FAILED_PRECONDITION -> ErrorType.PRECONDITION_FAILED
            Status.Code.OUT_OF_RANGE -> ErrorType.INVALID_REQUEST
            
            // Server errors
            Status.Code.INTERNAL -> ErrorType.SERVER_ERROR
            Status.Code.UNIMPLEMENTED -> ErrorType.FEATURE_NOT_SUPPORTED
            Status.Code.DATA_LOSS -> ErrorType.DATA_CORRUPTION
            
            // Resource errors
            Status.Code.RESOURCE_EXHAUSTED -> ErrorType.RESOURCE_EXHAUSTION
            
            // Unknown errors
            Status.Code.UNKNOWN -> ErrorType.UNKNOWN_ERROR
            
            else -> ErrorType.UNKNOWN_ERROR
        }
    }
    
    /**
     * Determines retry strategy.
     */
    private fun determineRetryStrategy(
        errorType: ErrorType,
        networkState: NetworkState,
        attemptNumber: Int
    ): RetryStrategy {
        
        // Check whether max attempts are reached.
        val maxAttempts = getMaxAttemptsForError(errorType, networkState)
        if (attemptNumber >= maxAttempts) {
            return RetryStrategy(
                action = RetryAction.GIVE_UP,
                reason = "Max retries reached ($maxAttempts)"
            )
        }
        
        return when (errorType) {
            // Immediate retry
            ErrorType.OPERATION_CANCELLED,
            ErrorType.NETWORK_TIMEOUT -> RetryStrategy(
                action = RetryAction.IMMEDIATE_RETRY
            )
            
            // Exponential backoff
            ErrorType.NETWORK_INTERRUPTION,
            ErrorType.SERVER_UNAVAILABLE,
            ErrorType.RESOURCE_EXHAUSTION -> RetryStrategy(
                action = RetryAction.EXPONENTIAL_BACKOFF,
                baseDelayMs = adjustDelayForNetwork(DEFAULT_BASE_DELAY_MS, networkState),
                maxDelayMs = DEFAULT_MAX_DELAY_MS,
                backoffMultiplier = DEFAULT_BACKOFF_MULTIPLIER
            )
            
            // Fixed delay
            ErrorType.SERVER_ERROR,
            ErrorType.DNS_RESOLUTION_FAILED,
            ErrorType.TLS_HANDSHAKE_FAILED -> RetryStrategy(
                action = RetryAction.FIXED_DELAY,
                baseDelayMs = adjustDelayForNetwork(5000L, networkState)
            )
            
            // Do not retry
            ErrorType.AUTHENTICATION_FAILURE,
            ErrorType.PERMISSION_DENIED,
            ErrorType.CERTIFICATE_ERROR,
            ErrorType.INVALID_REQUEST,
            ErrorType.ENDPOINT_NOT_FOUND,
            ErrorType.FEATURE_NOT_SUPPORTED,
            ErrorType.DATA_CORRUPTION,
            ErrorType.PRECONDITION_FAILED -> RetryStrategy(
                action = RetryAction.GIVE_UP,
                reason = "Not retryable: $errorType"
            )
            
            // Conservative fallback
            else -> RetryStrategy(
                action = RetryAction.EXPONENTIAL_BACKOFF,
                baseDelayMs = adjustDelayForNetwork(DEFAULT_BASE_DELAY_MS * 2, networkState),
                maxDelayMs = DEFAULT_MAX_DELAY_MS,
                backoffMultiplier = 1.5
            )
        }
    }
    
    /**
     * Returns the max attempts based on error type and network state.
     */
    private fun getMaxAttemptsForError(errorType: ErrorType, networkState: NetworkState): Int {
        val baseAttempts = when (errorType) {
            ErrorType.NETWORK_INTERRUPTION,
            ErrorType.NETWORK_TIMEOUT,
            ErrorType.SERVER_UNAVAILABLE -> 8
            
            ErrorType.SERVER_ERROR,
            ErrorType.RESOURCE_EXHAUSTION -> 5
            
            ErrorType.DNS_RESOLUTION_FAILED,
            ErrorType.TLS_HANDSHAKE_FAILED -> 3
            
            ErrorType.OPERATION_CANCELLED -> 10
            
            else -> 3
        }
        
        // Adjust attempts based on network quality.
        return when (networkState) {
            NetworkState.CELLULAR_POOR,
            NetworkState.WIFI_POOR -> (baseAttempts * 1.5).toInt()
            
            NetworkState.CELLULAR_EXCELLENT,
            NetworkState.WIFI_EXCELLENT -> maxOf(baseAttempts - 1, 2)
            
            else -> baseAttempts
        }
    }
    
    /**
     * Adjusts delay based on network state.
     */
    private fun adjustDelayForNetwork(baseDelayMs: Long, networkState: NetworkState): Long {
        return when (networkState) {
            NetworkState.CELLULAR_POOR,
            NetworkState.WIFI_POOR -> (baseDelayMs * 2)
            
            NetworkState.CELLULAR_EXCELLENT,
            NetworkState.WIFI_EXCELLENT -> (baseDelayMs * 0.7).toLong()
            
            else -> baseDelayMs
        }
    }
    
    /**
     * Calculates exponential backoff delay.
     */
    private fun calculateExponentialBackoffDelay(
        attemptNumber: Int,
        baseDelayMs: Long,
        maxDelayMs: Long,
        multiplier: Double
    ): Long {
        // Exponential backoff: baseDelay * multiplier^(attempt-1)
        val exponentialDelay = baseDelayMs * multiplier.pow(attemptNumber - 1)
        
        // Add jitter to avoid thundering herd.
        val jitter = Random.nextDouble(-DEFAULT_JITTER_FACTOR, DEFAULT_JITTER_FACTOR)
        val delayWithJitter = exponentialDelay * (1 + jitter)
        
        // Clamp to max delay.
        return min(delayWithJitter.toLong(), maxDelayMs)
    }
    
    /**
     * Delays execution before retry.
     */
    suspend fun executeWithRetry(delayMs: Long) {
        if (delayMs > 0) {
            Log.d(TAG, "Waiting ${delayMs}ms before retry")
            delay(delayMs)
        }
    }
}

/**
 * Error types.
 */
enum class ErrorType {
    NO_ERROR,                   // No error
    
    // Network errors
    NETWORK_INTERRUPTION,       // Network interruption
    NETWORK_TIMEOUT,           // Network timeout
    DNS_RESOLUTION_FAILED,     // DNS resolution failed
    
    // Auth errors
    AUTHENTICATION_FAILURE,     // Authentication failure
    PERMISSION_DENIED,         // Permission denied
    CERTIFICATE_ERROR,         // Certificate error
    TLS_HANDSHAKE_FAILED,      // TLS handshake failed
    
    // Server errors
    SERVER_ERROR,              // Server internal error
    SERVER_UNAVAILABLE,        // Server unavailable
    ENDPOINT_NOT_FOUND,        // Endpoint not found
    FEATURE_NOT_SUPPORTED,     // Feature not supported
    
    // Client errors
    INVALID_REQUEST,           // Invalid request
    RESOURCE_CONFLICT,         // Resource conflict
    PRECONDITION_FAILED,       // Precondition failed
    
    // Resource errors
    RESOURCE_EXHAUSTION,       // Resource exhausted
    DATA_CORRUPTION,           // Data corruption
    
    // Operation errors
    OPERATION_CANCELLED,       // Operation cancelled
    IO_ERROR,                  // IO error
    
    // Other
    UNKNOWN_ERROR              // Unknown error
}

/**
 * Retry actions.
 */
enum class RetryAction {
    IMMEDIATE_RETRY,           // Retry immediately
    EXPONENTIAL_BACKOFF,       // Exponential backoff
    FIXED_DELAY,               // Fixed delay
    GIVE_UP                    // Give up
}

/**
 * Retry strategy.
 */
data class RetryStrategy(
    val action: RetryAction,
    val baseDelayMs: Long = DEFAULT_BASE_DELAY_MS,
    val maxDelayMs: Long = DEFAULT_MAX_DELAY_MS,
    val backoffMultiplier: Double = DEFAULT_BACKOFF_MULTIPLIER,
    val reason: String = ""
) {
    companion object {
        private const val DEFAULT_BASE_DELAY_MS = 1000L
        private const val DEFAULT_MAX_DELAY_MS = 30000L
        private const val DEFAULT_BACKOFF_MULTIPLIER = 2.0
    }
}

/**
 * Error handling result.
 */
sealed class ErrorHandlingResult {
    /**
     * Retryable result.
     */
    data class Retry(
        val delayMs: Long,
        val nextAttempt: Int,
        val errorType: ErrorType,
        val message: String
    ) : ErrorHandlingResult()
    
    /**
     * Non-retryable result.
     */
    data class GiveUp(
        val errorType: ErrorType,
        val message: String,
        val originalError: Throwable
    ) : ErrorHandlingResult()
}
