package com.continuousauth.network

import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Query

/**
 * API service interface.
 *
 * Defines HTTP/REST endpoints used to communicate with the server.
 */
interface ApiService {
    
    /**
     * Fetches the server public key.
     */
    @GET("/api/v1/crypto/public-key")
    suspend fun getPublicKey(): PublicKeyResponse
    
    /**
     * Fetches the HMAC key.
     */
    @POST("/api/v1/crypto/hmac-key")
    suspend fun getHmacKey(@Query("device_instance_id") deviceInstanceId: String): HmacKeyResponse
    
    /**
     * Registers the device.
     */
    @POST("/api/v1/device/register")
    suspend fun registerDevice(
        @Query("device_id_hash") deviceIdHash: String,
        @Body deviceInfo: com.continuousauth.registration.DeviceRegistration.DeviceInfo
    ): RegistrationResponse
    
    /**
     * Checks device status.
     */
    @GET("/api/v1/device/status")
    suspend fun checkDeviceStatus(@Query("device_id_hash") deviceIdHash: String): DeviceStatusResponse
    
    /**
     * Fetches the latest policy.
     */
    @GET("/api/v1/policy/latest")
    suspend fun getLatestPolicy(@Query("device_id_hash") deviceIdHash: String): PolicyResponse
    
    /**
     * Reports metrics.
     */
    @POST("/api/v1/metrics/report")
    suspend fun reportMetrics(@Body metrics: MetricsReportRequest): MetricsReportResponse
    
    /**
     * Requests data deletion (GDPR).
     */
    @POST("/api/v1/data/delete")
    suspend fun requestDataDeletion(@Query("device_id_hash") deviceIdHash: String): DataDeletionResponse
}

// Response models.
data class PublicKeyResponse(
    val publicKey: ByteArray,
    val keyId: String,
    val fingerprint: String,
    val validFrom: Long,
    val validUntil: Long
)

data class HmacKeyResponse(
    val hmacKeyId: String,
    val hmacKey: String,
    val createdAt: Long,
    val expiresAt: Long?
)

data class RegistrationResponse(
    val success: Boolean,
    val sessionId: String,
    val serverTime: Long,
    val message: String?
)

data class DeviceStatusResponse(
    val isActive: Boolean,
    val isSuspended: Boolean,
    val lastSeen: Long?,
    val suspensionReason: String?
)

data class PolicyResponse(
    val policyId: String,
    val policyVersion: String,
    val batchIntervalMs: Int,
    val maxPayloadSizeBytes: Int,
    val transmissionProfile: String,
    val compressionAlgorithm: String,
    val sensorSamplingRates: Map<String, Int>,
    val enabledSensors: List<String>
)

data class MetricsReportRequest(
    val deviceIdHash: String,
    val timestamp: Long,
    val metrics: Map<String, Float>
)

data class MetricsReportResponse(
    val accepted: Boolean,
    val nextReportAfterMs: Long?
)

data class DataDeletionResponse(
    val requestId: String,
    val status: String,
    val estimatedCompletionTime: Long,
    val message: String
)
