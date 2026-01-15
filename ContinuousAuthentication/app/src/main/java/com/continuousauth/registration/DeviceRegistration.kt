package com.continuousauth.registration

import android.util.Log
import com.continuousauth.BuildConfig
import com.continuousauth.crypto.KeyManagementService
import com.continuousauth.network.ApiService
import com.continuousauth.utils.HmacKeyManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.security.MessageDigest
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Device registration and key exchange.
 *
 * Handles device registration, server public key retrieval and fingerprint verification,
 * and HMAC key provisioning.
 */
@Singleton
class DeviceRegistration @Inject constructor(
    private val apiService: ApiService,
    private val keyManagementService: KeyManagementService,
    private val hmacKeyManager: HmacKeyManager
) {
    
    companion object {
        private const val TAG = "DeviceRegistration"
    }
    
    data class RegistrationResult(
        val success: Boolean,
        val deviceIdHash: String,
        val sessionId: String?,
        val errorMessage: String?
    )
    
    /**
     * Run the full device registration flow:
     * generate device ID -> fetch server public key -> verify fingerprint -> store keys -> register.
     */
    suspend fun register(): RegistrationResult = withContext(Dispatchers.IO) {
        try {
            Log.i(TAG, "Starting device registration flow")
            
            // 1. Generate device identifier
            val deviceInstanceId = UUID.randomUUID().toString()
            Log.d(TAG, "Generated device instance ID: $deviceInstanceId")
            
            // 2. Fetch server public key (HTTPS)
            Log.d(TAG, "Fetching server public key")
            val publicKeyResponse = apiService.getPublicKey()
            
            // 3. Verify public key fingerprint (optional: pinned fingerprint)
            val expectedFingerprint = BuildConfig.SERVER_KEY_FINGERPRINT
            if (expectedFingerprint.isNotEmpty() && publicKeyResponse.fingerprint != expectedFingerprint) {
                val error =
                    "Public key fingerprint mismatch: expected=$expectedFingerprint, actual=${publicKeyResponse.fingerprint}"
                Log.e(TAG, error)
                throw SecurityException(error)
            }
            Log.d(TAG, "Public key fingerprint verified")
            
            // 4. Save public key for subsequent encryption
            keyManagementService.saveServerPublicKey(
                publicKey = publicKeyResponse.publicKey,
                keyId = publicKeyResponse.keyId,
                fingerprint = publicKeyResponse.fingerprint
            )
            Log.d(TAG, "Server public key saved: keyId=${publicKeyResponse.keyId}")
            
            // 5. Fetch and save HMAC key
            Log.d(TAG, "Fetching HMAC key")
            val hmacKeyResponse = apiService.getHmacKey(deviceInstanceId)
            
            hmacKeyManager.saveHmacKey(
                keyId = hmacKeyResponse.hmacKeyId,
                key = hmacKeyResponse.hmacKey
            )
            hmacKeyManager.saveDeviceInstanceId(deviceInstanceId)
            Log.d(TAG, "HMAC key saved: keyId=${hmacKeyResponse.hmacKeyId}")
            
            // 6. Complete registration
            val deviceIdHash = hmacKeyManager.generateDeviceIdHash(deviceInstanceId)
            Log.d(TAG, "Generated device ID hash: $deviceIdHash")
            
            val registrationResponse = apiService.registerDevice(
                deviceIdHash = deviceIdHash,
                deviceInfo = collectDeviceInfo()
            )
            
            Log.i(TAG, "Device registration succeeded: sessionId=${registrationResponse.sessionId}")
            
            return@withContext RegistrationResult(
                success = true,
                deviceIdHash = deviceIdHash,
                sessionId = registrationResponse.sessionId,
                errorMessage = null
            )
            
        } catch (e: SecurityException) {
            Log.e(TAG, "Security verification failed", e)
            return@withContext RegistrationResult(
                success = false,
                deviceIdHash = "",
                sessionId = null,
                errorMessage = "Security verification failed: ${e.message}"
            )
        } catch (e: Exception) {
            Log.e(TAG, "Device registration failed", e)
            return@withContext RegistrationResult(
                success = false,
                deviceIdHash = "",
                sessionId = null,
                errorMessage = "Registration failed: ${e.message}"
            )
        }
    }
    
    /**
     * Returns whether the device has been registered.
     */
    suspend fun isRegistered(): Boolean = withContext(Dispatchers.IO) {
        try {
            val deviceInstanceId = hmacKeyManager.getDeviceInstanceId()
            val hmacKeyId = hmacKeyManager.getHmacKeyId()
            
            return@withContext deviceInstanceId != null && hmacKeyId != null
        } catch (e: Exception) {
            Log.e(TAG, "Failed to check registration status", e)
            return@withContext false
        }
    }
    
    /**
     * Re-register device (e.g., after key rotation or invalidation).
     */
    suspend fun reregister(): RegistrationResult = withContext(Dispatchers.IO) {
        try {
            Log.i(TAG, "Re-registering device")
            
            // Clear old keys and identifiers
            hmacKeyManager.clearAll()
            
            // Run a fresh registration flow
            return@withContext register()
        } catch (e: Exception) {
            Log.e(TAG, "Re-registration failed", e)
            return@withContext RegistrationResult(
                success = false,
                deviceIdHash = "",
                sessionId = null,
                errorMessage = "Re-registration failed: ${e.message}"
            )
        }
    }
    
    /**
     * Collect device info for registration.
     */
    private fun collectDeviceInfo(): DeviceInfo {
        return DeviceInfo(
            manufacturer = android.os.Build.MANUFACTURER,
            model = android.os.Build.MODEL,
            androidVersion = android.os.Build.VERSION.RELEASE,
            apiLevel = android.os.Build.VERSION.SDK_INT,
            appVersion = BuildConfig.VERSION_NAME,
            appVersionCode = BuildConfig.VERSION_CODE
        )
    }
    
    /**
     * Calculate public key fingerprint.
     */
    fun calculateFingerprint(publicKey: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val hash = digest.digest(publicKey)
        return hash.joinToString("") { "%02x".format(it) }
    }
    
    data class DeviceInfo(
        val manufacturer: String,
        val model: String,
        val androidVersion: String,
        val apiLevel: Int,
        val appVersion: String,
        val appVersionCode: Int
    )
}
