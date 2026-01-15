package com.continuousauth.utils

import android.content.Context
import android.util.Log
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.charset.StandardCharsets
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import javax.inject.Inject
import javax.inject.Singleton

/**
 * HMAC key manager.
 *
 * Stores HMAC keys securely using EncryptedSharedPreferences and generates hashed identifiers
 * (e.g., device ID hash) to avoid leaking sensitive values.
 */
@Singleton
class HmacKeyManager @Inject constructor(
    @ApplicationContext private val context: Context
) {
    
    companion object {
        private const val TAG = "HmacKeyManager"
        private const val PREFS_NAME = "hmac_keys"
        private const val KEY_HMAC_KEY_ID = "hmac_key_id"
        private const val KEY_HMAC_KEY = "hmac_key"
        private const val KEY_DEVICE_INSTANCE_ID = "device_instance_id"
        private const val HMAC_ALGORITHM = "HmacSHA256"
    }
    
    private val masterKey by lazy {
        MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
    }
    
    private val encryptedPrefs by lazy {
        EncryptedSharedPreferences.create(
            context,
            PREFS_NAME,
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
    }
    
    /**
     * Save HMAC key.
     */
    fun saveHmacKey(keyId: String, key: String) {
        try {
            encryptedPrefs.edit()
                .putString(KEY_HMAC_KEY_ID, keyId)
                .putString(KEY_HMAC_KEY, key)
                .apply()
            Log.d(TAG, "HMAC key saved: keyId=$keyId")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save HMAC key", e)
            throw SecurityException("Failed to save HMAC key", e)
        }
    }
    
    /**
     * Save device instance ID.
     */
    fun saveDeviceInstanceId(deviceInstanceId: String) {
        try {
            encryptedPrefs.edit()
                .putString(KEY_DEVICE_INSTANCE_ID, deviceInstanceId)
                .apply()
            Log.d(TAG, "Device instance ID saved")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save device instance ID", e)
            throw SecurityException("Failed to save device instance ID", e)
        }
    }
    
    /**
     * Get HMAC key ID.
     */
    fun getHmacKeyId(): String? {
        return try {
            encryptedPrefs.getString(KEY_HMAC_KEY_ID, null)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get HMAC key ID", e)
            null
        }
    }
    
    /**
     * Get device instance ID.
     */
    fun getDeviceInstanceId(): String? {
        return try {
            encryptedPrefs.getString(KEY_DEVICE_INSTANCE_ID, null)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get device instance ID", e)
            null
        }
    }
    
    /**
     * Generate device ID hash.
     *
     * Computes a one-way identifier using HMAC(keyId, deviceInstanceId).
     */
    fun generateDeviceIdHash(deviceInstanceId: String? = null): String {
        try {
            val keyId = getHmacKeyId()
                ?: throw IllegalStateException("HMAC key ID not found")
            
            val key = encryptedPrefs.getString(KEY_HMAC_KEY, null)
                ?: throw IllegalStateException("HMAC key not found")
            
            val actualDeviceId = deviceInstanceId ?: getDeviceInstanceId()
                ?: throw IllegalStateException("Device instance ID not found")
            
            return computeHMAC(key, "$keyId:$actualDeviceId")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate device ID hash", e)
            throw SecurityException("Failed to generate device ID hash", e)
        }
    }
    
    /**
     * Generate a hash for an app package name.
     *
     * Used to protect privacy when recording foreground app information.
     */
    fun generateAppHash(packageName: String): String {
        try {
            val keyId = getHmacKeyId()
                ?: throw IllegalStateException("HMAC key ID not found")
            
            val key = encryptedPrefs.getString(KEY_HMAC_KEY, null)
                ?: throw IllegalStateException("HMAC key not found")
            
            return computeHMAC(key, "$keyId:$packageName")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate app hash", e)
            // Return empty string to avoid breaking data collection.
            return ""
        }
    }
    
    /**
     * Generate user ID hash.
     */
    fun generateUserIdHash(userId: String): String {
        try {
            val keyId = getHmacKeyId()
                ?: throw IllegalStateException("HMAC key ID not found")
            
            val key = encryptedPrefs.getString(KEY_HMAC_KEY, null)
                ?: throw IllegalStateException("HMAC key not found")
            
            return computeHMAC(key, "$keyId:$userId")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate user ID hash", e)
            throw SecurityException("Failed to generate user ID hash", e)
        }
    }
    
    /**
     * Compute HMAC.
     */
    private fun computeHMAC(key: String, data: String): String {
        return try {
            val secretKey = SecretKeySpec(
                key.toByteArray(StandardCharsets.UTF_8),
                HMAC_ALGORITHM
            )
            
            val mac = Mac.getInstance(HMAC_ALGORITHM)
            mac.init(secretKey)
            
            val hmacBytes = mac.doFinal(data.toByteArray(StandardCharsets.UTF_8))
            
            // Convert to hex string
            hmacBytes.joinToString("") { "%02x".format(it) }
        } catch (e: Exception) {
            Log.e(TAG, "HMAC computation failed", e)
            throw SecurityException("HMAC computation failed", e)
        }
    }
    
    /**
     * Verify HMAC.
     */
    fun verifyHMAC(data: String, expectedHmac: String): Boolean {
        return try {
            val keyId = getHmacKeyId()
                ?: throw IllegalStateException("HMAC key ID not found")
            
            val key = encryptedPrefs.getString(KEY_HMAC_KEY, null)
                ?: throw IllegalStateException("HMAC key not found")
            
            val computedHmac = computeHMAC(key, data)
            
            // Constant-time comparison to mitigate timing attacks.
            constantTimeEquals(computedHmac, expectedHmac)
        } catch (e: Exception) {
            Log.e(TAG, "HMAC verification failed", e)
            false
        }
    }
    
    /**
     * Constant-time string comparison.
     */
    private fun constantTimeEquals(a: String, b: String): Boolean {
        if (a.length != b.length) {
            return false
        }
        
        var result = 0
        for (i in a.indices) {
            result = result or (a[i].code xor b[i].code)
        }
        return result == 0
    }
    
    /**
     * Clear all stored keys and identifiers.
     *
     * Used for device logout or re-registration flows.
     */
    fun clearAll() {
        try {
            encryptedPrefs.edit()
                .clear()
                .apply()
            Log.i(TAG, "All HMAC keys and identifiers cleared")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to clear keys", e)
            throw SecurityException("Failed to clear keys", e)
        }
    }
    
    /**
     * Rotate HMAC key.
     */
    suspend fun rotateHmacKey(newKeyId: String, newKey: String) = withContext(Dispatchers.IO) {
        try {
            Log.i(TAG, "Starting HMAC key rotation")
            
            // Save new key
            saveHmacKey(newKeyId, newKey)
            
            // Regenerate device ID hash
            val deviceInstanceId = getDeviceInstanceId()
            if (deviceInstanceId != null) {
                val newDeviceIdHash = generateDeviceIdHash(deviceInstanceId)
                Log.d(TAG, "New device ID hash generated: $newDeviceIdHash")
            }
            
            Log.i(TAG, "HMAC key rotation completed")
        } catch (e: Exception) {
            Log.e(TAG, "HMAC key rotation failed", e)
            throw SecurityException("HMAC key rotation failed", e)
        }
    }
    
    /**
     * Returns whether key rotation is needed.
     */
    fun shouldRotateKey(): Boolean {
        // TODO: Implement key rotation policy (time-based and/or usage-based).
        return false
    }
}
