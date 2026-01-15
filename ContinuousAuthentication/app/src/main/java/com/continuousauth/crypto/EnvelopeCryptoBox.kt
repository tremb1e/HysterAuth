package com.continuousauth.crypto

import android.content.Context
import android.util.Base64
import android.util.Log
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import com.google.crypto.tink.Aead
import com.google.crypto.tink.HybridEncrypt
import com.google.crypto.tink.KeyTemplate
import com.google.crypto.tink.KeysetHandle
import com.google.crypto.tink.StreamingAead
import com.google.crypto.tink.aead.AeadConfig
import com.google.crypto.tink.aead.AeadKeyTemplates
import com.google.crypto.tink.config.TinkConfig
import com.google.crypto.tink.hybrid.HybridConfig
import com.google.crypto.tink.integration.android.AndroidKeysetManager
import com.google.crypto.tink.streamingaead.StreamingAeadConfig
import com.google.crypto.tink.streamingaead.StreamingAeadKeyTemplates
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Envelope encryption implementation.
 *
 * Uses Google Tink to implement envelope (hybrid) encryption.
 */
@Singleton
class EnvelopeCryptoBox @Inject constructor(
    @ApplicationContext private val context: Context
) : CryptoBox {
    
    companion object {
        private const val TAG = "EnvelopeCryptoBox"
        private const val PREFS_FILE = "crypto_secure_prefs"
        private const val KEY_DEVICE_ID = "device_instance_id"
        private const val KEY_HMAC_KEY = "hmac_key"
        private const val KEY_SERVER_PUBLIC_KEY = "server_public_key"
        private const val KEY_DEK_KEY_ID = "dek_key_id"
        private const val KEY_PACKET_SEQ_NO = "packet_seq_no"
        private const val KEY_ROTATION_COUNT = "key_rotation_count"
        private const val KEY_CREATION_TIME = "key_creation_time"
        private const val KEY_FAILED_COUNT = "crypto_failed_count"
        private const val KEY_SESSION_DEK = "session_dek"
        private const val KEY_DEK_EXPIRY_TIME = "dek_expiry_time"
        
        private const val KEYSET_NAME = "ca_master_keyset"
        private const val KEYSET_PREF_NAME = "ca_master_keyset_prefs"
        
        private const val MAX_FAILURE_COUNT = 5  // Circuit breaker after 5 consecutive failures
        private const val CHUNK_SIZE = 256 * 1024  // 256KB chunks for streaming
        private const val DEK_LIFETIME_MS = 3600_000L  // 1 hour DEK lifetime
    }
    
    // Encrypted SharedPreferences for sensitive state
    private val encryptedPrefs by lazy {
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
            
        EncryptedSharedPreferences.create(
            context,
            PREFS_FILE,
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
    }
    
    // Tink key manager
    private var keysetManager: AndroidKeysetManager? = null
    private var aeadPrimitive: Aead? = null
    private var streamingAeadKeyset: KeysetHandle? = null
    private var streamingAead: StreamingAead? = null
    
    // Server public key (used to encrypt DEKs)
    private var serverPublicKey: HybridEncrypt? = null
    
    // Synchronization lock
    private val mutex = Mutex()
    
    // Packet sequence number (monotonic, anti-replay)
    private val packetSeqNo = AtomicLong(0)
    
    // Key rotation count
    private val rotationCount = AtomicInteger(0)
    
    // Crypto failure count (circuit breaker)
    private val failureCount = AtomicInteger(0)
    
    // Device instance id (generated on first install)
    private var deviceInstanceId: String? = null
    
    // HMAC key (used to derive hashes)
    private var hmacKey: ByteArray? = null
    
    // Session DEK (rotated hourly)
    private var sessionDEK: ByteArray? = null
    private var dekExpiryTime: Long = 0L
    private var encryptedDEK: ByteArray? = null
    
    override suspend fun initialize(): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                mutex.withLock {
                    // 1) Initialize Tink
                    TinkConfig.register()
                    AeadConfig.register()
                    StreamingAeadConfig.register()
                    HybridConfig.register()
                    
                    // 2) Initialize or restore device id
                    deviceInstanceId = encryptedPrefs.getString(KEY_DEVICE_ID, null)
                    if (deviceInstanceId == null) {
                        deviceInstanceId = UUID.randomUUID().toString()
                        encryptedPrefs.edit()
                            .putString(KEY_DEVICE_ID, deviceInstanceId)
                            .apply()
                        Log.i(TAG, "Generated new device instance ID")
                    }
                    
                    // 3) Initialize or restore HMAC key
                    val hmacKeyStr = encryptedPrefs.getString(KEY_HMAC_KEY, null)
                    hmacKey = if (hmacKeyStr == null) {
                        val newKey = generateRandomBytes(32)  // 256-bit HMAC key
                        encryptedPrefs.edit()
                            .putString(KEY_HMAC_KEY, Base64.encodeToString(newKey, Base64.NO_WRAP))
                            .apply()
                        Log.i(TAG, "Generated new HMAC key")
                        newKey
                    } else {
                        Base64.decode(hmacKeyStr, Base64.NO_WRAP)
                    }
                    
                    // 4) Initialize AEAD keyset (small payloads)
                    keysetManager = AndroidKeysetManager.Builder()
                        .withSharedPref(context, KEYSET_NAME, KEYSET_PREF_NAME)
                        .withKeyTemplate(AeadKeyTemplates.AES256_GCM)
                        .withMasterKeyUri("android-keystore://ca_master_key")
                        .build()
                    
                    aeadPrimitive = keysetManager?.keysetHandle?.getPrimitive(Aead::class.java)
                    
                    // 5) Initialize StreamingAEAD keyset (large payloads)
                    streamingAeadKeyset = KeysetHandle.generateNew(
                        StreamingAeadKeyTemplates.AES256_GCM_HKDF_4KB
                    )
                    streamingAead = streamingAeadKeyset?.getPrimitive(StreamingAead::class.java)
                    
                    // 6) Restore packet sequence number
                    packetSeqNo.set(encryptedPrefs.getLong(KEY_PACKET_SEQ_NO, 0))
                    
                    // 7) Restore rotation count
                    rotationCount.set(encryptedPrefs.getInt(KEY_ROTATION_COUNT, 0))
                    
                    // 8) Restore session DEK if present and not expired
                    val savedDEKExpiry = encryptedPrefs.getLong(KEY_DEK_EXPIRY_TIME, 0)
                    if (savedDEKExpiry > System.currentTimeMillis()) {
                        val savedDEK = encryptedPrefs.getString(KEY_SESSION_DEK, null)
                        if (savedDEK != null) {
                            sessionDEK = Base64.decode(savedDEK, Base64.NO_WRAP)
                            dekExpiryTime = savedDEKExpiry
                            Log.i(TAG, "Restored session DEK; valid until: ${java.util.Date(dekExpiryTime)}")
                        }
                    }
                    
                    // 9) Reset failure counter
                    failureCount.set(0)
                    encryptedPrefs.edit().putInt(KEY_FAILED_COUNT, 0).apply()
                    
                    // 10) TODO: Fetch server public key (requires server endpoint)
                    // serverPublicKey = fetchServerPublicKey()
                    
                    Log.i(TAG, "EnvelopeCryptoBox initialized successfully")
                    true
                }
            } catch (e: Exception) {
                Log.e(TAG, "EnvelopeCryptoBox initialization failed", e)
                false
            }
        }
    }
    
    override suspend fun encrypt(
        plaintext: ByteArray,
        associatedData: ByteArray?
    ): ByteArray? {
        return withContext(Dispatchers.IO) {
            try {
                mutex.withLock {
                    // Check circuit breaker
                    if (failureCount.get() >= MAX_FAILURE_COUNT) {
                        Log.e(TAG, "Too many encryption failures; circuit breaker triggered")
                        return@withContext null
                    }
                    
                    val result = if (plaintext.size > CHUNK_SIZE) {
                        // Large payloads: streaming encryption
                        encryptStreaming(plaintext, associatedData)
                    } else {
                        // Small payloads: AEAD
                        aeadPrimitive?.encrypt(plaintext, associatedData)
                    }
                    
                    // Reset failure counter
                    if (result != null) {
                        failureCount.set(0)
                        encryptedPrefs.edit().putInt(KEY_FAILED_COUNT, 0).apply()
                    }
                    
                    result
                }
            } catch (e: Exception) {
                Log.e(TAG, "Encryption failed", e)
                handleCryptoFailure()
                null
            }
        }
    }
    
    override suspend fun decrypt(
        ciphertext: ByteArray,
        associatedData: ByteArray?
    ): ByteArray? {
        return withContext(Dispatchers.IO) {
            try {
                mutex.withLock {
                    // Check circuit breaker
                    if (failureCount.get() >= MAX_FAILURE_COUNT) {
                        Log.e(TAG, "Too many decryption failures; circuit breaker triggered")
                        return@withContext null
                    }
                    
                    val result = if (ciphertext.size > CHUNK_SIZE + 100) {  // Account for encryption overhead
                        // Large payloads: streaming decryption
                        decryptStreaming(ciphertext, associatedData)
                    } else {
                        // Small payloads: AEAD
                        aeadPrimitive?.decrypt(ciphertext, associatedData)
                    }
                    
                    // Reset failure counter
                    if (result != null) {
                        failureCount.set(0)
                        encryptedPrefs.edit().putInt(KEY_FAILED_COUNT, 0).apply()
                    }
                    
                    result
                }
            } catch (e: Exception) {
                Log.e(TAG, "Decryption failed", e)
                handleCryptoFailure()
                null
            }
        }
    }
    
    override fun getKeyInfo(): CryptoKeyInfo {
        return CryptoKeyInfo(
            masterKeyExists = aeadPrimitive != null,
            sessionKeyActive = streamingAead != null,
            keysetProvider = "Android Keystore (Tink)",
            encryptionAlgorithm = "AES-256-GCM with StreamingAEAD",
            keyCreationTime = encryptedPrefs.getLong(KEY_CREATION_TIME, System.currentTimeMillis()),
            keyRotationCount = rotationCount.get().toLong()
        )
    }
    
    override fun destroy() {
        // Persist current state
        encryptedPrefs.edit()
            .putLong(KEY_PACKET_SEQ_NO, packetSeqNo.get())
            .putInt(KEY_ROTATION_COUNT, rotationCount.get())
            .apply()
        
        // Clear in-memory secrets
        hmacKey?.fill(0)
        hmacKey = null
        
        Log.i(TAG, "EnvelopeCryptoBox destroyed")
    }
    
    override suspend fun rotateSessionKey(): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                mutex.withLock {
                    // 1) Rotate session DEK (force new value)
                    sessionDEK = generateRandomBytes(32)
                    dekExpiryTime = System.currentTimeMillis() + DEK_LIFETIME_MS
                    encryptedDEK = encryptDEKWithServerKey(sessionDEK!!)
                    
                    // 2) Rotate AEAD key
                    val keyTemplate = com.google.crypto.tink.aead.AeadKeyTemplates.AES256_GCM
                    keysetManager?.let { manager ->
                        manager.add(keyTemplate)
                        val keyId = manager.keysetHandle.keysetInfo.keyInfoList.last().keyId
                        manager.setPrimary(keyId)
                    }
                    aeadPrimitive = keysetManager?.keysetHandle?.getPrimitive(Aead::class.java)
                    
                    // 3) Generate new StreamingAEAD key
                    streamingAeadKeyset = KeysetHandle.generateNew(
                        StreamingAeadKeyTemplates.AES256_GCM_HKDF_4KB
                    )
                    streamingAead = streamingAeadKeyset?.getPrimitive(StreamingAead::class.java)
                    
                    // 4) Increment rotation counter
                    val newCount = rotationCount.incrementAndGet()
                    
                    // 5) Persist updated key metadata
                    encryptedPrefs.edit()
                        .putString(KEY_SESSION_DEK, Base64.encodeToString(sessionDEK, Base64.NO_WRAP))
                        .putLong(KEY_DEK_EXPIRY_TIME, dekExpiryTime)
                        .putString(KEY_DEK_KEY_ID, generateDEKKeyId())
                        .putInt(KEY_ROTATION_COUNT, newCount)
                        .putLong(KEY_CREATION_TIME, System.currentTimeMillis())
                        .apply()
                    
                    Log.i(
                        TAG,
                        "Key rotation succeeded; rotationCount: $newCount, new DEK valid until: ${java.util.Date(dekExpiryTime)}"
                    )
                    true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Key rotation failed", e)
                false
            }
        }
    }
    
    /**
     * Get next packet sequence number.
     */
    fun getNextPacketSeqNo(): Long {
        val seqNo = packetSeqNo.incrementAndGet()
        // Persist periodically
        if (seqNo % 100 == 0L) {
            encryptedPrefs.edit().putLong(KEY_PACKET_SEQ_NO, seqNo).apply()
        }
        return seqNo
    }
    
    /**
     * Get DEK key id.
     */
    fun getDekKeyId(): String {
        return encryptedPrefs.getString(KEY_DEK_KEY_ID, null) 
            ?: "DEK_${deviceInstanceId}_${System.currentTimeMillis()}"
    }
    
    /**
     * Get or create a session DEK (rotated hourly).
     */
    fun getOrCreateSessionDEK(): ByteArray {
        val now = System.currentTimeMillis()
        
        // Check if a new DEK is needed
        if (sessionDEK == null || now > dekExpiryTime) {
            synchronized(this) {
                // Double-check
                if (sessionDEK == null || now > dekExpiryTime) {
                    // Generate new 256-bit DEK
                    sessionDEK = generateRandomBytes(32)
                    dekExpiryTime = now + DEK_LIFETIME_MS
                    
                    // Encrypt DEK with the server public key (TODO: wire up real server key).
                    encryptedDEK = encryptDEKWithServerKey(sessionDEK!!)
                    
                    // Persist to encrypted storage
                    encryptedPrefs.edit()
                        .putString(KEY_SESSION_DEK, Base64.encodeToString(sessionDEK, Base64.NO_WRAP))
                        .putLong(KEY_DEK_EXPIRY_TIME, dekExpiryTime)
                        .putString(KEY_DEK_KEY_ID, generateDEKKeyId())
                        .apply()
                    
                    Log.i(TAG, "Generated new session DEK; valid until: ${java.util.Date(dekExpiryTime)}")
                }
            }
        }
        
        return sessionDEK!!
    }
    
    /**
     * Get encrypted DEK (sent to the server).
     */
    fun getEncryptedDEK(): ByteArray? {
        getOrCreateSessionDEK() // Ensure DEK exists
        return encryptedDEK
    }
    
    /**
     * Generate DEK key id.
     */
    private fun generateDEKKeyId(): String {
        return "DEK_${deviceInstanceId}_${System.currentTimeMillis()}"
    }
    
    /**
     * Encrypt DEK using the server public key.
     */
    private fun encryptDEKWithServerKey(dek: ByteArray): ByteArray {
        // TODO: Encrypt using the real server public key.
        // return serverPublicKey?.encrypt(dek, null) ?: dek
        
        // Fallback when server key encryption is not wired: return a base64-encoded DEK.
        return Base64.encode(dek, Base64.NO_WRAP)
    }
    
    /**
     * Generate and encrypt a DEK (data encryption key).
     *
     * A real implementation requires the server public key.
     */
    fun generateAndEncryptDEK(): ByteArray {
        return getEncryptedDEK() ?: Base64.encode(generateRandomBytes(32), Base64.NO_WRAP)
    }
    
    /**
     * Get HMAC hash of device id.
     */
    fun getDeviceIdHash(): String {
        return computeHmac(deviceInstanceId ?: "", getDekKeyId())
    }
    
    /**
     * Get HMAC hash of app package name.
     */
    fun getAppPackageHash(packageName: String): String {
        return computeHmac(packageName, getDekKeyId())
    }
    
    /**
     * Get HMAC hash of user id.
     */
    fun getUserIdHash(userId: String): String {
        return computeHmac(userId, getDekKeyId())
    }
    
    /**
     * Compute an HMAC hash (to avoid leaking sensitive values).
     */
    private fun computeHmac(data: String, keyId: String): String {
        return try {
            val mac = Mac.getInstance("HmacSHA256")
            val key = SecretKeySpec(hmacKey ?: generateRandomBytes(32), "HmacSHA256")
            mac.init(key)
            val input = "$keyId:$data".toByteArray(StandardCharsets.UTF_8)
            val hash = mac.doFinal(input)
            Base64.encodeToString(hash, Base64.NO_WRAP)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to compute HMAC", e)
            ""
        }
    }
    
    /**
     * Compute HMAC.
     *
     * @param data Data to hash.
     * @param context Context label (used to separate different hash domains).
     */
    private fun computeHMAC(data: String, context: String): String {
        return try {
            val mac = Mac.getInstance("HmacSHA256")
            val keyWithContext = "${context}_${Base64.encodeToString(hmacKey, Base64.NO_WRAP)}"
            val secretKey = SecretKeySpec(
                keyWithContext.toByteArray(StandardCharsets.UTF_8),
                "HmacSHA256"
            )
            mac.init(secretKey)
            val hash = mac.doFinal(data.toByteArray(StandardCharsets.UTF_8))
            Base64.encodeToString(hash, Base64.NO_WRAP or Base64.URL_SAFE)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to compute HMAC", e)
            // Fall back to SHA-256
            val digest = MessageDigest.getInstance("SHA-256")
            val hash = digest.digest("${context}_${data}".toByteArray(StandardCharsets.UTF_8))
            Base64.encodeToString(hash, Base64.NO_WRAP or Base64.URL_SAFE)
        }
    }
    
    /**
     * Streaming encryption (large payloads).
     */
    private suspend fun encryptStreaming(
        plaintext: ByteArray,
        associatedData: ByteArray?
    ): ByteArray {
        return withContext(Dispatchers.IO) {
            val outputStream = ByteArrayOutputStream()
            streamingAead?.newEncryptingStream(outputStream, associatedData)?.use { encStream ->
                encStream.write(plaintext)
            }
            outputStream.toByteArray()
        }
    }
    
    /**
     * Streaming decryption (large payloads).
     */
    private suspend fun decryptStreaming(
        ciphertext: ByteArray,
        associatedData: ByteArray?
    ): ByteArray {
        return withContext(Dispatchers.IO) {
            val inputStream = ByteArrayInputStream(ciphertext)
            val outputStream = ByteArrayOutputStream()
            
            streamingAead?.newDecryptingStream(inputStream, associatedData)?.use { decStream ->
                val buffer = ByteArray(8192)
                var bytesRead: Int
                while (decStream.read(buffer).also { bytesRead = it } != -1) {
                    outputStream.write(buffer, 0, bytesRead)
                }
            }
            outputStream.toByteArray()
        }
    }
    
    /**
     * Handle crypto failure.
     */
    private fun handleCryptoFailure() {
        val count = failureCount.incrementAndGet()
        encryptedPrefs.edit().putInt(KEY_FAILED_COUNT, count).apply()
        
        if (count >= MAX_FAILURE_COUNT) {
            Log.e(TAG, "Circuit breaker triggered: crypto failures reached limit ($MAX_FAILURE_COUNT)")
            // TODO: Notify upper layers to stop collection
        }
    }
    
    /**
     * Generate random bytes.
     */
    private fun generateRandomBytes(size: Int): ByteArray {
        return ByteArray(size).also { bytes ->
            java.security.SecureRandom().nextBytes(bytes)
        }
    }
    
    /**
     * Check whether crypto is disabled due to excessive failures.
     */
    fun isCryptoDisabled(): Boolean {
        return failureCount.get() >= MAX_FAILURE_COUNT
    }
    
    /**
     * Reset failure count (use only in exceptional cases).
     */
    fun resetFailureCount() {
        failureCount.set(0)
        encryptedPrefs.edit().putInt(KEY_FAILED_COUNT, 0).apply()
        Log.w(TAG, "Failure counter reset")
    }
}
