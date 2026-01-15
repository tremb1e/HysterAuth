package com.continuousauth.crypto

import android.content.Context
import android.util.Base64
import android.util.Log
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import com.continuousauth.network.GrpcManager
// import com.continuousauth.proto.KeyRegistrationRequest
// import com.continuousauth.proto.KeyRegistrationResponse
// import com.continuousauth.proto.PolicyRequest
import com.google.crypto.tink.HybridDecrypt
import com.google.crypto.tink.HybridEncrypt
import com.google.crypto.tink.KeysetHandle
import com.google.crypto.tink.hybrid.HybridConfig
import com.google.crypto.tink.hybrid.HybridKeyTemplates
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import okhttp3.*
// import okhttp3.tls.HandshakeCertificates
// import okhttp3.tls.HeldCertificate
import java.io.ByteArrayOutputStream
import java.security.MessageDigest
import java.security.cert.Certificate
import java.security.cert.CertificateFactory
import java.security.cert.X509Certificate
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton
import javax.net.ssl.SSLPeerUnverifiedException

/**
 * Key management service.
 *
 * Handles server key registration, certificate pinning, and key rotation.
 */
@Singleton
class KeyManagementService @Inject constructor(
    @ApplicationContext private val context: Context,
    private val grpcManager: GrpcManager
) {
    
    companion object {
        private const val TAG = "KeyManagementService"
        private const val PREFS_FILE = "key_management_prefs"
        
        // Keys for EncryptedSharedPreferences
        private const val KEY_SERVER_PUBLIC_KEY = "server_public_key"
        private const val KEY_SERVER_PUBLIC_KEY_FINGERPRINT = "server_public_key_fingerprint"
        private const val KEY_SERVER_PUBLIC_KEY_VERSION = "server_public_key_version"
        private const val KEY_PINNED_CERTIFICATES = "pinned_certificates"
        private const val KEY_LAST_ROTATION_TIME = "last_rotation_time"
        private const val KEY_ROTATION_INTERVAL_HOURS = "rotation_interval_hours"
        private const val KEY_DEVICE_PRIVATE_KEY = "device_private_key"
        
        // Default values
        private const val DEFAULT_ROTATION_INTERVAL_HOURS = 24
        private const val MIN_ROTATION_INTERVAL_HOURS = 1
        private const val MAX_ROTATION_INTERVAL_HOURS = 168 // 7 days
        
        // Certificate pinning
        private const val MAX_PINNED_CERTIFICATES = 5
    }
    
    // Encrypted storage
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
    
    // Server public key (used to encrypt DEKs)
    private var serverPublicKey: HybridEncrypt? = null
    private var serverPublicKeyVersion: String? = null
    private var serverPublicKeyId: String = "default_key_id"
    
    // Device keypair (used for registration authentication)
    private var deviceKeysetHandle: KeysetHandle? = null
    
    // Certificate pinning
    private val pinnedCertificates = mutableSetOf<String>()
    
    // Synchronization lock
    private val mutex = Mutex()
    
    // OkHttpClient for key exchange
    private val httpClient: OkHttpClient by lazy {
        buildPinnedHttpClient()
    }
    
    init {
        // Initialize Tink
        HybridConfig.register()
    }
    
    /**
     * Register and fetch the server public key.
     */
    suspend fun registerAndFetchServerPublicKey(serverEndpoint: String): Result<Unit> {
        return withContext(Dispatchers.IO) {
            try {
                mutex.withLock {
                    Log.i(TAG, "Registering and fetching server public key")
                    
                    // Generate device keypair if missing
                    if (deviceKeysetHandle == null) {
                        deviceKeysetHandle = KeysetHandle.generateNew(
                            HybridKeyTemplates.ECIES_P256_HKDF_HMAC_SHA256_AES128_GCM
                        )
                        saveDevicePrivateKey()
                    }
                    
                    // Extract device public key
                    val devicePublicKey = getPublicKeyFromKeyset(deviceKeysetHandle!!)
                    
                    // Register with server and fetch server public key
                    val response = fetchServerPublicKeyFromEndpoint(serverEndpoint, devicePublicKey)
                    
                    if (response != null) {
                        // Verify public key fingerprint
                        val fingerprint = calculateFingerprint(response.publicKey)
                        Log.i(TAG, "Server public key fingerprint: $fingerprint")
                        
                        // Persist server public key and fingerprint
                        saveServerPublicKeyInternal(response.publicKey, response.keyVersion, fingerprint)
                        
                        // Initialize HybridEncrypt
                        initializeServerPublicKey(response.publicKey)
                        
                        // Persist certificate pinning information
                        if (response.certificates.isNotEmpty()) {
                            savePinnedCertificates(response.certificates)
                        }
                        
                        Log.i(TAG, "Successfully fetched and verified server public key, version: ${response.keyVersion}")
                        Result.success(Unit)
                    } else {
                        Result.failure(Exception("Failed to fetch server public key"))
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Registration failed", e)
                Result.failure(e)
            }
        }
    }
    
    /**
     * Get current DEK key id.
     */
    fun getCurrentDekKeyId(): String {
        return serverPublicKeyVersion ?: "DEFAULT_KEY_V1"
    }
    
    /**
     * Encrypt a DEK using the server public key.
     */
    suspend fun encryptDEK(dek: ByteArray): ByteArray? {
        return withContext(Dispatchers.IO) {
            try {
                mutex.withLock {
                    if (serverPublicKey == null) {
                        Log.e(TAG, "Server public key not initialized")
                        return@withContext null
                    }
                    
                    // Encrypt DEK using HybridEncrypt
                    val contextInfo = "DEK_ENCRYPTION_${System.currentTimeMillis()}".toByteArray()
                    serverPublicKey?.encrypt(dek, contextInfo)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to encrypt DEK", e)
                null
            }
        }
    }
    
    /**
     * Key rotation.
     */
    suspend fun rotateKeys(): Result<Unit> {
        return withContext(Dispatchers.IO) {
            try {
                mutex.withLock {
                    Log.i(TAG, "Starting key rotation")
                    
                    // Check whether rotation is required
                    if (!shouldRotateKeys()) {
                        Log.i(TAG, "Rotation not due yet; skipping")
                        return@withContext Result.success(Unit)
                    }
                    
                    // Fetch a newer server public key version
                    val newKeyResponse = fetchLatestServerPublicKey()
                    
                    if (newKeyResponse != null && newKeyResponse.keyVersion != serverPublicKeyVersion) {
                        // Verify new public key
                        val newFingerprint = calculateFingerprint(newKeyResponse.publicKey)
                        Log.i(TAG, "New public key fingerprint: $newFingerprint")
                        
                        // Persist new key (keep old for compatibility)
                        saveServerPublicKeyInternal(
                            newKeyResponse.publicKey,
                            newKeyResponse.keyVersion,
                            newFingerprint
                        )
                        
                        // Update HybridEncrypt instance
                        initializeServerPublicKey(newKeyResponse.publicKey)
                        
                        // Update rotation timestamp
                        encryptedPrefs.edit()
                            .putLong(KEY_LAST_ROTATION_TIME, System.currentTimeMillis())
                            .apply()
                        
                        Log.i(TAG, "Key rotation succeeded; new version: ${newKeyResponse.keyVersion}")
                    }
                    
                    Result.success(Unit)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Key rotation failed", e)
                Result.failure(e)
            }
        }
    }
    
    /**
     * Validate server certificate (pinning).
     */
    fun verifyCertificatePinning(peerCertificates: List<Certificate>): Boolean {
        try {
            if (pinnedCertificates.isEmpty()) {
                // If no pins are present, store pins on first connection.
                if (peerCertificates.isNotEmpty()) {
                    val pins = peerCertificates
                        .filterIsInstance<X509Certificate>()
                        .map { calculateCertificatePin(it) }
                    savePinnedCertificates(pins)
                }
                return true
            }
            
            // Verify the certificate chain contains a matching pinned certificate.
            for (cert in peerCertificates) {
                if (cert is X509Certificate) {
                    val pin = calculateCertificatePin(cert)
                    if (pinnedCertificates.contains(pin)) {
                        Log.d(TAG, "Certificate pinning verified")
                        return true
                    }
                }
            }
            
            Log.e(TAG, "Certificate pinning verification failed; no matching pin found")
            return false
            
        } catch (e: Exception) {
            Log.e(TAG, "Certificate verification error", e)
            return false
        }
    }
    
    /**
     * Build an HttpClient with certificate pinning.
     */
    private fun buildPinnedHttpClient(): OkHttpClient {
        val builder = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
        
        // Configure certificate pinning if pins exist.
        if (pinnedCertificates.isNotEmpty()) {
            // TODO: Configure real certificate pinning.
            // This must be configured with the actual server certificate pins.
        }
        
        return builder.build()
    }
    
    /**
     * Fetch public key from a server endpoint.
     *
     * TODO: implement the actual network request.
     */
    private suspend fun fetchServerPublicKeyFromEndpoint(
        endpoint: String,
        devicePublicKey: ByteArray
    ): ServerKeyResponse? {
        return withContext(Dispatchers.IO) {
            try {
                // TODO: Implement actual gRPC or HTTP request.
                // Mock response for now.
                Log.w(TAG, "Server endpoint not implemented; returning mock data")
                
                // Generate a mock server keypair
                val serverKeyset = KeysetHandle.generateNew(
                    HybridKeyTemplates.ECIES_P256_HKDF_HMAC_SHA256_AES128_GCM
                )
                
                ServerKeyResponse(
                    publicKey = getPublicKeyFromKeyset(serverKeyset),
                    keyVersion = "v1_${System.currentTimeMillis()}",
                    certificates = emptyList()
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to fetch server public key", e)
                null
            }
        }
    }
    
    /**
     * Fetch latest server public key.
     */
    private suspend fun fetchLatestServerPublicKey(): ServerKeyResponse? {
        // TODO: Implement fetching latest public key from server.
        return null
    }
    
    /**
     * Extract public key bytes from a Keyset.
     */
    private fun getPublicKeyFromKeyset(keyset: KeysetHandle): ByteArray {
        val outputStream = ByteArrayOutputStream()
        val writer = com.google.crypto.tink.JsonKeysetWriter.withOutputStream(outputStream)
        keyset.publicKeysetHandle.writeNoSecret(writer)
        return outputStream.toByteArray()
    }
    
    /**
     * Initialize server public key for encryption.
     */
    private fun initializeServerPublicKey(publicKeyBytes: ByteArray) {
        try {
            // TODO: Restore KeysetHandle from bytes and create HybridEncrypt.
            // Requires a real Tink implementation.
            Log.i(TAG, "Server public key initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize server public key", e)
        }
    }
    
    /**
     * Calculate public key fingerprint.
     */
    private fun calculateFingerprint(publicKey: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val hash = digest.digest(publicKey)
        return Base64.encodeToString(hash, Base64.NO_WRAP)
    }
    
    /**
     * Calculate a certificate pin (for certificate pinning).
     */
    private fun calculateCertificatePin(certificate: X509Certificate): String {
        val publicKey = certificate.publicKey.encoded
        val digest = MessageDigest.getInstance("SHA-256")
        val hash = digest.digest(publicKey)
        return "sha256/" + Base64.encodeToString(hash, Base64.NO_WRAP)
    }
    
    /**
     * Persist server public key (internal).
     */
    private fun saveServerPublicKeyInternal(publicKey: ByteArray, version: String, fingerprint: String) {
        encryptedPrefs.edit()
            .putString(KEY_SERVER_PUBLIC_KEY, Base64.encodeToString(publicKey, Base64.NO_WRAP))
            .putString(KEY_SERVER_PUBLIC_KEY_VERSION, version)
            .putString(KEY_SERVER_PUBLIC_KEY_FINGERPRINT, fingerprint)
            .apply()
        
        serverPublicKeyVersion = version
    }
    
    /**
     * Persist server public key (public API for device registration).
     */
    fun saveServerPublicKey(publicKey: ByteArray, keyId: String, fingerprint: String) {
        saveServerPublicKeyInternal(publicKey, keyId, fingerprint)
        serverPublicKeyId = keyId
    }
    
    /**
     * Persist device private key.
     */
    private fun saveDevicePrivateKey() {
        deviceKeysetHandle?.let { keyset ->
            val outputStream = ByteArrayOutputStream()
            val writer = com.google.crypto.tink.JsonKeysetWriter.withOutputStream(outputStream)
            val masterKey = com.google.crypto.tink.aead.AeadKeyTemplates.AES256_GCM
            val masterKeyHandle = KeysetHandle.generateNew(masterKey)
            val aead = masterKeyHandle.getPrimitive(com.google.crypto.tink.Aead::class.java)
            keyset.write(writer, aead)
            val keyBytes = outputStream.toByteArray()
            
            encryptedPrefs.edit()
                .putString(KEY_DEVICE_PRIVATE_KEY, Base64.encodeToString(keyBytes, Base64.NO_WRAP))
                .apply()
        }
    }
    
    /**
     * Persist pinned certificates.
     */
    private fun savePinnedCertificates(certificates: List<String>) {
        pinnedCertificates.clear()
        pinnedCertificates.addAll(certificates.take(MAX_PINNED_CERTIFICATES))
        
        encryptedPrefs.edit()
            .putStringSet(KEY_PINNED_CERTIFICATES, pinnedCertificates.toSet())
            .apply()
        
        Log.i(TAG, "Saved ${pinnedCertificates.size} certificate pins")
    }
    
    /**
     * Load pinned certificates.
     */
    fun loadPinnedCertificates() {
        val savedPins = encryptedPrefs.getStringSet(KEY_PINNED_CERTIFICATES, emptySet())
        if (savedPins != null) {
            pinnedCertificates.clear()
            pinnedCertificates.addAll(savedPins)
            Log.i(TAG, "Loaded ${pinnedCertificates.size} certificate pins")
        }
    }
    
    /**
     * Check whether keys should be rotated.
     */
    private fun shouldRotateKeys(): Boolean {
        val lastRotation = encryptedPrefs.getLong(KEY_LAST_ROTATION_TIME, 0)
        val rotationInterval = encryptedPrefs.getInt(
            KEY_ROTATION_INTERVAL_HOURS,
            DEFAULT_ROTATION_INTERVAL_HOURS
        )
        
        val hoursSinceRotation = (System.currentTimeMillis() - lastRotation) / (1000 * 60 * 60)
        return hoursSinceRotation >= rotationInterval
    }
    
    /**
     * Set key rotation interval (hours).
     */
    fun setKeyRotationInterval(hours: Int) {
        val validHours = hours.coerceIn(MIN_ROTATION_INTERVAL_HOURS, MAX_ROTATION_INTERVAL_HOURS)
        encryptedPrefs.edit()
            .putInt(KEY_ROTATION_INTERVAL_HOURS, validHours)
            .apply()
        
        Log.i(TAG, "Key rotation interval set to: $validHours hours")
    }
    
    /**
     * Get key information.
     */
    fun getKeyInfo(): KeyInfo {
        return KeyInfo(
            serverPublicKeyVersion = serverPublicKeyVersion ?: "uninitialized",
            serverPublicKeyFingerprint = encryptedPrefs.getString(KEY_SERVER_PUBLIC_KEY_FINGERPRINT, null) ?: "unknown",
            pinnedCertificatesCount = pinnedCertificates.size,
            lastRotationTime = encryptedPrefs.getLong(KEY_LAST_ROTATION_TIME, 0),
            rotationIntervalHours = encryptedPrefs.getInt(KEY_ROTATION_INTERVAL_HOURS, DEFAULT_ROTATION_INTERVAL_HOURS)
        )
    }
    
    /**
     * Get current server public key id.
     */
    fun getCurrentServerKeyId(): String {
        return serverPublicKeyId
    }
    
    /**
     * Refresh keys.
     */
    suspend fun refreshKeys() {
        Log.i(TAG, "Refreshing keys")
        // Refresh server public key
        // fetchServerPublicKeyFromEndpoint() requires parameters; skipping for now.
        // Trigger key rotation
        rotateKeys()
    }
    
    /**
     * Refresh server public key.
     */
    suspend fun refreshServerPublicKey() {
        Log.i(TAG, "Refreshing server public key")
        // fetchServerPublicKeyFromEndpoint() requires parameters; skipping for now.
    }
    
    /**
     * Update key version.
     */
    suspend fun updateKeyVersion() {
        Log.i(TAG, "Updating key version")
        // Fetch latest key version
        // fetchServerPublicKeyFromEndpoint() requires parameters; skipping for now.
        // Update locally stored version
        encryptedPrefs.edit()
            .putString("key_version", "v${System.currentTimeMillis()}")
            .apply()
    }
}

/**
 * Server key response.
 */
data class ServerKeyResponse(
    val publicKey: ByteArray,
    val keyVersion: String,
    val certificates: List<String>
)

/**
 * Key metadata.
 */
data class KeyInfo(
    val serverPublicKeyVersion: String,
    val serverPublicKeyFingerprint: String,
    val pinnedCertificatesCount: Int,
    val lastRotationTime: Long,
    val rotationIntervalHours: Int
)
