package com.continuousauth.crypto

/**
 * Crypto box interface.
 *
 * Provides AEAD encryption/decryption primitives.
 */
interface CryptoBox {
    
    /**
     * Initialize crypto subsystem.
     */
    suspend fun initialize(): Boolean
    
    /**
     * Encrypt data.
     *
     * @param plaintext Plaintext bytes.
     * @param associatedData Associated data (AAD) for additional integrity binding.
     * @return Ciphertext, or null on failure.
     */
    suspend fun encrypt(plaintext: ByteArray, associatedData: ByteArray? = null): ByteArray?
    
    /**
     * Decrypt data.
     *
     * @param ciphertext Ciphertext bytes.
     * @param associatedData Associated data (AAD); must match encryption input.
     * @return Plaintext, or null on failure.
     */
    suspend fun decrypt(ciphertext: ByteArray, associatedData: ByteArray? = null): ByteArray?
    
    /**
     * Rotate session key.
     */
    suspend fun rotateSessionKey(): Boolean
    
    /**
     * Get current key info.
     */
    fun getKeyInfo(): CryptoKeyInfo
    
    /**
     * Destroy sensitive data.
     */
    fun destroy()
    
    /**
     * Get current key version.
     */
    fun getCurrentKeyVersion(): String {
        return "KEY_V1"
    }
    
    /**
     * Get security status.
     */
    fun getSecurityStatus(): SecurityStatus {
        return SecurityStatus()
    }
    
    /**
     * Rotate keys.
     */
    suspend fun rotateKeys(): Boolean {
        return rotateSessionKey()
    }
    
    /**
     * Update server public key.
     */
    suspend fun updateServerPublicKey(publicKey: ByteArray): Boolean {
        return true
    }
}

/**
 * Security status snapshot.
 */
data class SecurityStatus(
    val isInitialized: Boolean = true,
    val hasValidKeys: Boolean = true,
    val isLocked: Boolean = false
)

/**
 * Key info snapshot.
 */
data class CryptoKeyInfo(
    val masterKeyExists: Boolean,      // Master key exists
    val sessionKeyActive: Boolean,     // Session key active
    val keysetProvider: String,        // Keyset provider (Android Keystore, StrongBox, etc.)
    val encryptionAlgorithm: String,   // Encryption algorithm
    val keyCreationTime: Long,         // Key creation time
    val keyRotationCount: Long         // Key rotation count
)
