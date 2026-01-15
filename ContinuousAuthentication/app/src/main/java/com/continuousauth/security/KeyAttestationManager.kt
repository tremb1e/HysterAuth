package com.continuousauth.security

import android.content.Context
import android.os.Build
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Log
import androidx.annotation.RequiresApi
import dagger.hilt.android.qualifiers.ApplicationContext
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.cert.X509Certificate
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Key attestation result.
 *
 * Contains details from key attestation verification.
 */
data class AttestationResult(
    val isSuccess: Boolean,                     // Whether attestation succeeded
    val isHardwareBacked: Boolean,              // Whether hardware-backed keystore is used
    val isStrongBoxBacked: Boolean,             // Whether StrongBox is used
    val attestationSecurityLevel: Int,          // Attestation security level
    val keySecurityLevel: Int,                  // Key security level
    val bootPatchLevel: Int?,                   // Boot patch level
    val vendorPatchLevel: Int?,                 // Vendor patch level
    val appId: String?,                         // App ID
    val appVersion: Long?,                      // App version
    val errorMessage: String? = null            // Error message
)

/**
 * Key attestation manager interface.
 *
 * Defines the core key attestation APIs.
 */
interface KeyAttestationManager {
    /**
     * Generates a key and requests attestation.
     *
     * @param keyAlias Key alias
     * @param challenge Attestation challenge
     * @return Attestation result
     */
    suspend fun generateAndAttestKey(keyAlias: String, challenge: ByteArray): AttestationResult
    
    /**
     * Verifies attestation for an existing key.
     *
     * @param keyAlias Key alias
     * @return Attestation result
     */
    suspend fun verifyKeyAttestation(keyAlias: String): AttestationResult
    
    /**
     * Checks whether the device supports key attestation.
     *
     * @return True if supported
     */
    fun isAttestationSupported(): Boolean
    
    /**
     * Checks whether the device supports StrongBox.
     *
     * @return True if StrongBox is supported
     */
    fun isStrongBoxSupported(): Boolean
}

/**
 * Key attestation manager implementation.
 *
 * Uses Android Keystore Key Attestation to validate key properties.
 */
@Singleton
class KeyAttestationManagerImpl @Inject constructor(
    @ApplicationContext private val context: Context
) : KeyAttestationManager {
    
    companion object {
        private const val TAG = "KeyAttestationManager"
        private const val KEYSTORE_PROVIDER = "AndroidKeyStore"
        
        // Attestation extension OID
        private const val KEY_DESCRIPTION_OID = "1.3.6.1.4.1.11129.2.1.17"
        
        // Security level constants
        private const val SECURITY_LEVEL_SOFTWARE = 0
        private const val SECURITY_LEVEL_TRUSTED_ENVIRONMENT = 1
        private const val SECURITY_LEVEL_STRONGBOX = 2
    }
    
    override suspend fun generateAndAttestKey(keyAlias: String, challenge: ByteArray): AttestationResult {
        return try {
            if (!isAttestationSupported()) {
                return AttestationResult(
                    isSuccess = false,
                    isHardwareBacked = false,
                    isStrongBoxBacked = false,
                    attestationSecurityLevel = SECURITY_LEVEL_SOFTWARE,
                    keySecurityLevel = SECURITY_LEVEL_SOFTWARE,
                    bootPatchLevel = null,
                    vendorPatchLevel = null,
                    appId = null,
                    appVersion = null,
                    errorMessage = "Device does not support key attestation"
                )
            }
            
            // Generate a key pair and request attestation.
            val keyPair = generateAttestationKey(keyAlias, challenge)
            
            // Retrieve the attestation certificate chain.
            val keyStore = KeyStore.getInstance(KEYSTORE_PROVIDER).apply { load(null) }
            val certChain = keyStore.getCertificateChain(keyAlias)
            
            if (certChain.isNullOrEmpty()) {
                return AttestationResult(
                    isSuccess = false,
                    isHardwareBacked = false,
                    isStrongBoxBacked = false,
                    attestationSecurityLevel = SECURITY_LEVEL_SOFTWARE,
                    keySecurityLevel = SECURITY_LEVEL_SOFTWARE,
                    bootPatchLevel = null,
                    vendorPatchLevel = null,
                    appId = null,
                    appVersion = null,
                    errorMessage = "Unable to retrieve attestation certificate chain"
                )
            }
            
            // Parse attestation data.
            parseAttestationCertificate(certChain[0] as X509Certificate)
            
        } catch (e: Exception) {
            Log.e(TAG, "Key attestation failed", e)
            AttestationResult(
                isSuccess = false,
                isHardwareBacked = false,
                isStrongBoxBacked = false,
                attestationSecurityLevel = SECURITY_LEVEL_SOFTWARE,
                keySecurityLevel = SECURITY_LEVEL_SOFTWARE,
                bootPatchLevel = null,
                vendorPatchLevel = null,
                appId = null,
                appVersion = null,
                errorMessage = e.message
            )
        }
    }
    
    override suspend fun verifyKeyAttestation(keyAlias: String): AttestationResult {
        return try {
            val keyStore = KeyStore.getInstance(KEYSTORE_PROVIDER).apply { load(null) }
            
            if (!keyStore.containsAlias(keyAlias)) {
                return AttestationResult(
                    isSuccess = false,
                    isHardwareBacked = false,
                    isStrongBoxBacked = false,
                    attestationSecurityLevel = SECURITY_LEVEL_SOFTWARE,
                    keySecurityLevel = SECURITY_LEVEL_SOFTWARE,
                    bootPatchLevel = null,
                    vendorPatchLevel = null,
                    appId = null,
                    appVersion = null,
                    errorMessage = "Key does not exist"
                )
            }
            
            val certChain = keyStore.getCertificateChain(keyAlias)
            if (certChain.isNullOrEmpty()) {
                return AttestationResult(
                    isSuccess = false,
                    isHardwareBacked = false,
                    isStrongBoxBacked = false,
                    attestationSecurityLevel = SECURITY_LEVEL_SOFTWARE,
                    keySecurityLevel = SECURITY_LEVEL_SOFTWARE,
                    bootPatchLevel = null,
                    vendorPatchLevel = null,
                    appId = null,
                    appVersion = null,
                    errorMessage = "Unable to retrieve certificate chain"
                )
            }
            
            parseAttestationCertificate(certChain[0] as X509Certificate)
            
        } catch (e: Exception) {
            Log.e(TAG, "Key attestation verification failed", e)
            AttestationResult(
                isSuccess = false,
                isHardwareBacked = false,
                isStrongBoxBacked = false,
                attestationSecurityLevel = SECURITY_LEVEL_SOFTWARE,
                keySecurityLevel = SECURITY_LEVEL_SOFTWARE,
                bootPatchLevel = null,
                vendorPatchLevel = null,
                appId = null,
                appVersion = null,
                errorMessage = e.message
            )
        }
    }
    
    override fun isAttestationSupported(): Boolean {
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.N
    }
    
    override fun isStrongBoxSupported(): Boolean {
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.P &&
               context.packageManager.hasSystemFeature("android.hardware.strongbox_keystore")
    }
    
    /**
     * Generates an attested key pair.
     */
    @RequiresApi(Build.VERSION_CODES.N)
    private fun generateAttestationKey(keyAlias: String, challenge: ByteArray) {
        val keyGenParameterSpec = KeyGenParameterSpec.Builder(
            keyAlias,
            KeyProperties.PURPOSE_SIGN or KeyProperties.PURPOSE_VERIFY
        ).apply {
            setDigests(KeyProperties.DIGEST_SHA256)
            setSignaturePaddings(KeyProperties.SIGNATURE_PADDING_RSA_PSS)
            setKeySize(2048)
            setAttestationChallenge(challenge)
            
            // Prefer StrongBox when available.
            if (isStrongBoxSupported()) {
                setIsStrongBoxBacked(true)
            }
            
            // Require user authentication (if needed).
            setUserAuthenticationRequired(false)
            
        }.build()
        
        val keyPairGenerator = KeyPairGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_RSA, 
            KEYSTORE_PROVIDER
        )
        keyPairGenerator.initialize(keyGenParameterSpec)
        keyPairGenerator.generateKeyPair()
    }
    
    /**
     * Parses the attestation certificate.
     *
     * Note: this is a simplified implementation; production code should perform full ASN.1 parsing.
     */
    private fun parseAttestationCertificate(certificate: X509Certificate): AttestationResult {
        return try {
            // Check for the attestation extension.
            val attestationExtension = certificate.getExtensionValue(KEY_DESCRIPTION_OID)
            val hasAttestation = attestationExtension != null
            
            // Simplified parsing of attestation information.
            val isHardwareBacked = hasAttestation
            val isStrongBoxBacked = isStrongBoxSupported() && hasAttestation
            
            AttestationResult(
                isSuccess = hasAttestation,
                isHardwareBacked = isHardwareBacked,
                isStrongBoxBacked = isStrongBoxBacked,
                attestationSecurityLevel = if (isStrongBoxBacked) SECURITY_LEVEL_STRONGBOX 
                    else if (isHardwareBacked) SECURITY_LEVEL_TRUSTED_ENVIRONMENT 
                    else SECURITY_LEVEL_SOFTWARE,
                keySecurityLevel = if (isStrongBoxBacked) SECURITY_LEVEL_STRONGBOX 
                    else if (isHardwareBacked) SECURITY_LEVEL_TRUSTED_ENVIRONMENT 
                    else SECURITY_LEVEL_SOFTWARE,
                bootPatchLevel = extractPatchLevel("boot"),
                vendorPatchLevel = extractPatchLevel("vendor"),
                appId = context.packageName,
                appVersion = try {
                    context.packageManager.getPackageInfo(context.packageName, 0).longVersionCode
                } catch (e: Exception) {
                    null
                }
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse attestation certificate", e)
            AttestationResult(
                isSuccess = false,
                isHardwareBacked = false,
                isStrongBoxBacked = false,
                attestationSecurityLevel = SECURITY_LEVEL_SOFTWARE,
                keySecurityLevel = SECURITY_LEVEL_SOFTWARE,
                bootPatchLevel = null,
                vendorPatchLevel = null,
                appId = null,
                appVersion = null,
                errorMessage = e.message
            )
        }
    }
    
    /**
     * Extracts patch level information.
     *
     * Note: vendor patch level requires parsing attestation extensions; this implementation only reads the OS
     * security patch for boot.
     */
    private fun extractPatchLevel(type: String): Int? {
        return try {
            when (type) {
                "boot" -> Build.VERSION.SECURITY_PATCH.replace("-", "").toIntOrNull()
                "vendor" -> null // Parse from attestation extension in a complete implementation.
                else -> null
            }
        } catch (e: Exception) {
            null
        }
    }
    
    /**
     * Builds a formatted attestation report.
     */
    fun getFormattedAttestationReport(result: AttestationResult): String {
        val sb = StringBuilder()
        
        sb.appendLine("=== Key Attestation Report ===")
        sb.appendLine("Attestation: ${if (result.isSuccess) "success" else "failure"}")
        sb.appendLine("Hardware-backed: ${if (result.isHardwareBacked) "yes" else "no"}")
        sb.appendLine("StrongBox-backed: ${if (result.isStrongBoxBacked) "yes" else "no"}")
        sb.appendLine("Attestation security level: ${getSecurityLevelName(result.attestationSecurityLevel)}")
        sb.appendLine("Key security level: ${getSecurityLevelName(result.keySecurityLevel)}")
        
        result.bootPatchLevel?.let {
            sb.appendLine("Boot patch level: $it")
        }
        
        result.vendorPatchLevel?.let {
            sb.appendLine("Vendor patch level: $it")
        }
        
        result.appId?.let {
            sb.appendLine("App ID: $it")
        }
        
        result.appVersion?.let {
            sb.appendLine("App version: $it")
        }
        
        result.errorMessage?.let {
            sb.appendLine("Error: $it")
        }
        
        return sb.toString()
    }
    
    private fun getSecurityLevelName(level: Int): String {
        return when (level) {
            SECURITY_LEVEL_SOFTWARE -> "software"
            SECURITY_LEVEL_TRUSTED_ENVIRONMENT -> "trusted environment"
            SECURITY_LEVEL_STRONGBOX -> "StrongBox"
            else -> "unknown($level)"
        }
    }
}
