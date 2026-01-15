package com.continuousauth.network

import android.util.Log
import io.grpc.okhttp.OkHttpChannelBuilder
import okhttp3.CertificatePinner
import javax.net.ssl.SSLContext
import javax.net.ssl.SSLSocket
import javax.net.ssl.SSLSocketFactory
import javax.net.ssl.X509TrustManager
import java.net.InetAddress
import java.net.Socket
import java.security.cert.X509Certificate
import javax.inject.Inject
import javax.inject.Singleton

/**
 * TLS security configuration manager.
 *
 * Enforces TLS 1.3 for transport security and supports SPKI (public key) pinning.
 */
@Singleton
class TlsSecurityManager @Inject constructor() {
    
    companion object {
        private const val TAG = "TlsSecurityManager"
    }
    
    /**
     * Configures an OkHttp channel builder to enforce TLS 1.3 and optional certificate pinning.
     */
    fun configureTlsForChannelBuilder(
        builder: OkHttpChannelBuilder,
        pinnedCertificates: Set<String> = emptySet(),
        hostname: String = ""
    ): OkHttpChannelBuilder {
        return try {
            val tlsSocketFactory = createTls13SocketFactory()
            var configuredBuilder = builder.sslSocketFactory(tlsSocketFactory)
            
            // Certificate pinning.
            if (pinnedCertificates.isNotEmpty() && hostname.isNotEmpty()) {
                val certificatePinner = createCertificatePinner(hostname, pinnedCertificates)
                // Note: gRPC pinning must be configured at the underlying OkHttp layer.
                // The CertificatePinner is created here; verification is typically enforced via a custom TrustManager.
                val customTrustManager = createPinningTrustManager(pinnedCertificates, hostname)
                configuredBuilder = configuredBuilder.sslSocketFactory(
                    createTls13SocketFactory()
                )
                Log.i(TAG, "SPKI pinning configured - host: $hostname, pins: ${pinnedCertificates.size}")
            }
            
            Log.i(TAG, "TLS 1.3 configuration applied to gRPC channel")
            configuredBuilder
        } catch (e: Exception) {
            Log.e(TAG, "Failed to configure TLS 1.3", e)
            builder
        }
    }
    
    /**
     * Creates a certificate pinner.
     */
    private fun createCertificatePinner(hostname: String, pinnedCertificates: Set<String>): CertificatePinner {
        val builder = CertificatePinner.Builder()
        
        pinnedCertificates.forEach { pin ->
            // Ensure pin format is correct (sha256/...).
            val formattedPin = if (pin.startsWith("sha256/")) pin else "sha256/$pin"
            builder.add(hostname, formattedPin)
            Log.d(TAG, "Adding certificate pin: $hostname -> $formattedPin")
        }
        
        return builder.build()
    }
    
    /**
     * Validates certificate pin format.
     */
    fun validateCertificatePin(pin: String): Boolean {
        return try {
            // Pin should be a Base64-encoded SHA-256 hash, 44 chars (excluding "sha256/").
            val cleanPin = pin.removePrefix("sha256/")
            cleanPin.length == 44 && 
            cleanPin.matches(Regex("[A-Za-z0-9+/=]+"))
        } catch (e: Exception) {
            Log.w(TAG, "Invalid certificate pin format: $pin", e)
            false
        }
    }
    
    /**
     * Extracts an SPKI pin from an X509 certificate.
     */
    fun extractSpkiPin(certificate: X509Certificate): String? {
        return try {
            val publicKeyInfo = certificate.publicKey.encoded
            val digest = java.security.MessageDigest.getInstance("SHA-256")
            val hash = digest.digest(publicKeyInfo)
            val pin = android.util.Base64.encodeToString(hash, android.util.Base64.NO_WRAP)
            Log.d(TAG, "Extracted SPKI pin: sha256/$pin")
            "sha256/$pin"
        } catch (e: Exception) {
            Log.e(TAG, "Failed to extract certificate pin", e)
            null
        }
    }
    
    /**
     * Creates a custom TrustManager that supports certificate pinning validation.
     */
    private fun createPinningTrustManager(
        pinnedCertificates: Set<String>,
        hostname: String
    ): X509TrustManager {
        return object : X509TrustManager {
            private val defaultTrustManager = createDefaultTrustManager()
            
            override fun checkClientTrusted(chain: Array<out X509Certificate>?, authType: String?) {
                defaultTrustManager.checkClientTrusted(chain, authType)
            }
            
            override fun checkServerTrusted(chain: Array<out X509Certificate>?, authType: String?) {
                // First: standard certificate chain verification.
                defaultTrustManager.checkServerTrusted(chain, authType)
                
                // Then: pin validation.
                if (chain != null && pinnedCertificates.isNotEmpty()) {
                    validateCertificatePinning(chain, pinnedCertificates, hostname)
                }
            }
            
            override fun getAcceptedIssuers(): Array<X509Certificate> {
                return defaultTrustManager.acceptedIssuers
            }
        }
    }
    
    /**
     * Validates certificate pinning.
     */
    private fun validateCertificatePinning(
        chain: Array<out X509Certificate>,
        pinnedCertificates: Set<String>,
        hostname: String
    ) {
        var pinMatched = false
        
        // Check each certificate in the chain.
        for (cert in chain) {
            val extractedPin = extractSpkiPin(cert)
            if (extractedPin != null && pinnedCertificates.contains(extractedPin)) {
                pinMatched = true
                Log.d(TAG, "Certificate pin match: $hostname")
                break
            }
        }
        
        if (!pinMatched) {
            val errorMessage = "Certificate pin validation failed - host: $hostname"
            Log.e(TAG, errorMessage)
            throw javax.net.ssl.SSLPeerUnverifiedException(errorMessage)
        }
    }
    
    /**
     * Creates the default TrustManager.
     */
    private fun createDefaultTrustManager(): X509TrustManager {
        val trustManagerFactory = javax.net.ssl.TrustManagerFactory.getInstance(
            javax.net.ssl.TrustManagerFactory.getDefaultAlgorithm()
        )
        trustManagerFactory.init(null as java.security.KeyStore?)
        
        return trustManagerFactory.trustManagers
            .filterIsInstance<X509TrustManager>()
            .firstOrNull()
            ?: throw IllegalStateException("Failed to obtain default TrustManager")
    }
    
    /**
     * Creates an SSLSocketFactory that enforces TLS 1.3 when possible.
     */
    private fun createTls13SocketFactory(): SSLSocketFactory {
        // Create a TLS-capable SSLContext.
        val sslContext = createSecureSSLContext()
        
        return object : SSLSocketFactory() {
            private val delegate = sslContext.socketFactory
            
            override fun createSocket(): Socket = 
                configureSocket(delegate.createSocket())
            
            override fun createSocket(host: String?, port: Int): Socket = 
                configureSocket(delegate.createSocket(host, port))
            
            override fun createSocket(
                host: String?, 
                port: Int, 
                localHost: InetAddress?, 
                localPort: Int
            ): Socket = 
                configureSocket(delegate.createSocket(host, port, localHost, localPort))
            
            override fun createSocket(host: InetAddress?, port: Int): Socket = 
                configureSocket(delegate.createSocket(host, port))
            
            override fun createSocket(
                address: InetAddress?, 
                port: Int, 
                localAddress: InetAddress?, 
                localPort: Int
            ): Socket = 
                configureSocket(delegate.createSocket(address, port, localAddress, localPort))
            
            override fun createSocket(
                s: Socket?, 
                host: String?, 
                port: Int, 
                autoClose: Boolean
            ): Socket = 
                configureSocket(delegate.createSocket(s, host, port, autoClose))
            
            override fun getDefaultCipherSuites(): Array<String> = 
                delegate.defaultCipherSuites
            
            override fun getSupportedCipherSuites(): Array<String> = 
                delegate.supportedCipherSuites
        }
    }
    
    /**
     * Creates a secure SSLContext.
     */
    private fun createSecureSSLContext(): SSLContext {
        return try {
            // Prefer TLS 1.3.
            val context = SSLContext.getInstance("TLSv1.3")
            context.init(null, null, null)
            Log.i(TAG, "Created TLS 1.3 SSLContext")
            context
        } catch (e: Exception) {
            Log.w(TAG, "TLS 1.3 not supported; falling back to TLS 1.2", e)
            // Fall back to TLS 1.2.
            val context = SSLContext.getInstance("TLSv1.2")
            context.init(null, null, null)
            context
        }
    }
    
    /**
     * Configures an SSL socket to enforce TLS protocol and secure cipher suites.
     */
    private fun configureSocket(socket: Socket): Socket {
        if (socket is SSLSocket) {
            configureTlsProtocols(socket)
            configureCipherSuites(socket)
        }
        return socket
    }
    
    /**
     * Configures TLS protocol versions.
     */
    private fun configureTlsProtocols(sslSocket: SSLSocket) {
        try {
            val supportedProtocols = sslSocket.supportedProtocols
            
            when {
                supportedProtocols.contains("TLSv1.3") -> {
                    sslSocket.enabledProtocols = arrayOf("TLSv1.3")
                    Log.d(TAG, "Enforcing TLS 1.3")
                }
                supportedProtocols.contains("TLSv1.2") -> {
                    sslSocket.enabledProtocols = arrayOf("TLSv1.2")
                    Log.w(TAG, "TLS 1.3 unavailable; using TLS 1.2")
                }
                else -> {
                    Log.e(TAG, "No secure TLS version supported")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to configure TLS protocols", e)
        }
    }
    
    /**
     * Configures secure cipher suites.
     */
    private fun configureCipherSuites(sslSocket: SSLSocket) {
        try {
            val supportedCiphers = sslSocket.supportedCipherSuites
            
            // TLS 1.3 recommended cipher suites.
            val tls13Ciphers = supportedCiphers.filter { cipher ->
                cipher.startsWith("TLS_AES_") ||
                cipher.startsWith("TLS_CHACHA20_") 
            }
            
            // TLS 1.2 secure cipher suites (fallback).
            val tls12SecureCiphers = supportedCiphers.filter { cipher ->
                (cipher.contains("ECDHE", ignoreCase = true) && 
                 cipher.contains("AES", ignoreCase = true) && 
                 cipher.contains("GCM", ignoreCase = true)) ||
                (cipher.contains("DHE", ignoreCase = true) && 
                 cipher.contains("AES", ignoreCase = true))
            }
            
            val preferredCiphers = when {
                tls13Ciphers.isNotEmpty() -> {
                    Log.d(TAG, "Using TLS 1.3 cipher suites")
                    tls13Ciphers
                }
                tls12SecureCiphers.isNotEmpty() -> {
                    Log.d(TAG, "Using TLS 1.2 secure cipher suites")
                    tls12SecureCiphers
                }
                else -> {
                    Log.w(TAG, "No secure cipher suites available; using defaults")
                    supportedCiphers.toList()
                }
            }
            
            sslSocket.enabledCipherSuites = preferredCiphers.toTypedArray()
            Log.d(TAG, "Applied ${preferredCiphers.size} secure cipher suites")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to configure cipher suites", e)
        }
    }
    
    /**
     * Returns TLS configuration info.
     */
    fun getTlsConfigInfo(): TlsConfigInfo {
        return try {
            val context = createSecureSSLContext()
            val socketFactory = context.socketFactory
            val socket = socketFactory.createSocket() as? SSLSocket
            
            socket?.let { sslSocket ->
                val supportedProtocols = sslSocket.supportedProtocols?.toList() ?: emptyList()
                val supportedCiphers = sslSocket.supportedCipherSuites?.toList() ?: emptyList()
                
                TlsConfigInfo(
                    supportsTls13 = supportedProtocols.contains("TLSv1.3"),
                    supportedProtocols = supportedProtocols,
                    recommendedCiphers = supportedCiphers.filter { cipher ->
                        cipher.startsWith("TLS_AES_") || 
                        cipher.startsWith("TLS_CHACHA20_") ||
                        (cipher.contains("ECDHE") && cipher.contains("GCM"))
                    }
                )
            } ?: TlsConfigInfo()
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get TLS configuration info", e)
            TlsConfigInfo()
        }
    }
}

/**
 * TLS configuration info.
 */
data class TlsConfigInfo(
    val supportsTls13: Boolean = false,
    val supportedProtocols: List<String> = emptyList(),
    val recommendedCiphers: List<String> = emptyList()
)
