package com.continuousauth.network

import android.content.Context
import android.net.*
import android.net.wifi.WifiManager
import android.os.Build
import android.telephony.TelephonyManager
import android.util.Log
import androidx.annotation.RequiresApi
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Network environment detector.
 *
 * Detects current network type and quality to adjust transmission strategy.
 */
@Singleton
class NetworkEnvironmentDetector @Inject constructor(
    @ApplicationContext private val context: Context
) {
    
    companion object {
        private const val TAG = "NetworkEnvironmentDetector"
        private const val PING_TIMEOUT_MS = 3000
        private const val GOOD_LATENCY_THRESHOLD_MS = 100
        private const val POOR_LATENCY_THRESHOLD_MS = 500
    }
    
    private val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
    private val telephonyManager = context.getSystemService(Context.TELEPHONY_SERVICE) as TelephonyManager
    private val wifiManager = context.getSystemService(Context.WIFI_SERVICE) as WifiManager
    
    private val detectorScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    
    // Network state updates.
    private val _networkStateFlow = MutableSharedFlow<NetworkState>(replay = 1)
    val networkStateFlow: SharedFlow<NetworkState> = _networkStateFlow.asSharedFlow()
    
    // Current state.
    private var currentNetworkState = NetworkState.UNKNOWN
    
    // Network callback.
    private var networkCallback: ConnectivityManager.NetworkCallback? = null
    
    init {
        startNetworkMonitoring()
    }
    
    /**
     * Starts monitoring network state.
     */
    private fun startNetworkMonitoring() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            registerNetworkCallback()
        }
        
        // Initial detection.
        detectorScope.launch {
            updateNetworkState()
        }
    }
    
    /**
     * Registers network callback (Android N+).
     */
    @RequiresApi(Build.VERSION_CODES.N)
    private fun registerNetworkCallback() {
        networkCallback = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) {
                Log.d(TAG, "Network available: $network")
                detectorScope.launch {
                    updateNetworkState()
                }
            }
            
            override fun onLost(network: Network) {
                Log.d(TAG, "Network lost: $network")
                detectorScope.launch {
                    updateNetworkState()
                }
            }
            
            override fun onCapabilitiesChanged(
                network: Network,
                networkCapabilities: NetworkCapabilities
            ) {
                Log.d(TAG, "Network capabilities changed: $network")
                detectorScope.launch {
                    updateNetworkState()
                }
            }
        }
        
        try {
            connectivityManager.registerDefaultNetworkCallback(networkCallback!!)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to register network callback", e)
        }
    }
    
    /**
     * Updates network state.
     */
    private suspend fun updateNetworkState() {
        val newState = detectNetworkState()
        
        if (newState != currentNetworkState) {
            currentNetworkState = newState
            Log.i(TAG, "Network state changed: $newState")
            _networkStateFlow.emit(newState)
        }
    }
    
    /**
     * Detects current network state.
     */
    private suspend fun detectNetworkState(): NetworkState {
        return try {
            when {
                !isNetworkAvailable() -> NetworkState.DISCONNECTED
                isWifiConnected() -> detectWifiQuality()
                isCellularConnected() -> detectCellularQuality()
                else -> NetworkState.UNKNOWN
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to detect network state", e)
            NetworkState.UNKNOWN
        }
    }
    
    /**
     * Detects WiFi quality.
     */
    private suspend fun detectWifiQuality(): NetworkState {
        return try {
            val wifiInfo = wifiManager.connectionInfo
            val rssi = wifiInfo.rssi
            val linkSpeed = wifiInfo.linkSpeed
            
            Log.d(TAG, "WiFi - RSSI: $rssi, linkSpeed: ${linkSpeed}Mbps")
            
            // Determine WiFi quality from RSSI and link speed.
            when {
                rssi > -50 && linkSpeed >= 100 -> NetworkState.WIFI_EXCELLENT
                rssi > -70 && linkSpeed >= 50 -> NetworkState.WIFI_GOOD
                else -> NetworkState.WIFI_POOR
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to detect WiFi quality", e)
            NetworkState.WIFI_UNKNOWN
        }
    }
    
    /**
     * Detects cellular network quality.
     */
    private suspend fun detectCellularQuality(): NetworkState {
        return try {
            val networkType = getNetworkType()
            val signalStrength = getCellularSignalStrength()
            
            Log.d(TAG, "Cellular - type: $networkType, signalStrength: $signalStrength")
            
            when (networkType) {
                // 5G
                "5G" -> if (signalStrength >= -85) NetworkState.CELLULAR_EXCELLENT else NetworkState.CELLULAR_GOOD
                
                // 4G
                "LTE", "4G" -> when {
                    signalStrength >= -85 -> NetworkState.CELLULAR_EXCELLENT
                    signalStrength >= -105 -> NetworkState.CELLULAR_GOOD
                    else -> NetworkState.CELLULAR_POOR
                }
                
                // 3G
                "3G", "HSPA", "HSUPA", "HSDPA" -> when {
                    signalStrength >= -90 -> NetworkState.CELLULAR_GOOD
                    else -> NetworkState.CELLULAR_POOR
                }
                
                // 2G
                "2G", "GSM", "EDGE", "GPRS" -> NetworkState.CELLULAR_POOR
                
                else -> NetworkState.CELLULAR_UNKNOWN
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to detect cellular quality", e)
            NetworkState.CELLULAR_UNKNOWN
        }
    }
    
    /**
     * Returns true if the network is available.
     */
    private fun isNetworkAvailable(): Boolean {
        val activeNetwork = connectivityManager.activeNetwork
        val networkCapabilities = connectivityManager.getNetworkCapabilities(activeNetwork)
        
        return networkCapabilities?.let {
            it.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
            it.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
        } ?: false
    }
    
    
    /**
     * Returns true if connected to cellular.
     */
    private fun isCellularConnected(): Boolean {
        val activeNetwork = connectivityManager.activeNetwork
        val networkCapabilities = connectivityManager.getNetworkCapabilities(activeNetwork)
        
        return networkCapabilities?.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) ?: false
    }
    
    /**
     * Returns a network type label.
     */
    private fun getNetworkType(): String {
        return try {
            @Suppress("DEPRECATION")
            when (telephonyManager.networkType) {
                TelephonyManager.NETWORK_TYPE_GPRS,
                TelephonyManager.NETWORK_TYPE_EDGE,
                TelephonyManager.NETWORK_TYPE_CDMA,
                TelephonyManager.NETWORK_TYPE_1xRTT,
                TelephonyManager.NETWORK_TYPE_IDEN -> "2G"
                
                TelephonyManager.NETWORK_TYPE_UMTS,
                TelephonyManager.NETWORK_TYPE_EVDO_0,
                TelephonyManager.NETWORK_TYPE_EVDO_A,
                TelephonyManager.NETWORK_TYPE_HSDPA,
                TelephonyManager.NETWORK_TYPE_HSUPA,
                TelephonyManager.NETWORK_TYPE_HSPA,
                TelephonyManager.NETWORK_TYPE_EVDO_B,
                TelephonyManager.NETWORK_TYPE_EHRPD,
                TelephonyManager.NETWORK_TYPE_HSPAP -> "3G"
                
                TelephonyManager.NETWORK_TYPE_LTE -> "4G"
                
                else -> {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        if (telephonyManager.networkType == TelephonyManager.NETWORK_TYPE_NR) {
                            "5G"
                        } else {
                            "UNKNOWN"
                        }
                    } else {
                        "UNKNOWN"
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get network type", e)
            "UNKNOWN"
        }
    }
    
    /**
     * Gets cellular signal strength (dBm).
     */
    private fun getCellularSignalStrength(): Int {
        return try {
            val signalStrength = telephonyManager.signalStrength
            signalStrength?.let {
                // Use reflection because APIs differ across Android versions.
                val method = it.javaClass.getMethod("getDbm")
                method.invoke(it) as? Int ?: -999
            } ?: -999
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get signal strength; using estimated value", e)
            // Default to a medium signal strength.
            -95
        }
    }
    
    /**
     * Returns current network state.
     */
    fun getCurrentNetworkState(): NetworkState {
        return currentNetworkState
    }
    
    /**
     * Returns true if connected to WiFi.
     */
    fun isWifiConnected(): Boolean {
        return when (currentNetworkState) {
            NetworkState.WIFI_EXCELLENT,
            NetworkState.WIFI_GOOD,
            NetworkState.WIFI_POOR,
            NetworkState.WIFI_UNKNOWN -> true
            else -> false
        }
    }
    
    /**
     * Returns true if network is available for upload (policy-aware).
     *
     * @param wifiOnly If true, only WiFi is allowed.
     */
    fun isNetworkAvailableForUpload(wifiOnly: Boolean): Boolean {
        return if (wifiOnly) {
            isWifiConnected()
        } else {
            currentNetworkState != NetworkState.DISCONNECTED && 
            currentNetworkState != NetworkState.UNKNOWN
        }
    }
    
    /**
     * Returns suggested transmission configuration.
     */
    fun getTransmissionConfig(): TransmissionConfig {
        return when (currentNetworkState) {
            NetworkState.WIFI_EXCELLENT -> TransmissionConfig(
                maxConcurrentUploads = 4,
                uploadTimeoutMs = 10000,
                retryDelayMs = 1000,
                maxRetryAttempts = 3
            )
            
            NetworkState.WIFI_GOOD -> TransmissionConfig(
                maxConcurrentUploads = 3,
                uploadTimeoutMs = 15000,
                retryDelayMs = 2000,
                maxRetryAttempts = 3
            )
            
            NetworkState.CELLULAR_EXCELLENT -> TransmissionConfig(
                maxConcurrentUploads = 2,
                uploadTimeoutMs = 15000,
                retryDelayMs = 2000,
                maxRetryAttempts = 4
            )
            
            NetworkState.CELLULAR_GOOD -> TransmissionConfig(
                maxConcurrentUploads = 1,
                uploadTimeoutMs = 20000,
                retryDelayMs = 3000,
                maxRetryAttempts = 5
            )
            
            NetworkState.CELLULAR_POOR,
            NetworkState.WIFI_POOR -> TransmissionConfig(
                maxConcurrentUploads = 1,
                uploadTimeoutMs = 30000,
                retryDelayMs = 5000,
                maxRetryAttempts = 8
            )
            
            NetworkState.DISCONNECTED -> TransmissionConfig(
                maxConcurrentUploads = 0,
                uploadTimeoutMs = 0,
                retryDelayMs = 10000,
                maxRetryAttempts = 0
            )
            
            else -> TransmissionConfig() // Default config
        }
    }
    
    /**
     * Cleans up resources.
     */
    fun cleanup() {
        networkCallback?.let { callback ->
            try {
                connectivityManager.unregisterNetworkCallback(callback)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to unregister network callback", e)
            }
        }
        detectorScope.cancel()
    }
}

/**
 * Network state.
 */
enum class NetworkState {
    UNKNOWN,                    // Unknown
    DISCONNECTED,              // Disconnected
    
    // WiFi
    WIFI_EXCELLENT,            // Excellent (RSSI > -50, linkSpeed >= 100Mbps)
    WIFI_GOOD,                 // Good (RSSI > -70, linkSpeed >= 50Mbps)
    WIFI_POOR,                 // Poor (other WiFi conditions)
    WIFI_UNKNOWN,              // Unknown quality
    
    // Cellular
    CELLULAR_EXCELLENT,        // Excellent (5G or high-quality 4G)
    CELLULAR_GOOD,             // Good (mid-quality 4G/3G)
    CELLULAR_POOR,             // Poor (weak 4G or 3G/2G)
    CELLULAR_UNKNOWN           // Unknown quality
}

/**
 * Suggested transmission configuration.
 */
data class TransmissionConfig(
    val maxConcurrentUploads: Int = 2,      // Max concurrent uploads
    val uploadTimeoutMs: Long = 15000,       // Upload timeout (ms)
    val retryDelayMs: Long = 3000,          // Retry delay (ms)
    val maxRetryAttempts: Int = 5           // Max retry attempts
)
