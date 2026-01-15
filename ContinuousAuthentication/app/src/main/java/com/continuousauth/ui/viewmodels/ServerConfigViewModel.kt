package com.continuousauth.ui.viewmodels

import android.annotation.SuppressLint
import android.content.Context
import android.provider.Settings
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.continuousauth.network.Uploader
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import javax.inject.Inject

/**
 * ViewModel for the server configuration screen.
 */
@HiltViewModel
class ServerConfigViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val uploader: Uploader
) : ViewModel() {
    
    enum class ConnectionStatus {
        CONNECTED,
        DISCONNECTED,
        CONNECTING,
        ERROR
    }
    
    data class ServerConfig(
        val ip: String = "",
        val port: Int = 0  // Default 0 means unset.
    )
    
    // Device ID
    private val _deviceId = MutableLiveData<String>()
    val deviceId: LiveData<String> = _deviceId
    
    // Server config
    private val _serverConfig = MutableLiveData<ServerConfig>()
    val serverConfig: LiveData<ServerConfig> = _serverConfig
    
    // Connection status
    private val _connectionStatus = MutableLiveData(ConnectionStatus.DISCONNECTED)
    val connectionStatus: LiveData<ConnectionStatus> = _connectionStatus
    
    // Upload status
    private val _isUploading = MutableLiveData(false)
    val isUploading: LiveData<Boolean> = _isUploading
    
    // Error message
    private val _errorMessage = MutableLiveData<String?>()
    val errorMessage: LiveData<String?> = _errorMessage
    
    // Success message
    private val _successMessage = MutableLiveData<String?>()
    val successMessage: LiveData<String?> = _successMessage
    
    init {
        loadDeviceId()
    }
    
    /**
     * Loads the device ID.
     */
    @SuppressLint("HardwareIds")
    private fun loadDeviceId() {
        _deviceId.value = Settings.Secure.getString(
            context.contentResolver,
            Settings.Secure.ANDROID_ID
        )
    }
    
    /**
     * Loads the persisted server config.
     */
    fun loadServerConfig() {
        viewModelScope.launch {
            // Load config from SharedPreferences (or DataStore).
            val prefs = context.getSharedPreferences("server_config", Context.MODE_PRIVATE)
            val ip = prefs.getString("server_ip", "") ?: ""  // Default empty string
            val port = prefs.getInt("server_port", 0)  // Default 0
            
            _serverConfig.value = ServerConfig(ip, port)
        }
    }
    
    /**
     * Saves the server config.
     */
    fun saveServerConfig(ip: String, port: Int) {
        viewModelScope.launch {
            _serverConfig.value = ServerConfig(ip, port)
            
            // Persist to SharedPreferences.
            val prefs = context.getSharedPreferences("server_config", Context.MODE_PRIVATE)
            prefs.edit()
                .putString("server_ip", ip)
                .putInt("server_port", port)
                .apply()
        }
    }
    
    /**
     * Tests server connectivity.
     */
    suspend fun testConnection() {
        withContext(Dispatchers.IO) {
            try {
                _connectionStatus.postValue(ConnectionStatus.CONNECTING)
                
                val config = _serverConfig.value ?: return@withContext
                val endpoint = "${config.ip}:${config.port}"
                
                // Connect and disconnect immediately to validate connectivity.
                val result = uploader.connect(endpoint)
                
                if (result) {
                    _connectionStatus.postValue(ConnectionStatus.CONNECTED)
                    // Use string resources for localization.
                    _successMessage.postValue(context.getString(com.continuousauth.R.string.connection_test_success))
                    // Disconnect after the test.
                    uploader.disconnect()
                    _connectionStatus.postValue(ConnectionStatus.DISCONNECTED)
                } else {
                    _connectionStatus.postValue(ConnectionStatus.ERROR)
                    _errorMessage.postValue(context.getString(com.continuousauth.R.string.connection_test_failed))
                }
            } catch (e: Exception) {
                _connectionStatus.postValue(ConnectionStatus.ERROR)
                _errorMessage.postValue(context.getString(com.continuousauth.R.string.connection_test_error, e.message))
            }
        }
    }
    
    /**
     * Starts upload.
     */
    suspend fun startUpload() {
        withContext(Dispatchers.IO) {
            try {
                val config = _serverConfig.value ?: return@withContext
                
                // Connect to server.
                _connectionStatus.postValue(ConnectionStatus.CONNECTING)
                val endpoint = "${config.ip}:${config.port}"
                val connected = uploader.connect(endpoint)
                
                if (connected) {
                    _connectionStatus.postValue(ConnectionStatus.CONNECTED)
                    _isUploading.postValue(true)
                    _successMessage.postValue(context.getString(com.continuousauth.R.string.upload_started))
                } else {
                    _connectionStatus.postValue(ConnectionStatus.ERROR)
                    _errorMessage.postValue(context.getString(com.continuousauth.R.string.connection_failed))
                }
            } catch (e: Exception) {
                _connectionStatus.postValue(ConnectionStatus.ERROR)
                _errorMessage.postValue(context.getString(com.continuousauth.R.string.upload_start_failed, e.message))
                _isUploading.postValue(false)
            }
        }
    }
    
    /**
     * Stops upload.
     */
    suspend fun stopUpload() {
        withContext(Dispatchers.IO) {
            try {
                // Disconnect.
                uploader.disconnect()
                
                _isUploading.postValue(false)
                _connectionStatus.postValue(ConnectionStatus.DISCONNECTED)
                _successMessage.postValue(context.getString(com.continuousauth.R.string.upload_stopped))
            } catch (e: Exception) {
                _errorMessage.postValue(context.getString(com.continuousauth.R.string.upload_stop_failed, e.message))
            }
        }
    }
    
    /**
     * Clears the current messages.
     */
    fun clearMessages() {
        _errorMessage.value = null
        _successMessage.value = null
    }
}
