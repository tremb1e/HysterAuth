package com.continuousauth.ui.viewmodels

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.continuousauth.sensor.SensorCollector
import com.continuousauth.utils.ForegroundAppDetector
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import java.util.Locale
import javax.inject.Inject
import com.continuousauth.model.SensorType

/**
 * ViewModel for the sensor data screen.
 */
@HiltViewModel
class SensorDataViewModel @Inject constructor(
    private val sensorCollector: SensorCollector,
    private val foregroundAppDetector: ForegroundAppDetector
) : ViewModel() {
    
    data class SensorData(
        val x: Float = 0f,
        val y: Float = 0f,
        val z: Float = 0f,
        val timestamp: Long = System.currentTimeMillis()
    )
    
    data class RecentApp(
        val packageName: String,
        val appName: String,
        val timestamp: Long
    )
    
    // Sensor data streams (exposed as read-only StateFlow).
    private val _accelerometerData = MutableStateFlow(SensorData())
    val accelerometerData: StateFlow<SensorData> = _accelerometerData.asStateFlow()
    
    private val _gyroscopeData = MutableStateFlow(SensorData())
    val gyroscopeData: StateFlow<SensorData> = _gyroscopeData.asStateFlow()
    
    private val _magnetometerData = MutableStateFlow(SensorData())
    val magnetometerData: StateFlow<SensorData> = _magnetometerData.asStateFlow()
    
    // Recent foreground apps (max 10).
    private val _recentApps = MutableStateFlow<List<RecentApp>>(emptyList())
    val recentApps: StateFlow<List<RecentApp>> = _recentApps.asStateFlow()
    
    // Sensor state
    private val _sensorsActive = MutableLiveData(false)
    val sensorsActive: LiveData<Boolean> = _sensorsActive
    
    // Job management
    private var sensorCollectionJob: Job? = null
    private var appDetectionJob: Job? = null
    
    // Chart data cache (used for chart display)
    private val _chartAccelerometerData = MutableStateFlow(SensorData())
    val chartAccelerometerData: StateFlow<SensorData> = _chartAccelerometerData.asStateFlow()
    
    private val _chartGyroscopeData = MutableStateFlow(SensorData())
    val chartGyroscopeData: StateFlow<SensorData> = _chartGyroscopeData.asStateFlow()
    
    private val _chartMagnetometerData = MutableStateFlow(SensorData())
    val chartMagnetometerData: StateFlow<SensorData> = _chartMagnetometerData.asStateFlow()
    
    private var isChartActive = false
    
    /**
     * Starts collecting sensor data.
     */
    fun startSensorCollection() {
        // Cancel any previous jobs.
        stopSensorCollection()
        
        viewModelScope.launch {
            // Start sensor collection.
            sensorCollector.startCollection()
            _sensorsActive.value = true
            
            // Subscribe to sensor data stream (use update for thread safety).
            sensorCollectionJob = launch {
                sensorCollector.getSensorDataFlow().collect { sample ->
                    val sensorData = SensorData(
                        x = sample.x,
                        y = sample.y,
                        z = sample.z,
                        timestamp = System.currentTimeMillis() // Use wall time for UI display
                    )
                    
                    when (sample.type) {
                        SensorType.ACCELEROMETER -> {
                            _accelerometerData.update { sensorData }
                            if (isChartActive) {
                                _chartAccelerometerData.update { sensorData }
                            }
                        }
                        SensorType.GYROSCOPE -> {
                            _gyroscopeData.update { sensorData }
                            if (isChartActive) {
                                _chartGyroscopeData.update { sensorData }
                            }
                        }
                        SensorType.MAGNETOMETER -> {
                            _magnetometerData.update { sensorData }
                            if (isChartActive) {
                                _chartMagnetometerData.update { sensorData }
                            }
                        }
                    }
                }
            }

            // Periodically poll the foreground app (performance-friendly).
            appDetectionJob = launch {
                while (_sensorsActive.value == true) {
                    try {
                        val currentApp = foregroundAppDetector.getCurrentForegroundApp()
                        if (currentApp.isNotEmpty()) {
                            // Derive a friendly app name (from package name).
                            val appName = getAppNameFromPackage(currentApp)

                            // Avoid adding duplicates back-to-back.
                            val currentList = _recentApps.value
                            if (currentList.isEmpty() || currentList.last().packageName != currentApp) {
                                addRecentApp(currentApp, appName)
                            }
                        }
                    } catch (e: Exception) {
                        // Ignore and keep polling.
                    }
                    delay(1000) // Poll every second
                }
            }

        }
    }
    /**
     * Updates the recent apps list, ensuring uniqueness and correct ordering.
     */
    private fun updateRecentAppsList(newApp: RecentApp) {
        val currentList = _recentApps.value.toMutableList()

        // Remove old entry for the same package.
        currentList.removeAll { it.packageName == newApp.packageName }

        // Append new record.
        currentList.add(newApp)

        // Keep only the latest 10 apps.
        while (currentList.size > 10) {
            currentList.removeAt(0)
        }

        // Publish updated list.
        _recentApps.value = currentList
    }


    /**
     * Stops sensor data collection.
     */
    fun stopSensorCollection() {
        // Cancel jobs.
        sensorCollectionJob?.cancel()
        appDetectionJob?.cancel()
        sensorCollectionJob = null
        appDetectionJob = null
        
        viewModelScope.launch {
            sensorCollector.stopCollection()
            _sensorsActive.value = false
        }
    }
    
    /**
     * Derives a friendly app name from a package name.
     */
    private fun getAppNameFromPackage(packageName: String): String {
        return when {
            packageName.contains("chrome") -> "Chrome"
            packageName.contains("whatsapp") -> "WhatsApp"
            packageName.contains("spotify") -> "Spotify"
            packageName.contains("instagram") -> "Instagram"
            packageName.contains("youtube") -> "YouTube"
            packageName.contains("facebook") -> "Facebook"
            packageName.contains("twitter") -> "Twitter"
            packageName.contains("telegram") -> "Telegram"
            packageName.contains("gmail") -> "Gmail"
            packageName.contains("maps") -> "Maps"
            packageName.contains("camera") -> "Camera"
            packageName.contains("gallery") -> "Gallery"
            packageName.contains("settings") -> "Settings"
            packageName.contains("phone") -> "Phone"
            packageName.contains("messages") -> "Messages"
            packageName.contains("calculator") -> "Calculator"
            packageName.contains("calendar") -> "Calendar"
            packageName.contains("clock") -> "Clock"
            else -> packageName.split(".").lastOrNull()?.replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.getDefault()) else it.toString() } ?: packageName
        }
    }
    
    /**
     * Resets chart data (called when the chart is opened).
     */
    fun resetChartData() {
        isChartActive = true
        _chartAccelerometerData.value = SensorData()
        _chartGyroscopeData.value = SensorData()
        _chartMagnetometerData.value = SensorData()
    }
    
    /**
     * Stops chart data updates (called when the chart is closed).
     */
    fun stopChartData() {
        isChartActive = false
    }
    
    override fun onCleared() {
        super.onCleared()
        stopSensorCollection()
    }
    /**
     * Adds a recently used app entry, ensuring uniqueness and ordering.
     */
    private fun addRecentApp(packageName: String, appName: String) {
        val currentTime = System.currentTimeMillis()
        val newApp = RecentApp(
            packageName = packageName,
            appName = appName,
            timestamp = currentTime
        )

        // Update recent app list.
        updateRecentAppsList(newApp)
    }

}
