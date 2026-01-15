package com.continuousauth.detection

import android.app.usage.UsageStats
import android.app.usage.UsageStatsManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.concurrent.ConcurrentLinkedQueue
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.sqrt

/**
 * Lightweight on-device anomaly detector implementation.
 *
 * Detects device unlocks, accelerometer spikes, and sensitive-app entry events.
 */
@Singleton
class AnomalyDetectorImpl @Inject constructor(
    @ApplicationContext private val context: Context
) : AnomalyDetector {
    
    companion object {
        private const val TAG = "AnomalyDetector"
    }
    
    // Detection state.
    private var isDetecting = false
    private var detectionJob: Job? = null
    private val detectorScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    
    // Listener.
    private var anomalyListener: OnAnomalyListener? = null
    
    // Current policy.
    private var currentPolicy = DetectionPolicy()
    
    // Device unlock monitoring.
    private var unlockReceiver: BroadcastReceiver? = null
    private var lastUnlockTime = 0L
    
    // Accelerometer processing via a sliding window.
    private val accelerometerData = ConcurrentLinkedQueue<AccelerometerSample>()
    private var lastAccelerometerTrigger = 0L
    
    // Short/long-term window statistics (Epic 4.1.1).
    private val shortTermWindow = mutableListOf<Float>()
    private val longTermWindow = mutableListOf<Float>()
    private var adaptiveThreshold = 2.5f // Adaptive threshold (initially 2.5 standard deviations).
    
    // Percentile method configuration.
    private val percentileThreshold = 95 // Use the 95th percentile as the anomaly threshold.
    
    // Cooldown with progressive backoff (Epic 4.1.1).
    private var cooldownEndTime = 0L
    private var cooldownCount = 0
    private val maxCooldownCount = 5 // Max cooldown escalation steps.
    private val cooldownBackoffFactor = 1.5 // Backoff factor.
    
    // Foreground app monitoring.
    private var lastForegroundApp: String? = null
    private var usageStatsManager: UsageStatsManager? = null
    
    init {
        // Initialize system services.
        usageStatsManager = context.getSystemService(Context.USAGE_STATS_SERVICE) as? UsageStatsManager
    }
    
    override suspend fun startDetection() {
        if (isDetecting) return
        
        Log.i(TAG, "Starting anomaly detection")
        isDetecting = true
        
        // Register unlock listener.
        if (currentPolicy.deviceUnlockEnabled) {
            registerUnlockReceiver()
        }
        
        // Start foreground app monitoring.
        startForegroundAppMonitoring()
        
        Log.i(TAG, "Anomaly detection started")
    }
    
    override suspend fun stopDetection() {
        if (!isDetecting) return
        
        Log.i(TAG, "Stopping anomaly detection")
        isDetecting = false
        
        // Stop monitoring jobs.
        detectionJob?.cancel()
        detectionJob = null
        
        // Unregister unlock listener.
        unregisterUnlockReceiver()
        
        // Clear state.
        accelerometerData.clear()
        
        Log.i(TAG, "Anomaly detection stopped")
    }
    
    override fun setOnAnomalyListener(listener: OnAnomalyListener?) {
        this.anomalyListener = listener
    }
    
    override suspend fun updatePolicy(config: DetectionPolicy) {
        Log.d(TAG, "Updating detection policy")
        val oldPolicy = currentPolicy
        currentPolicy = config
        
        // Re-apply settings if detection is running.
        if (isDetecting) {
            // Re-register unlock receiver if the setting changed.
            if (oldPolicy.deviceUnlockEnabled != config.deviceUnlockEnabled) {
                if (config.deviceUnlockEnabled) {
                    registerUnlockReceiver()
                } else {
                    unregisterUnlockReceiver()
                }
            }
        }
    }
    
    override fun processSensorData(x: Float, y: Float, z: Float, timestamp: Long) {
        if (!isDetecting || !currentPolicy.enabled) return
        
        // Acceleration magnitude.
        val magnitude = sqrt(x * x + y * y + z * z)
        val sample = AccelerometerSample(magnitude, timestamp)
        
        // Add to sample queue.
        accelerometerData.offer(sample)
        
        // Keep window size.
        while (accelerometerData.size > currentPolicy.accelerometerWindowSize) {
            accelerometerData.poll()
        }
        
        // Spike detection.
        checkAccelerometerSpike(magnitude, timestamp)
    }
    
    override fun isDetecting(): Boolean = isDetecting
    
    /**
     * Registers the device unlock receiver.
     */
    private fun registerUnlockReceiver() {
        if (unlockReceiver != null) return
        
        unlockReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                if (intent?.action == Intent.ACTION_USER_PRESENT) {
                    handleDeviceUnlock()
                }
            }
        }
        
        val filter = IntentFilter(Intent.ACTION_USER_PRESENT)
        context.registerReceiver(unlockReceiver, filter)
        Log.d(TAG, "Device unlock receiver registered")
    }
    
    /**
     * Unregisters the device unlock receiver.
     */
    private fun unregisterUnlockReceiver() {
        unlockReceiver?.let { receiver ->
            try {
                context.unregisterReceiver(receiver)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to unregister device unlock receiver", e)
            }
            unlockReceiver = null
            Log.d(TAG, "Device unlock receiver unregistered")
        }
    }
    
    /**
     * Handles device unlock events.
     */
    private fun handleDeviceUnlock() {
        val currentTime = System.currentTimeMillis()
        
        // Cooldown.
        if (currentTime - lastUnlockTime < currentPolicy.deviceUnlockCooldownMs) {
            return
        }
        
        lastUnlockTime = currentTime
        
        if (currentPolicy.debugMode) {
            Log.d(TAG, "Device unlock event detected")
        }
        
        // Notify listener.
        anomalyListener?.onAnomalyDetected(AnomalyTrigger.DeviceUnlocked)
    }
    
    /**
     * Starts foreground app monitoring.
     */
    private fun startForegroundAppMonitoring() {
        if (currentPolicy.sensitiveApps.isEmpty()) {
            return
        }
        
        detectionJob = detectorScope.launch {
            while (isActive && isDetecting) {
                try {
                    checkForegroundApp()
                } catch (e: Exception) {
                    Log.w(TAG, "Foreground app check failed", e)
                }
                delay(currentPolicy.appCheckIntervalMs)
            }
        }
        
        Log.d(TAG, "Foreground app monitoring started")
    }
    
    /**
     * Checks the current foreground app.
     */
    @RequiresApi(Build.VERSION_CODES.LOLLIPOP_MR1)
    private fun checkForegroundApp() {
        val usageStats = usageStatsManager ?: return
        
        try {
            val currentTime = System.currentTimeMillis()
            // Use a wider query window (5 minutes) to improve the chance of receiving events.
            val startTime = currentTime - 5 * 60 * 1000L
            
            // Use different APIs depending on Android version.
            val currentForegroundApp = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                // Android Q+ uses queryEvents.
                getForegroundAppUsingEvents(usageStats, startTime, currentTime)
            } else {
                // Older versions use queryUsageStats.
                getForegroundAppUsingStats(usageStats, startTime, currentTime)
            }
            
            if (currentForegroundApp != null && 
                currentForegroundApp != lastForegroundApp &&
                currentPolicy.sensitiveApps.contains(currentForegroundApp)) {
                
                handleSensitiveAppEntered(currentForegroundApp)
            }
            
            lastForegroundApp = currentForegroundApp
            
        } catch (e: Exception) {
            Log.w(TAG, "Failed to resolve foreground app", e)
        }
    }
    
    /**
     * Resolves the foreground app using UsageEvents (Android Q+).
     */
    @RequiresApi(Build.VERSION_CODES.Q)
    private fun getForegroundAppUsingEvents(
        usageStats: UsageStatsManager,
        startTime: Long,
        endTime: Long
    ): String? {
        val events = usageStats.queryEvents(startTime, endTime)
        val event = android.app.usage.UsageEvents.Event()
        
        var lastForegroundPackage: String? = null
        var lastForegroundTime = 0L
        
        while (events.hasNextEvent()) {
            events.getNextEvent(event)
            
            // Look for ACTIVITY_RESUMED/MOVE_TO_FOREGROUND events.
            if (event.eventType == android.app.usage.UsageEvents.Event.ACTIVITY_RESUMED ||
                event.eventType == android.app.usage.UsageEvents.Event.MOVE_TO_FOREGROUND) {
                
                if (event.timeStamp > lastForegroundTime) {
                    lastForegroundTime = event.timeStamp
                    lastForegroundPackage = event.packageName
                }
            }
        }
        
        return lastForegroundPackage
    }
    
    /**
     * Resolves the foreground app using UsageStats (older Android versions).
     */
    private fun getForegroundAppUsingStats(
        usageStats: UsageStatsManager,
        startTime: Long,
        endTime: Long
    ): String? {
        val stats = usageStats.queryUsageStats(
            UsageStatsManager.INTERVAL_BEST,
            startTime,
            endTime
        )
        
        // Find the most recently used app.
        return stats?.filter { 
            it.totalTimeInForeground > 0 && 
            it.lastTimeUsed >= startTime 
        }?.maxByOrNull { it.lastTimeUsed }?.packageName
    }
    
    /**
     * Handles sensitive-app entry events.
     */
    private fun handleSensitiveAppEntered(packageName: String) {
        if (currentPolicy.debugMode) {
            Log.d(TAG, "Sensitive app entered: $packageName")
        }
        
        // Notify listener.
        val trigger = AnomalyTrigger.SensitiveAppEntered(packageName)
        anomalyListener?.onAnomalyDetected(trigger)
    }
    
    /**
     * Checks for accelerometer spikes using short-term mean/variance and a percentile method.
     *
     * Epic 4.1.1: adaptive thresholds and progressive backoff.
     */
    private fun checkAccelerometerSpike(magnitude: Float, timestamp: Long) {
        // Update sliding windows.
        updateSlidingWindows(magnitude)
        
        if (shortTermWindow.size < 10 || longTermWindow.size < 50) {
            return // Not enough data; skip detection.
        }
        
        // Cooldown period check (with progressive backoff).
        if (isInCooldownPeriod(timestamp)) {
            return
        }
        
        // Two detection methods.
        val isAnomalyByStats = checkByStatisticalMethod(magnitude)
        val isAnomalyByPercentile = checkByPercentileMethod(magnitude)
        
        // Trigger if either method flags an anomaly.
        if (isAnomalyByStats || isAnomalyByPercentile) {
            handleAnomalyDetection(magnitude, timestamp)
        } else {
            // No anomaly detected; gradually decrease the cooldown count.
            if (cooldownCount > 0 && System.currentTimeMillis() > cooldownEndTime) {
                cooldownCount--
            }
        }
    }
    
    /**
     * Updates sliding windows.
     */
    private fun updateSlidingWindows(magnitude: Float) {
        // Short-term window (last 20 samples).
        shortTermWindow.add(magnitude)
        if (shortTermWindow.size > 20) {
            shortTermWindow.removeAt(0)
        }
        
        // Long-term window (last 100 samples).
        longTermWindow.add(magnitude)
        if (longTermWindow.size > 100) {
            longTermWindow.removeAt(0)
        }
    }
    
    /**
     * Statistical anomaly detection (mean/variance).
     */
    private fun checkByStatisticalMethod(magnitude: Float): Boolean {
        // Baseline statistics from the long-term window.
        val baselineMean = longTermWindow.average().toFloat()
        val baselineVariance = longTermWindow.map { (it - baselineMean) * (it - baselineMean) }.average().toFloat()
        val baselineStdDev = sqrt(baselineVariance)
        
        // Current statistics from the short-term window.
        val currentMean = shortTermWindow.average().toFloat()
        val currentVariance = shortTermWindow.map { (it - currentMean) * (it - currentMean) }.average().toFloat()
        val currentStdDev = sqrt(currentVariance)
        
        // Adapt threshold based on cooldown frequency.
        adaptiveThreshold = when {
            cooldownCount > 3 -> 3.5f // Raise threshold when triggering frequently.
            cooldownCount > 1 -> 3.0f
            else -> 2.5f // Normal threshold.
        }
        
        // Detect deviation relative to long-term baseline.
        val deviation = if (baselineStdDev > 0.1f) {
            (magnitude - baselineMean) / baselineStdDev
        } else {
            0f
        }
        
        return deviation > adaptiveThreshold && currentStdDev > baselineStdDev * 1.5f
    }
    
    /**
     * Percentile-based anomaly detection.
     */
    private fun checkByPercentileMethod(magnitude: Float): Boolean {
        // Sort long-term window to compute percentile.
        val sortedValues = longTermWindow.sorted()
        val percentileIndex = (sortedValues.size * percentileThreshold / 100).coerceIn(0, sortedValues.size - 1)
        val percentileValue = sortedValues[percentileIndex]
        
        // Dynamic multiplier (accounts for cooldown state).
        val dynamicMultiplier = if (cooldownCount > 0) {
            1.2f + (cooldownCount * 0.1f) // Raise threshold during cooldown.
        } else {
            1.1f
        }
        
        return magnitude > percentileValue * dynamicMultiplier
    }
    
    /**
     * Returns true if currently in the cooldown period (progressive backoff).
     */
    private fun isInCooldownPeriod(timestamp: Long): Boolean {
        if (cooldownEndTime > timestamp) {
            if (currentPolicy.debugMode) {
                Log.d(TAG, "In cooldown period, remaining: ${(cooldownEndTime - timestamp) / 1000}s")
            }
            return true
        }
        return false
    }
    
    /**
     * Handles anomaly detection results.
     */
    private fun handleAnomalyDetection(magnitude: Float, timestamp: Long) {
        lastAccelerometerTrigger = timestamp
        
        // Update cooldown (progressive backoff).
        cooldownCount = (cooldownCount + 1).coerceAtMost(maxCooldownCount)
        val cooldownDuration = (currentPolicy.accelerometerCooldownMs * 
                               Math.pow(cooldownBackoffFactor.toDouble(), cooldownCount.toDouble())).toLong()
        cooldownEndTime = timestamp + cooldownDuration
        
        // Compute stats for logging.
        val mean = longTermWindow.average().toFloat()
        val stdDev = sqrt(longTermWindow.map { (it - mean) * (it - mean) }.average().toFloat())
        val threshold = mean + adaptiveThreshold * stdDev
        val deviation = if (stdDev > 0) (magnitude - mean) / stdDev else 0f
        
        if (currentPolicy.debugMode) {
            Log.d(TAG, "Accelerometer spike detected: magnitude=$magnitude, threshold=$threshold, " +
                      "deviation=$deviation, cooldown=${cooldownDuration/1000}s")
        }
        
        // Notify listener.
        val trigger = AnomalyTrigger.AccelerometerSpike(magnitude, threshold, deviation)
        anomalyListener?.onAnomalyDetected(trigger)
    }
    
    /**
     * Releases resources.
     */
    fun cleanup() {
        detectorScope.launch {
            stopDetection()
        }
        detectorScope.cancel()
    }
}

/**
 * Accelerometer sample.
 */
private data class AccelerometerSample(
    val magnitude: Float,
    val timestamp: Long
)
