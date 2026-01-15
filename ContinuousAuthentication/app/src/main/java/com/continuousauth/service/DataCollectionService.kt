package com.continuousauth.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import com.continuousauth.R
import com.continuousauth.core.SmartTransmissionManager
import com.continuousauth.privacy.PrivacyManager
import com.continuousauth.ui.MainActivity
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.*
import android.util.Log
import javax.inject.Inject

/**
 * Foreground data collection service.
 *
 * Collects sensor data continuously in the background.
 *
 * Important: this service must only start after the user has granted privacy consent.
 */
@AndroidEntryPoint
class DataCollectionService : Service() {
    
    companion object {
        private const val TAG = "DataCollectionService"
        private const val NOTIFICATION_CHANNEL_ID = "continuous_auth_service"
        private const val NOTIFICATION_ID = 1001
        
        // Intent actions
        const val ACTION_START_COLLECTION = "com.continuousauth.START_COLLECTION"
        const val ACTION_STOP_COLLECTION = "com.continuousauth.STOP_COLLECTION"
        const val ACTION_PAUSE_COLLECTION = "com.continuousauth.PAUSE_COLLECTION"
        const val ACTION_RESUME_COLLECTION = "com.continuousauth.RESUME_COLLECTION"
    }
    
    @Inject
    lateinit var privacyManager: PrivacyManager
    
    @Inject
    lateinit var smartTransmissionManager: SmartTransmissionManager
    
    // Coroutine scope
    private val serviceScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    
    // Wake lock to keep the device partially awake
    private var wakeLock: PowerManager.WakeLock? = null
    
    // Service state
    private var isCollecting = false
    private var isPaused = false
    
    // Receives system events
    private val systemEventReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                Intent.ACTION_BATTERY_LOW -> {
                    Log.w(TAG, "Battery low; pausing data collection")
                    pauseCollection()
                }
                Intent.ACTION_BATTERY_OKAY -> {
                    Log.i(TAG, "Battery OK; resuming data collection")
                    resumeCollection()
                }
                Intent.ACTION_POWER_CONNECTED -> {
                    Log.i(TAG, "Power connected")
                    updateCollectionStrategy(isCharging = true)
                }
                Intent.ACTION_POWER_DISCONNECTED -> {
                    Log.i(TAG, "Power disconnected")
                    updateCollectionStrategy(isCharging = false)
                }
            }
        }
    }
    
    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service created")
        
        // Check privacy consent state
        if (!checkPrivacyConsent()) {
            Log.w(TAG, "Privacy consent not granted; service will not start")
            stopSelf()
            return
        }
        
        // Create notification channel
        createNotificationChannel()
        
        // Register system event receivers
        registerSystemEventReceivers()
        
        // Acquire wake lock
        acquireWakeLock()
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "onStartCommand action: ${intent?.action}")
        
        // Re-check privacy consent
        if (!checkPrivacyConsent()) {
            Log.w(TAG, "Privacy consent not granted; stopping service")
            stopSelf()
            return START_NOT_STICKY
        }
        
        // Start in the foreground
        val notification = createNotification()
        startForeground(NOTIFICATION_ID, notification)
        
        // Handle commands
        when (intent?.action) {
            ACTION_START_COLLECTION -> startCollection()
            ACTION_STOP_COLLECTION -> stopCollection()
            ACTION_PAUSE_COLLECTION -> pauseCollection()
            ACTION_RESUME_COLLECTION -> resumeCollection()
            else -> {
                // Default behavior: start collection
                if (!isCollecting) {
                    startCollection()
                }
            }
        }
        
        return START_STICKY
    }
    
    override fun onBind(intent: Intent?): IBinder? {
        return null
    }
    
    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "Service destroyed")
        
        // Stop data collection
        stopCollection()
        
        // Cancel coroutines
        serviceScope.cancel()
        
        // Unregister receivers
        unregisterSystemEventReceivers()
        
        // Release wake lock
        releaseWakeLock()
    }
    
    /**
     * Check whether the user has granted privacy consent.
     */
    private fun checkPrivacyConsent(): Boolean {
        // Check consent flag from SharedPreferences
        val prefs = getSharedPreferences("app_prefs", MODE_PRIVATE)
        val hasAgreed = prefs.getBoolean("privacy_agreement_shown", false)
        
        if (!hasAgreed) {
            Log.w(TAG, "Privacy consent not granted; blocking service start")
        }
        
        return hasAgreed
    }
    
    /**
     * Create the notification channel.
     */
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                getString(R.string.app_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Continuous authentication data collection service"
                setShowBadge(false)
            }
            
            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager?.createNotificationChannel(channel)
        }
    }
    
    /**
     * Create the foreground service notification.
     */
    private fun createNotification(): Notification {
        // Tap notification to open main activity
        val notificationIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, notificationIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        
        // Action: stop
        val stopIntent = Intent(this, DataCollectionService::class.java).apply {
            action = ACTION_STOP_COLLECTION
        }
        val stopPendingIntent = PendingIntent.getService(
            this, 1, stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        
        // Action: pause/resume
        val pauseResumeIntent = Intent(this, DataCollectionService::class.java).apply {
            action = if (isPaused) ACTION_RESUME_COLLECTION else ACTION_PAUSE_COLLECTION
        }
        val pauseResumePendingIntent = PendingIntent.getService(
            this, 2, pauseResumeIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        
        val statusText = when {
            isPaused -> "Data collection paused"
            isCollecting -> "Data collection running"
            else -> "Data collection ready"
        }
        
        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(statusText)
            .setSmallIcon(R.drawable.ic_notification)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .setContentIntent(pendingIntent)
            .addAction(
                android.R.drawable.ic_media_pause,  // Use system drawable
                if (isPaused) "Resume" else "Pause",
                pauseResumePendingIntent
            )
            .addAction(
                android.R.drawable.ic_delete,  // Use system drawable
                "Stop",
                stopPendingIntent
            )
            .build()
    }
    
    /**
     * Start data collection.
     */
    private fun startCollection() {
        if (isCollecting) {
            Log.w(TAG, "Data collection is already running")
            return
        }
        
        Log.i(TAG, "Starting data collection")
        isCollecting = true
        isPaused = false
        
        // Start SmartTransmissionManager in a coroutine
        serviceScope.launch {
            try {
                smartTransmissionManager.start()
                updateNotification()
                Log.i(TAG, "Data collection started")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start data collection", e)
                isCollecting = false
                updateNotification()
            }
        }
    }
    
    /**
     * Stop data collection.
     */
    private fun stopCollection() {
        if (!isCollecting) {
            Log.w(TAG, "Data collection is not running")
            return
        }
        
        Log.i(TAG, "Stopping data collection")
        isCollecting = false
        isPaused = false
        
        serviceScope.launch {
            try {
                smartTransmissionManager.stop()
                updateNotification()
                Log.i(TAG, "Data collection stopped")
                
                // Stop service
                stopForeground(true)
                stopSelf()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to stop data collection", e)
            }
        }
    }
    
    /**
     * Pause data collection.
     */
    private fun pauseCollection() {
        if (!isCollecting || isPaused) {
            return
        }
        
        Log.i(TAG, "Pausing data collection")
        isPaused = true
        
        serviceScope.launch {
            try {
                smartTransmissionManager.pause()
                updateNotification()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to pause data collection", e)
            }
        }
    }
    
    /**
     * Resume data collection.
     */
    private fun resumeCollection() {
        if (!isCollecting || !isPaused) {
            return
        }
        
        Log.i(TAG, "Resuming data collection")
        isPaused = false
        
        serviceScope.launch {
            try {
                smartTransmissionManager.resume()
                updateNotification()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to resume data collection", e)
            }
        }
    }
    
    /**
     * Update notification.
     */
    private fun updateNotification() {
        val notification = createNotification()
        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager?.notify(NOTIFICATION_ID, notification)
    }
    
    /**
     * Register system event receivers.
     */
    private fun registerSystemEventReceivers() {
        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_BATTERY_LOW)
            addAction(Intent.ACTION_BATTERY_OKAY)
            addAction(Intent.ACTION_POWER_CONNECTED)
            addAction(Intent.ACTION_POWER_DISCONNECTED)
        }
        registerReceiver(systemEventReceiver, filter)
    }
    
    /**
     * Unregister system event receivers.
     */
    private fun unregisterSystemEventReceivers() {
        try {
            unregisterReceiver(systemEventReceiver)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to unregister receiver", e)
        }
    }
    
    /**
     * Acquire wake lock.
     */
    private fun acquireWakeLock() {
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "$packageName:DataCollectionWakeLock"
        ).apply {
            acquire(10 * 60 * 1000L) // max 10 minutes
        }
        Log.d(TAG, "Wake lock acquired")
    }
    
    /**
     * Release wake lock.
     */
    private fun releaseWakeLock() {
        wakeLock?.let {
            if (it.isHeld) {
                it.release()
                Log.d(TAG, "Wake lock released")
            }
        }
        wakeLock = null
    }
    
    /**
     * Update collection strategy.
     */
    private fun updateCollectionStrategy(isCharging: Boolean) {
        serviceScope.launch {
            if (isCharging) {
                // While charging, use a more aggressive collection strategy.
                smartTransmissionManager.updatePolicy(
                    batchInterval = 500L, // 500ms batch interval
                    compressionEnabled = false // avoid compression to save CPU
                )
            } else {
                // On battery, use a power-saving strategy.
                smartTransmissionManager.updatePolicy(
                    batchInterval = 2000L, // 2s batch interval
                    compressionEnabled = true // enable compression to reduce upload size
                )
            }
        }
    }
}
