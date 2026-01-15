package com.continuousauth.crypto

import kotlinx.coroutines.*
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.ScheduledThreadPoolExecutor
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Key rotation manager.
 *
 * Schedules periodic checks and rotates keys to improve forward secrecy.
 */
@Singleton
class KeyRotationManager @Inject constructor(
    private val cryptoBox: CryptoBox
) {
    
    companion object {
        private const val ROTATION_INTERVAL_HOURS = 24L
        private const val INITIAL_DELAY_MINUTES = 5L // First check after 5 minutes
        private const val TAG = "KeyRotationManager"
    }
    
    private var scheduledExecutor: ScheduledExecutorService? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var isRunning = false
    
    /**
     * Start automatic key rotation checks.
     */
    fun startAutoRotation() {
        if (isRunning) {
            android.util.Log.w(TAG, "Key rotation is already running")
            return
        }
        
        scheduledExecutor = ScheduledThreadPoolExecutor(1).apply {
            // Use a daemon thread with low priority
            threadFactory = java.util.concurrent.ThreadFactory { runnable ->
                Thread(runnable, "KeyRotation").apply {
                    isDaemon = true
                    priority = Thread.MIN_PRIORITY
                }
            }
        }
        
        // Schedule periodic checks: initial delay then every 24 hours.
        scheduledExecutor?.scheduleWithFixedDelay(
            { performRotationCheck() },
            INITIAL_DELAY_MINUTES,
            ROTATION_INTERVAL_HOURS * 60, // Convert to minutes
            TimeUnit.MINUTES
        )
        
        isRunning = true
        android.util.Log.i(TAG, "Automatic key rotation started - interval: ${ROTATION_INTERVAL_HOURS}h")
    }
    
    /**
     * Stop automatic key rotation.
     */
    fun stopAutoRotation() {
        if (!isRunning) {
            return
        }
        
        scheduledExecutor?.shutdown()
        try {
            if (scheduledExecutor?.awaitTermination(5, TimeUnit.SECONDS) == false) {
                scheduledExecutor?.shutdownNow()
            }
        } catch (e: InterruptedException) {
            scheduledExecutor?.shutdownNow()
            Thread.currentThread().interrupt()
        }
        
        scheduledExecutor = null
        isRunning = false
        
        android.util.Log.i(TAG, "Automatic key rotation stopped")
    }
    
    /**
     * Trigger key rotation manually.
     */
    suspend fun rotateNow(): Boolean {
        return withContext(Dispatchers.IO) {
            performKeyRotation()
        }
    }
    
    /**
     * Get rotation status.
     */
    fun getRotationStatus(): RotationStatus {
        val keyInfo = cryptoBox.getKeyInfo()
        return RotationStatus(
            isAutoRotationEnabled = isRunning,
            lastRotationTime = keyInfo.keyCreationTime,
            rotationCount = keyInfo.keyRotationCount,
            nextRotationTime = calculateNextRotationTime(keyInfo.keyCreationTime)
        )
    }
    
    /**
     * Perform rotation check (invoked by the scheduler).
     */
    private fun performRotationCheck() {
        scope.launch {
            try {
                val keyInfo = cryptoBox.getKeyInfo()
                val currentTime = System.currentTimeMillis()
                val timeSinceLastRotation = currentTime - keyInfo.keyCreationTime
                
                // Rotate if the interval has elapsed.
                if (timeSinceLastRotation >= TimeUnit.HOURS.toMillis(ROTATION_INTERVAL_HOURS)) {
                    android.util.Log.i(TAG, "Scheduled key rotation triggered")
                    performKeyRotation()
                } else {
                    android.util.Log.d(TAG, "Key rotation check complete - no rotation needed")
                }
            } catch (e: Exception) {
                android.util.Log.e(TAG, "Key rotation check failed", e)
            }
        }
    }
    
    /**
     * Perform key rotation.
     */
    private suspend fun performKeyRotation(): Boolean {
        return try {
            android.util.Log.i(TAG, "Starting key rotation")
            
            val success = cryptoBox.rotateSessionKey()
            
            if (success) {
                val keyInfo = cryptoBox.getKeyInfo()
                android.util.Log.i(TAG, "Key rotation succeeded - count: ${keyInfo.keyRotationCount}")
                
                // Hook for success notifications.
                onRotationSuccess(keyInfo.keyRotationCount)
                
                true
            } else {
                android.util.Log.e(TAG, "Key rotation failed")
                
                // Hook for failure notifications.
                onRotationFailure("CryptoBox rotation failed")
                
                false
            }
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Key rotation exception", e)
            onRotationFailure("Rotation exception: ${e.message}")
            false
        }
    }
    
    /**
     * Calculate next rotation time.
     */
    private fun calculateNextRotationTime(lastRotationTime: Long): Long {
        return lastRotationTime + TimeUnit.HOURS.toMillis(ROTATION_INTERVAL_HOURS)
    }
    
    /**
     * Rotation success callback.
     */
    private fun onRotationSuccess(rotationCount: Long) {
        // Add success notification/logging here if needed.
        android.util.Log.i(TAG, "Key rotation success callback - total rotations: $rotationCount")
    }
    
    /**
     * Rotation failure callback.
     */
    private fun onRotationFailure(reason: String) {
        // Add retry logic or failure notification here if needed.
        android.util.Log.w(TAG, "Key rotation failure callback - reason: $reason")
    }
}

/**
 * Rotation status snapshot.
 */
data class RotationStatus(
    val isAutoRotationEnabled: Boolean,    // Auto-rotation enabled
    val lastRotationTime: Long,           // Last rotation time
    val rotationCount: Long,              // Rotation count
    val nextRotationTime: Long            // Next rotation time
)
