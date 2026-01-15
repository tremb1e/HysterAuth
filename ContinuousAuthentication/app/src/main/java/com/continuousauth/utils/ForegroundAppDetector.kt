package com.continuousauth.utils

import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import android.content.pm.PackageManager
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Foreground app detector.
 *
 * Uses [UsageStatsManager] to track foreground app changes.
 */
@Singleton
class ForegroundAppDetector @Inject constructor(
    @ApplicationContext private val context: Context
) {
    
    private val usageStatsManager = context.getSystemService(Context.USAGE_STATS_SERVICE) as? UsageStatsManager
    private val packageManager = context.packageManager
    
    private var currentForegroundApp = ""
    private val mutex = Mutex()
    
    /**
     * Get the current foreground app package name.
     *
     * Returns the cached value when permission is missing or detection fails.
     */
    suspend fun getCurrentForegroundApp(): String = mutex.withLock {
        try {
            usageStatsManager?.let { statsManager ->
                val currentTime = System.currentTimeMillis()
                // Use a 60s window to ensure foreground events are captured.
                val events = statsManager.queryEvents(currentTime - 60000, currentTime)
                
                var lastForegroundPackage: String? = null
                var lastForegroundTime = 0L
                val event = UsageEvents.Event()
                
                // Iterate through all events
                while (events.hasNextEvent()) {
                    events.getNextEvent(event)
                    
                    // Consider multiple foreground-related event types.
                    when (event.eventType) {
                        UsageEvents.Event.ACTIVITY_RESUMED,
                        UsageEvents.Event.MOVE_TO_FOREGROUND -> {
                            // Track the most recent foreground app.
                            if (event.timeStamp > lastForegroundTime) {
                                lastForegroundTime = event.timeStamp
                                lastForegroundPackage = event.packageName
                            }
                        }
                        UsageEvents.Event.ACTIVITY_PAUSED,
                        UsageEvents.Event.MOVE_TO_BACKGROUND -> {
                            if (event.packageName == currentForegroundApp && 
                                event.timeStamp > lastForegroundTime) {
                                // Don't clear immediately; wait for the next foreground app.
                                lastForegroundTime = event.timeStamp
                            }
                        }
                    }
                }
                
                // If a new foreground app is found, update cache and return it.
                lastForegroundPackage?.let { packageName ->
                    // Filter system apps and launchers
                    if (!isSystemOrLauncherApp(packageName)) {
                        currentForegroundApp = packageName
                        android.util.Log.d("ForegroundAppDetector", "Detected foreground app: $packageName")
                        return currentForegroundApp
                    }
                }
                
                // If no new foreground app is found, return the cached value.
                if (currentForegroundApp.isNotEmpty()) {
                    return currentForegroundApp
                }
            }
        } catch (e: SecurityException) {
            // Missing PACKAGE_USAGE_STATS permission.
            android.util.Log.w("ForegroundAppDetector", "Missing PACKAGE_USAGE_STATS permission", e)
        } catch (e: Exception) {
            android.util.Log.e("ForegroundAppDetector", "Failed to get foreground app", e)
        }
        
        return currentForegroundApp // Return cached value
    }
    
    /**
     * Returns whether a package is a system app or launcher.
     */
    private fun isSystemOrLauncherApp(packageName: String): Boolean {
        return packageName.startsWith("com.android.") ||
               packageName == "android" ||
               packageName.contains("launcher") ||
               packageName.contains("systemui")
    }
    
    /**
     * Check whether usage stats permission is granted.
     */
    fun hasUsageStatsPermission(): Boolean {
        return try {
            val usageStatsManager = context.getSystemService(Context.USAGE_STATS_SERVICE) as? UsageStatsManager
            val currentTime = System.currentTimeMillis()
            val queryUsageStats = usageStatsManager?.queryUsageStats(
                UsageStatsManager.INTERVAL_DAILY,
                currentTime - 1000 * 60 * 60,
                currentTime
            )
            
            queryUsageStats?.isNotEmpty() ?: false
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Get application label (useful for debugging).
     */
    fun getAppName(packageName: String): String {
        return try {
            val appInfo = packageManager.getApplicationInfo(packageName, 0)
            packageManager.getApplicationLabel(appInfo).toString()
        } catch (e: PackageManager.NameNotFoundException) {
            packageName
        }
    }
}
