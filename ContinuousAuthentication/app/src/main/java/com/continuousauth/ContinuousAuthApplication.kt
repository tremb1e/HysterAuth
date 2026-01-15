package com.continuousauth

import android.app.Application
import android.util.Log
import com.continuousauth.time.EnhancedTimeSync
import dagger.hilt.android.HiltAndroidApp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Application entry point.
 *
 * Sets up dependency injection and initializes core services such as time sync.
 */
@HiltAndroidApp
class ContinuousAuthApplication : Application() {
    
    @Inject
    lateinit var timeSync: EnhancedTimeSync
    
    private val applicationScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    override fun onCreate() {
        super.onCreate()
        
        Log.d("ContinuousAuth", "App initialization started")
        
        // Initialize time sync service
        initializeTimeSync()
        
        Log.d("ContinuousAuth", "App initialization completed")
    }
    
    /**
     * Initialize NTP time synchronization.
     *
     * Performs a one-shot sync on startup and then starts periodic sync (hourly).
     */
    private fun initializeTimeSync() {
        applicationScope.launch {
            try {
                // Perform a one-shot NTP sync on startup
                val syncSuccess = timeSync.syncTime()
                if (syncSuccess) {
                    Log.i("ContinuousAuth", "Startup NTP sync succeeded")
                } else {
                    Log.w("ContinuousAuth", "Startup NTP sync failed")
                }
                
                // Start periodic sync (hourly)
                timeSync.startPeriodicSync()
                Log.i("ContinuousAuth", "Periodic NTP sync started")
                
            } catch (e: Exception) {
                Log.e("ContinuousAuth", "Failed to initialize NTP time sync", e)
            }
        }
    }
}
