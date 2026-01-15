package com.continuousauth.receiver

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

/**
 * Broadcast receiver for device unlock events.
 *
 * Listens for device unlock and screen-on events.
 *
 * Important: any data collection must first verify that the user has granted privacy consent.
 */
class DeviceUnlockReceiver : BroadcastReceiver() {
    
    companion object {
        private const val TAG = "DeviceUnlockReceiver"
        private const val PREFS_NAME = "app_prefs"
        private const val PRIVACY_AGREEMENT_KEY = "privacy_agreement_shown"
    }
    
    override fun onReceive(context: Context?, intent: Intent?) {
        if (context == null || intent == null) return
        
        // Always check privacy consent first.
        if (!checkPrivacyConsent(context)) {
            Log.w(TAG, "Privacy consent not granted; ignoring unlock event")
            return
        }
        
        when (intent.action) {
            Intent.ACTION_USER_PRESENT -> {
                Log.d(TAG, "Device unlocked")
                handleDeviceUnlock(context)
            }
            Intent.ACTION_SCREEN_ON -> {
                Log.d(TAG, "Screen turned on")
                handleScreenOn(context)
            }
        }
    }
    
    /**
     * Check whether the user has granted privacy consent.
     */
    private fun checkPrivacyConsent(context: Context): Boolean {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val hasAgreed = prefs.getBoolean(PRIVACY_AGREEMENT_KEY, false)
        
        if (!hasAgreed) {
            Log.w(TAG, "Privacy consent not granted; blocking data collection operations")
        }
        
        return hasAgreed
    }
    
    /**
     * Handle device unlock event.
     */
    private fun handleDeviceUnlock(context: Context) {
        // TODO: Implement logic after device unlock.
        // Example: trigger anomaly checks, switch to a faster profile, etc.
        // No-op for now.
        
        Log.i(TAG, "Device unlocked; ready to trigger related actions")
    }
    
    /**
     * Handle screen-on event.
     */
    private fun handleScreenOn(context: Context) {
        // TODO: Implement logic for screen-on events.
        // No-op for now.
        
        Log.i(TAG, "Screen turned on; ready to listen for unlock events")
    }
}
