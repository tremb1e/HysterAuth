package com.continuousauth.utils

import android.content.Context
import android.util.Log
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import dagger.hilt.android.qualifiers.ApplicationContext
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * User ID manager.
 *
 * Generates and manages an app-scoped unique user UUID.
 */
@Singleton
class UserIdManager @Inject constructor(
    @ApplicationContext private val context: Context
) {
    
    companion object {
        private const val TAG = "UserIdManager"
        private const val PREFS_FILE_NAME = "user_secure_prefs"
        private const val KEY_USER_ID = "user_id"
        private const val KEY_SESSION_ID = "session_id"
        private const val KEY_SESSION_START_TIME = "session_start_time"
    }
    
    // Encrypted SharedPreferences for sensitive values
    private val encryptedPrefs by lazy {
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
            
        EncryptedSharedPreferences.create(
            context,
            PREFS_FILE_NAME,
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
    }
    
    // Cached user ID
    private var cachedUserId: String? = null
    
    // Current session info
    private var currentSessionId: String? = null
    private var sessionStartTime: Long = 0
    
    /**
     * Get the user ID.
     *
     * Generates a new UUID if none exists.
     */
    fun getUserId(): String {
        // From cache
        cachedUserId?.let { return it }
        
        // From encrypted storage
        var userId = encryptedPrefs.getString(KEY_USER_ID, null)
        
        // Generate if missing
        if (userId == null) {
            userId = UUID.randomUUID().toString()
            encryptedPrefs.edit().putString(KEY_USER_ID, userId).apply()
            Log.i(TAG, "Generated new user ID: $userId")
        } else {
            Log.d(TAG, "Loaded existing user ID: $userId")
        }
        
        // Update cache
        cachedUserId = userId
        return userId
    }
    
    /**
     * Start a new session.
     *
     * Generates a new session ID and records the start time.
     */
    fun startNewSession(): String {
        currentSessionId = UUID.randomUUID().toString()
        sessionStartTime = System.currentTimeMillis()
        
        // Persist session info
        encryptedPrefs.edit()
            .putString(KEY_SESSION_ID, currentSessionId)
            .putLong(KEY_SESSION_START_TIME, sessionStartTime)
            .apply()
            
        Log.i(TAG, "Started new session: $currentSessionId")
        return currentSessionId!!
    }
    
    /**
     * Get current session ID.
     *
     * Returns null if no active session exists.
     */
    fun getCurrentSessionId(): String? {
        if (currentSessionId == null) {
            // Attempt to restore from storage
            currentSessionId = encryptedPrefs.getString(KEY_SESSION_ID, null)
            sessionStartTime = encryptedPrefs.getLong(KEY_SESSION_START_TIME, 0)
        }
        return currentSessionId
    }
    
    /**
     * Get session start time.
     */
    fun getSessionStartTime(): Long {
        if (sessionStartTime == 0L) {
            sessionStartTime = encryptedPrefs.getLong(KEY_SESSION_START_TIME, 0)
        }
        return sessionStartTime
    }
    
    /**
     * Get session duration (ms).
     */
    fun getSessionDuration(): Long {
        return if (sessionStartTime > 0) {
            System.currentTimeMillis() - sessionStartTime
        } else {
            0
        }
    }
    
    /**
     * Format session duration as a human-readable string.
     */
    fun getFormattedSessionDuration(): String {
        val duration = getSessionDuration()
        val seconds = (duration / 1000) % 60
        val minutes = (duration / (1000 * 60)) % 60
        val hours = duration / (1000 * 60 * 60)
        
        return when {
            hours > 0 -> String.format("%02d:%02d:%02d", hours, minutes, seconds)
            else -> String.format("%02d:%02d", minutes, seconds)
        }
    }
    
    /**
     * End the current session.
     */
    fun endSession() {
        currentSessionId = null
        sessionStartTime = 0
        
        // Remove persisted session info
        encryptedPrefs.edit()
            .remove(KEY_SESSION_ID)
            .remove(KEY_SESSION_START_TIME)
            .apply()
            
        Log.i(TAG, "Session ended")
    }
    
    /**
     * Reset user ID.
     *
     * Use only in exceptional cases (e.g., consent withdrawal).
     */
    fun resetUserId() {
        cachedUserId = null
        encryptedPrefs.edit().remove(KEY_USER_ID).apply()
        Log.w(TAG, "User ID reset")
    }
    
    /**
     * Clear all user data.
     */
    fun clearAllData() {
        cachedUserId = null
        currentSessionId = null
        sessionStartTime = 0
        encryptedPrefs.edit().clear().apply()
        Log.w(TAG, "All user data cleared")
    }
}
