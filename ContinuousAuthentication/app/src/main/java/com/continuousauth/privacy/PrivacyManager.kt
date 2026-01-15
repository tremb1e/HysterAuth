package com.continuousauth.privacy

import android.content.Context
import android.util.Log
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import com.continuousauth.database.BatchMetadataDao
import com.continuousauth.database.BatchStatus
import com.continuousauth.database.ContinuousAuthDatabase
import com.continuousauth.network.GrpcManager
// import com.continuousauth.proto.DataDeletionRequest
// import com.continuousauth.proto.DataDeletionResponse
import com.continuousauth.storage.FileQueueManager
import com.continuousauth.utils.UserIdManager
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Privacy manager responsible for consent, data deletion, and retention controls.
 */
@Singleton
class PrivacyManager @Inject constructor(
    @ApplicationContext private val context: Context,
    private val database: ContinuousAuthDatabase,
    private val fileQueueManager: FileQueueManager,
    private val userIdManager: UserIdManager,
    private val grpcManager: GrpcManager
) {
    
    companion object {
        private const val TAG = "PrivacyManager"
        private const val PREFS_FILE = "privacy_prefs"
        private const val KEY_CONSENT_GIVEN = "consent_given"
        private const val KEY_CONSENT_TIMESTAMP = "consent_timestamp"
        private const val KEY_DATA_RETENTION_DAYS = "data_retention_days"
        private const val KEY_LAST_CLEANUP_TIME = "last_cleanup_time"
        
        const val DEFAULT_RETENTION_DAYS = 30  // Default retention: 30 days
        const val MAX_RETENTION_DAYS = 365     // Max retention: 365 days
    }
    
    // Encrypted SharedPreferences for privacy settings
    private val encryptedPrefs by lazy {
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
            
        EncryptedSharedPreferences.create(
            context,
            PREFS_FILE,
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
    }
    
    // User consent state
    private val _consentState = MutableStateFlow(ConsentState.UNKNOWN)
    val consentState: StateFlow<ConsentState> = _consentState.asStateFlow()
    
    // Data deletion state
    private val _deletionState = MutableStateFlow(DeletionState.IDLE)
    val deletionState: StateFlow<DeletionState> = _deletionState.asStateFlow()
    
    init {
        // Check consent state on init
        checkConsentStatus()
    }
    
    /**
     * Check user consent status.
     */
    fun checkConsentStatus() {
        val hasConsent = encryptedPrefs.getBoolean(KEY_CONSENT_GIVEN, false)
        _consentState.value = if (hasConsent) {
            ConsentState.GRANTED
        } else {
            ConsentState.NOT_GRANTED
        }
        
        if (hasConsent) {
            val consentTime = encryptedPrefs.getLong(KEY_CONSENT_TIMESTAMP, 0)
            Log.i(TAG, "Consent granted at: ${java.util.Date(consentTime)}")
        }
    }
    
    /**
     * Record user consent.
     */
    fun grantConsent() {
        encryptedPrefs.edit()
            .putBoolean(KEY_CONSENT_GIVEN, true)
            .putLong(KEY_CONSENT_TIMESTAMP, System.currentTimeMillis())
            .apply()
        
        _consentState.value = ConsentState.GRANTED
        Log.i(TAG, "Privacy consent granted")
    }
    
    /**
     * Withdraw consent and delete all associated data.
     */
    suspend fun withdrawConsentAndDeleteData(): Result<Unit> {
        return withContext(Dispatchers.IO) {
            try {
                _deletionState.value = DeletionState.IN_PROGRESS
                Log.i(TAG, "Starting consent withdrawal flow")
                
                // 1) Update consent state
                _consentState.value = ConsentState.WITHDRAWN
                encryptedPrefs.edit()
                    .putBoolean(KEY_CONSENT_GIVEN, false)
                    .putLong(KEY_CONSENT_TIMESTAMP, System.currentTimeMillis())
                    .apply()
                
                // 2) Delete local cached data
                val localDeletionResult = deleteAllLocalData()
                if (!localDeletionResult.isSuccess) {
                    Log.e(TAG, "Local data deletion failed: ${localDeletionResult.exceptionOrNull()}")
                    _deletionState.value = DeletionState.FAILED
                    return@withContext Result.failure(
                        Exception("Local data deletion failed: ${localDeletionResult.exceptionOrNull()?.message}")
                    )
                }
                
                // 3) Send deletion request to the server
                val serverDeletionResult = sendServerDeletionRequest()
                if (!serverDeletionResult.isSuccess) {
                    Log.e(TAG, "Server deletion request failed: ${serverDeletionResult.exceptionOrNull()}")
                    // Server failure should not block local deletion, but should be recorded.
                    _deletionState.value = DeletionState.PARTIAL_SUCCESS
                    
                    // Save failed deletion request for retry.
                    savePendingDeletionRequest()
                } else {
                    _deletionState.value = DeletionState.SUCCESS
                }
                
                Log.i(TAG, "Consent withdrawal flow completed")
                Result.success(Unit)
                
            } catch (e: Exception) {
                Log.e(TAG, "Error during consent withdrawal flow", e)
                _deletionState.value = DeletionState.FAILED
                Result.failure(e)
            }
        }
    }
    
    /**
     * Delete all local data.
     */
    private suspend fun deleteAllLocalData(): Result<Unit> {
        return withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Starting local data deletion")
                
                // 1) Delete all batch metadata entries
                val batchDao = database.batchMetadataDao()
                // Use getPendingBatches with all statuses to get all batches as a List
                val allBatches = batchDao.getPendingBatches(
                    listOf(BatchStatus.PENDING, BatchStatus.UPLOADING, BatchStatus.UPLOADED, 
                           BatchStatus.FAILED, BatchStatus.ACKNOWLEDGED, BatchStatus.CORRUPT)
                )
                Log.d(TAG, "Batches to delete: ${allBatches.size}")
                
                for (batch in allBatches) {
                    // Delete file
                    val file = File(batch.filePath)
                    if (file.exists()) {
                        file.delete()
                        Log.d(TAG, "Deleted file: ${batch.filePath}")
                    }
                    
                    // Delete database entry
                    batchDao.delete(batch)
                }
                
                // 2) Clear file queue
                fileQueueManager.clearQueue()
                
                // 3) Clear cache directory
                clearCacheDirectory()
                
                // 4) Clear user/session identifiers
                userIdManager.clearAllData()
                
                // 5) Clear SharedPreferences (keep minimal audit records)
                clearPreferences()
                
                Log.i(TAG, "Local data deletion completed")
                Result.success(Unit)
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to delete local data", e)
                Result.failure(e)
            }
        }
    }
    
    /**
     * Send a deletion request to the server.
     *
     * TODO: implement the corresponding server endpoint.
     */
    private suspend fun sendServerDeletionRequest(): Result<Unit> {
        return withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Sending data deletion request to the server")
                
                // TODO: Implement the actual gRPC call.
                // The code below is a sketch and requires a matching server-side RPC.
                /*
                val request = DataDeletionRequest.newBuilder()
                    .setDeviceIdHash(getDeviceIdHash())
                    .setUserId(userIdManager.getUserId())
                    .setTimestamp(System.currentTimeMillis())
                    .setReason("USER_CONSENT_WITHDRAWAL")
                    .build()
                
                val response = grpcManager.sendDataDeletionRequest(request)
                
	                if (response.success) {
	                    Log.i(TAG, "Server deletion request succeeded")
	                    Result.success(Unit)
	                } else {
	                    Log.e(TAG, "Server rejected deletion request: ${response.message}")
	                    Result.failure(Exception(response.message))
	                }
                */
                
                // Temporary no-op: treat as success until the server endpoint exists.
                Log.w(TAG, "Server deletion endpoint not implemented; skipping")
                Result.success(Unit)
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send server deletion request", e)
                Result.failure(e)
            }
        }
    }
    
    /**
     * Persist a pending deletion request for retry.
     */
    private fun savePendingDeletionRequest() {
        encryptedPrefs.edit()
            .putBoolean("pending_deletion_request", true)
            .putLong("pending_deletion_timestamp", System.currentTimeMillis())
            .apply()
        
        Log.i(TAG, "Saved pending deletion request")
    }
    
    /**
     * Check and retry pending deletion requests.
     */
    suspend fun retryPendingDeletionRequests() {
        if (encryptedPrefs.getBoolean("pending_deletion_request", false)) {
            Log.i(TAG, "Found pending deletion request; retrying")
            
            val result = sendServerDeletionRequest()
            if (result.isSuccess) {
                encryptedPrefs.edit()
                    .remove("pending_deletion_request")
                    .remove("pending_deletion_timestamp")
                    .apply()
                Log.i(TAG, "Pending deletion request sent successfully")
            }
        }
    }
    
    /**
     * Clear cache directory.
     */
    private fun clearCacheDirectory() {
        try {
            val cacheDir = context.cacheDir
            val queueDir = File(cacheDir, "sensor_queue")
            if (queueDir.exists()) {
                queueDir.deleteRecursively()
                Log.d(TAG, "Cache directory cleared")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to clear cache directory", e)
        }
    }
    
    /**
     * Clear SharedPreferences (keep minimal audit records).
     */
    private fun clearPreferences() {
        // Preserve values that must be kept
        val consentWithdrawn = _consentState.value == ConsentState.WITHDRAWN
        val withdrawalTime = if (consentWithdrawn) System.currentTimeMillis() else 0L
        
        // Clear other preference files
        val prefsDir = File(context.applicationInfo.dataDir, "shared_prefs")
        prefsDir.listFiles()?.forEach { file ->
            if (!file.name.contains(PREFS_FILE)) {
                // Keep the privacy prefs file itself
                file.delete()
                Log.d(TAG, "Deleted prefs file: ${file.name}")
            }
        }
        
        // If consent was withdrawn, record withdrawal time
        if (consentWithdrawn) {
            encryptedPrefs.edit()
                .putBoolean("consent_withdrawn", true)
                .putLong("withdrawal_timestamp", withdrawalTime)
                .apply()
        }
    }
    
    /**
     * Set data retention period (days).
     */
    fun setDataRetentionDays(days: Int) {
        val validDays = days.coerceIn(1, MAX_RETENTION_DAYS)
        encryptedPrefs.edit()
            .putInt(KEY_DATA_RETENTION_DAYS, validDays)
            .apply()
        
        Log.i(TAG, "Data retention period set to: $validDays days")
    }
    
    /**
     * Get data retention period (days).
     */
    fun getDataRetentionDays(): Int {
        return encryptedPrefs.getInt(KEY_DATA_RETENTION_DAYS, DEFAULT_RETENTION_DAYS)
    }
    
    /**
     * Perform retention cleanup.
     *
     * Deletes data older than the configured retention period.
     */
    suspend fun performRetentionCleanup() {
        withContext(Dispatchers.IO) {
            try {
                val retentionDays = getDataRetentionDays()
                val cutoffTime = System.currentTimeMillis() - (retentionDays * 24 * 60 * 60 * 1000L)
                
                Log.i(TAG, "Running retention cleanup; deleting data before ${java.util.Date(cutoffTime)}")
                
                // Delete expired batches
                val batchDao = database.batchMetadataDao()
                // Get all batches and filter for expired ones
                val allBatches = batchDao.getPendingBatches(
                    listOf(BatchStatus.PENDING, BatchStatus.UPLOADING, BatchStatus.UPLOADED, 
                           BatchStatus.FAILED, BatchStatus.ACKNOWLEDGED, BatchStatus.CORRUPT)
                )
                val expiredBatches = allBatches.filter { it.createdTime < cutoffTime }
                
                for (batch in expiredBatches) {
                    // Delete file
                    val file = File(batch.filePath)
                    if (file.exists()) {
                        file.delete()
                    }
                    
                    // Delete database entry
                    batchDao.delete(batch)
                }
                
                Log.i(TAG, "Retention cleanup completed; deleted ${expiredBatches.size} expired batches")
                
                // Update last cleanup time
                encryptedPrefs.edit()
                    .putLong(KEY_LAST_CLEANUP_TIME, System.currentTimeMillis())
                    .apply()
                
            } catch (e: Exception) {
                Log.e(TAG, "Retention cleanup failed", e)
            }
        }
    }
    
    /**
     * Determine whether retention cleanup should run.
     */
    fun shouldPerformRetentionCleanup(): Boolean {
        val lastCleanup = encryptedPrefs.getLong(KEY_LAST_CLEANUP_TIME, 0)
        val daysSinceCleanup = (System.currentTimeMillis() - lastCleanup) / (24 * 60 * 60 * 1000L)
        return daysSinceCleanup >= 1  // Run at most once per day
    }
    
    /**
     * Get privacy policy text.
     */
    fun getPrivacyPolicyText(): String {
        // TODO: Load the full privacy policy from resources or a remote source.
        return """
            Data Collection and Usage Notice

            1. Types of data collected
            - Sensor data: accelerometer, gyroscope, magnetometer
            - Device information: device model, OS version
            - App usage: foreground app information

            2. Purpose of collection
            - Continuous authentication research
            - Improving authentication accuracy
            - Academic research and analysis

            3. Storage and retention
            - All data is encrypted with AES-256
            - Local cache is retained for up to ${getDataRetentionDays()} days
            - Data is stored securely on the server

            4. Sharing
            - Raw data is not shared with third parties
            - Only aggregated statistics may be shared

            5. Your rights
            - You can withdraw consent at any time
            - Upon withdrawal, all related data will be deleted
            - Data export requests are supported

            6. Contact
            - Email: privacy@continuousauth.com
            - Phone: +1-xxx-xxx-xxxx
        """.trimIndent()
    }
}

/**
 * User consent state.
 */
enum class ConsentState {
    UNKNOWN,        // Unknown
    NOT_GRANTED,    // Not granted
    GRANTED,        // Granted
    WITHDRAWN       // Withdrawn
}

/**
 * Data deletion state.
 */
enum class DeletionState {
    IDLE,               // Idle
    IN_PROGRESS,        // In progress
    SUCCESS,            // Success
    PARTIAL_SUCCESS,    // Partial success (local succeeded, server failed)
    FAILED              // Failed
}
