package com.continuousauth.policy

import android.content.Context
import android.util.Log
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import com.continuousauth.proto.*
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Policy manager.
 *
 * Persists and applies policy configuration received from the server.
 */
@Singleton
class PolicyManager @Inject constructor(
    @ApplicationContext private val context: Context
) {
    
    companion object {
        private const val TAG = "PolicyManager"
        private const val PREFERENCES_NAME = "policy_preferences"
        
        // DataStore Keys
        private val POLICY_ID = stringPreferencesKey("policy_id")
        private val POLICY_TIMESTAMP = longPreferencesKey("policy_timestamp")
        private val POLICY_VERSION = stringPreferencesKey("policy_version")
        
        // Transmission policy keys
        private val BATCH_INTERVAL_MS = intPreferencesKey("batch_interval_ms")
        private val MAX_PAYLOAD_SIZE = intPreferencesKey("max_payload_size_bytes")
        private val UPLOAD_RATE_LIMIT = floatPreferencesKey("upload_rate_limit")
        private val TRANSMISSION_PROFILE = stringPreferencesKey("transmission_profile")
        private val COMPRESSION_ALGORITHM = stringPreferencesKey("compression_algorithm")
        private val BATCH_SIZE_THRESHOLD = intPreferencesKey("batch_size_threshold")
        
        // Anomaly detection policy keys
        private val ANOMALY_ENABLED = booleanPreferencesKey("anomaly_enabled")
        private val ANOMALY_THRESHOLD_MULTIPLIER = floatPreferencesKey("anomaly_threshold_multiplier")
        private val ANOMALY_WINDOW_SIZE = intPreferencesKey("anomaly_window_size_sec")
        private val ANOMALY_COOLDOWN = intPreferencesKey("anomaly_cooldown_period_sec")
        
        // Sensor configuration keys
        private val ENABLED_SENSORS = stringSetPreferencesKey("enabled_sensors")
        private val SENSOR_SAMPLING_RATES = stringPreferencesKey("sensor_sampling_rates_json") // JSON-encoded map
        
        // Default policy values
        private const val DEFAULT_BATCH_INTERVAL = 1000
        private const val DEFAULT_MAX_PAYLOAD_SIZE = 10 * 1024 * 1024 // 10MB
        private const val DEFAULT_ANOMALY_THRESHOLD_MULTIPLIER = 2.0f
        private const val DEFAULT_TRANSMISSION_PROFILE = "UNRESTRICTED"
    }
    
    private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(
        name = PREFERENCES_NAME
    )
    
    // Policy update callback list
    private val policyUpdateCallbacks = mutableListOf<(PolicyConfiguration) -> Unit>()
    
    /**
     * Apply a policy update.
     */
    suspend fun updatePolicy(policyUpdate: PolicyUpdate) {
        applyPolicyUpdate(policyUpdate)
    }
    
    /**
     * Apply a policy update.
     */
    suspend fun applyPolicyUpdate(policyUpdate: PolicyUpdate) {
        Log.i(TAG, "Applying policy update - policyId: ${policyUpdate.policyId}")

        try {
            // Persist to DataStore
            context.dataStore.edit { preferences ->
                preferences[POLICY_ID] = policyUpdate.policyId
                preferences[POLICY_VERSION] = policyUpdate.policyVersion
                
                // Transmission configuration
                if (policyUpdate.batchIntervalMs > 0) {
                    preferences[BATCH_INTERVAL_MS] = policyUpdate.batchIntervalMs
                }
                if (policyUpdate.maxPayloadSizeBytes > 0) {
                    preferences[MAX_PAYLOAD_SIZE] = policyUpdate.maxPayloadSizeBytes
                }
                if (policyUpdate.uploadRateLimit > 0) {
                    preferences[UPLOAD_RATE_LIMIT] = policyUpdate.uploadRateLimit
                }
                if (policyUpdate.transmissionProfile.isNotEmpty()) {
                    preferences[TRANSMISSION_PROFILE] = policyUpdate.transmissionProfile
                }
                if (policyUpdate.compressionAlgorithm.isNotEmpty()) {
                    preferences[COMPRESSION_ALGORITHM] = policyUpdate.compressionAlgorithm
                }
                if (policyUpdate.batchSizeThreshold > 0) {
                    preferences[BATCH_SIZE_THRESHOLD] = policyUpdate.batchSizeThreshold
                }
                
                // Anomaly detection configuration
                if (policyUpdate.hasAnomalyConfig()) {
                    val ac = policyUpdate.anomalyConfig
                    preferences[ANOMALY_ENABLED] = ac.enabled
                    preferences[ANOMALY_THRESHOLD_MULTIPLIER] = ac.thresholdMultiplier
                    preferences[ANOMALY_WINDOW_SIZE] = ac.windowSizeSec
                    preferences[ANOMALY_COOLDOWN] = ac.cooldownPeriodSec
                }
                
                // Sensor configuration
                if (policyUpdate.enabledSensorsList.isNotEmpty()) {
                    preferences[ENABLED_SENSORS] = policyUpdate.enabledSensorsList.toSet()
                }
                if (policyUpdate.sensorSamplingRatesCount > 0) {
                    // Serialize the map as JSON
                    val ratesJson = org.json.JSONObject(policyUpdate.sensorSamplingRatesMap).toString()
                    preferences[SENSOR_SAMPLING_RATES] = ratesJson
                }
            }

            // Load current configuration and notify callbacks
            val currentConfig = getCurrentPolicyConfiguration()
            notifyPolicyUpdate(currentConfig)

            Log.i(TAG, "Policy update applied successfully")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to apply policy update", e)
        }
    }
    
    /**
     * Get the current policy configuration.
     */
    fun getCurrentPolicyConfiguration(): PolicyConfiguration {
        return runBlocking {
            context.dataStore.data
                .map { preferences ->
                    PolicyConfiguration(
                        policyId = preferences[POLICY_ID] ?: "",
                        transmissionConfig = TransmissionConfiguration(
                            batchIntervalMs = preferences[BATCH_INTERVAL_MS] ?: DEFAULT_BATCH_INTERVAL,
                            maxPayloadSizeBytes = preferences[MAX_PAYLOAD_SIZE] ?: DEFAULT_MAX_PAYLOAD_SIZE,
                            uploadRateLimit = preferences[UPLOAD_RATE_LIMIT] ?: 10.0f,
                            transmissionProfile = preferences[TRANSMISSION_PROFILE] ?: DEFAULT_TRANSMISSION_PROFILE,
                            compressionAlgorithm = preferences[COMPRESSION_ALGORITHM] ?: "lz4",
                            batchSizeThreshold = preferences[BATCH_SIZE_THRESHOLD] ?: 100
                        ),
                        collectionConfig = CollectionConfiguration(
                            anomalyEnabled = preferences[ANOMALY_ENABLED] ?: false,
                            anomalyThresholdMultiplier = preferences[ANOMALY_THRESHOLD_MULTIPLIER] ?: DEFAULT_ANOMALY_THRESHOLD_MULTIPLIER,
                            anomalyWindowSizeSec = preferences[ANOMALY_WINDOW_SIZE] ?: 60,
                            anomalyCooldownSec = preferences[ANOMALY_COOLDOWN] ?: 60,
                            enabledSensors = preferences[ENABLED_SENSORS] ?: setOf("ACCELEROMETER", "GYROSCOPE", "MAGNETOMETER"),
                            sensorSamplingRates = parseSamplingRates(preferences[SENSOR_SAMPLING_RATES])
                        ),
                        securityConfig = SecurityConfiguration(
                            serverEndpoint = "",
                            pinnedCertificates = emptySet()
                        )
                    )
                }
                .first()
        }
    }
    
    /**
     * Register a policy update callback.
     */
    fun registerPolicyUpdateCallback(callback: (PolicyConfiguration) -> Unit) {
        policyUpdateCallbacks.add(callback)
    }
    
    /**
     * Unregister a policy update callback.
     */
    fun unregisterPolicyUpdateCallback(callback: (PolicyConfiguration) -> Unit) {
        policyUpdateCallbacks.remove(callback)
    }
    
    /**
     * Notify policy updates.
     */
    private fun notifyPolicyUpdate(config: PolicyConfiguration) {
        policyUpdateCallbacks.forEach { callback ->
            try {
                callback(config)
            } catch (e: Exception) {
                Log.e(TAG, "Policy update callback failed", e)
            }
        }
    }
    
    /**
     * Get the current policy (simplified).
     */
    suspend fun getCurrentPolicy(): Policy {
        val config = withContext(Dispatchers.IO) {
            getCurrentPolicyConfiguration()
        }
        return Policy(
            batchIntervalMs = config.transmissionConfig.batchIntervalMs,
            anomalyThresholdMultiplier = config.collectionConfig.anomalyThresholdMultiplier,
            batchSizeThreshold = config.transmissionConfig.batchSizeThreshold,
            transmissionProfile = config.transmissionConfig.transmissionProfile
        )
    }
    
    /**
     * Parse sensor sampling rates JSON.
     */
    private fun parseSamplingRates(json: String?): Map<String, Int> {
        if (json.isNullOrEmpty()) {
            return mapOf(
                "ACCELEROMETER" to 200,
                "GYROSCOPE" to 200,
                "MAGNETOMETER" to 100
            )
        }
        return try {
            val jsonObject = org.json.JSONObject(json)
            val map = mutableMapOf<String, Int>()
            jsonObject.keys().forEach { key ->
                map[key] = jsonObject.getInt(key)
            }
            map
        } catch (e: Exception) {
            mapOf(
                "ACCELEROMETER" to 200,
                "GYROSCOPE" to 200,
                "MAGNETOMETER" to 100
            )
        }
    }
}

/**
 * Simplified policy model.
 */
data class Policy(
    val batchIntervalMs: Int, // Batch interval (ms)
    val anomalyThresholdMultiplier: Float, // Anomaly threshold multiplier
    val batchSizeThreshold: Int, // Batch size threshold
    val transmissionProfile: String // Transmission profile
)

/**
 * Policy configuration model.
 */
data class PolicyConfiguration(
    val policyId: String,
    val transmissionConfig: TransmissionConfiguration,
    val collectionConfig: CollectionConfiguration,
    val securityConfig: SecurityConfiguration
)

/**
 * Transmission configuration.
 */
data class TransmissionConfiguration(
    val batchIntervalMs: Int,
    val maxPayloadSizeBytes: Int,
    val uploadRateLimit: Float,
    val transmissionProfile: String,
    val compressionAlgorithm: String,
    val batchSizeThreshold: Int
)

/**
 * Data collection configuration.
 */
data class CollectionConfiguration(
    val anomalyEnabled: Boolean,
    val anomalyThresholdMultiplier: Float,
    val anomalyWindowSizeSec: Int,
    val anomalyCooldownSec: Int,
    val enabledSensors: Set<String>,
    val sensorSamplingRates: Map<String, Int>
)

/**
 * Security configuration.
 */
data class SecurityConfiguration(
    val serverEndpoint: String,
    val pinnedCertificates: Set<String>
)
