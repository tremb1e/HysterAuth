package com.continuousauth.versioning

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.first
import javax.inject.Inject
import javax.inject.Singleton

private val Context.dataStore by preferencesDataStore(name = "version_settings")

/**
 * Version manager.
 *
 * Handles app version upgrades and data model/schema migrations.
 */
@Singleton
class VersionManager @Inject constructor(
    @ApplicationContext private val context: Context
) {
    companion object {
        // Current version constants
        const val CURRENT_APP_VERSION_CODE = 1
        const val CURRENT_APP_VERSION_NAME = "1.0.0"
        const val CURRENT_SCHEMA_VERSION = 1
        
        // DataStore keys
        private val STORED_APP_VERSION_CODE = intPreferencesKey("stored_app_version_code")
        private val STORED_APP_VERSION_NAME = stringPreferencesKey("stored_app_version_name")
        private val STORED_SCHEMA_VERSION = intPreferencesKey("stored_schema_version")
        private val LAST_UPGRADE_TIME = stringPreferencesKey("last_upgrade_time")
    }
    
    /**
     * Checks and performs required upgrades.
     *
     * @return Upgrade result info
     */
    suspend fun checkAndPerformUpgrade(): UpgradeResult {
        val preferences = context.dataStore.data.first()
        
        val storedAppVersionCode = preferences[STORED_APP_VERSION_CODE] ?: 0
        val storedAppVersionName = preferences[STORED_APP_VERSION_NAME] ?: "0.0.0"
        val storedSchemaVersion = preferences[STORED_SCHEMA_VERSION] ?: 0
        
        val upgrades = mutableListOf<String>()
        
        // App version upgrade
        if (storedAppVersionCode < CURRENT_APP_VERSION_CODE) {
            upgrades.add("App version upgraded from $storedAppVersionName (code: $storedAppVersionCode) to $CURRENT_APP_VERSION_NAME (code: $CURRENT_APP_VERSION_CODE)")
            performAppVersionUpgrade(storedAppVersionCode, CURRENT_APP_VERSION_CODE)
        }
        
        // Schema/data model upgrade
        if (storedSchemaVersion < CURRENT_SCHEMA_VERSION) {
            upgrades.add("Data model version upgraded from $storedSchemaVersion to $CURRENT_SCHEMA_VERSION")
            performSchemaVersionUpgrade(storedSchemaVersion, CURRENT_SCHEMA_VERSION)
        }
        
        // Persist updated version info
        context.dataStore.edit { preferences ->
            preferences[STORED_APP_VERSION_CODE] = CURRENT_APP_VERSION_CODE
            preferences[STORED_APP_VERSION_NAME] = CURRENT_APP_VERSION_NAME
            preferences[STORED_SCHEMA_VERSION] = CURRENT_SCHEMA_VERSION
            preferences[LAST_UPGRADE_TIME] = System.currentTimeMillis().toString()
        }
        
        return UpgradeResult(
            upgrades = upgrades,
            wasFirstRun = storedAppVersionCode == 0,
            timestamp = System.currentTimeMillis()
        )
    }
    
    /**
     * Performs app version upgrade logic.
     */
    private suspend fun performAppVersionUpgrade(fromVersion: Int, toVersion: Int) {
        when {
            fromVersion == 0 && toVersion == 1 -> {
                // First install; nothing special to do.
            }
            fromVersion < toVersion -> {
                // Reserved for future upgrade logic (e.g., cache cleanup or config migrations).
                performIncrementalAppUpgrade(fromVersion, toVersion)
            }
        }
    }
    
    /**
     * Performs schema/data model upgrade logic.
     */
    private suspend fun performSchemaVersionUpgrade(fromVersion: Int, toVersion: Int) {
        when {
            fromVersion == 0 && toVersion == 1 -> {
                // First install; initialize schema.
            }
            fromVersion < toVersion -> {
                // Upgrade schema incrementally.
                for (version in (fromVersion + 1)..toVersion) {
                    upgradeSchemaToVersion(version)
                }
            }
        }
    }
    
    /**
     * Performs incremental app upgrades.
     */
    private suspend fun performIncrementalAppUpgrade(fromVersion: Int, toVersion: Int) {
        for (version in (fromVersion + 1)..toVersion) {
            when (version) {
                1 -> {
                    // v1.0.0 upgrade logic
                    // e.g. migrate old settings, clean obsolete files, etc.
                }
                // Future versions...
            }
        }
    }
    
    /**
     * Upgrades schema to a specific version.
     */
    private suspend fun upgradeSchemaToVersion(version: Int) {
        when (version) {
            1 -> {
                // Schema v1: initial version
                // - DataPacket structure
                // - Encrypted payload format
                // - Sensor data structures
            }
            // Future schema versions...
            // 2 -> {
            //     // Schema v2: add fields or change structures
            // }
        }
    }
    
    /**
     * Returns current version information.
     */
    suspend fun getCurrentVersionInfo(): VersionInfo {
        val preferences = context.dataStore.data.first()
        
        return VersionInfo(
            appVersionCode = CURRENT_APP_VERSION_CODE,
            appVersionName = CURRENT_APP_VERSION_NAME,
            schemaVersion = CURRENT_SCHEMA_VERSION,
            lastUpgradeTime = preferences[LAST_UPGRADE_TIME]?.toLongOrNull() ?: 0L
        )
    }
    
    /**
     * Returns whether an upgrade is needed.
     */
    suspend fun needsUpgrade(): Boolean {
        val preferences = context.dataStore.data.first()
        
        val storedAppVersionCode = preferences[STORED_APP_VERSION_CODE] ?: 0
        val storedSchemaVersion = preferences[STORED_SCHEMA_VERSION] ?: 0
        
        return storedAppVersionCode < CURRENT_APP_VERSION_CODE || 
               storedSchemaVersion < CURRENT_SCHEMA_VERSION
    }
    
    /**
     * Returns the schema version used for data packets.
     */
    fun getDataPacketSchemaVersion(): Int = CURRENT_SCHEMA_VERSION
    
    /**
     * Validates data packet schema compatibility.
     */
    fun isDataPacketSchemaCompatible(packetSchemaVersion: Int): Boolean {
        // Currently supports up to the current version. Backward compatibility can be added later.
        return packetSchemaVersion <= CURRENT_SCHEMA_VERSION
    }
}

/**
 * Upgrade result.
 */
data class UpgradeResult(
    val upgrades: List<String>,
    val wasFirstRun: Boolean,
    val timestamp: Long
)

/**
 * Version information.
 */
data class VersionInfo(
    val appVersionCode: Int,
    val appVersionName: String,
    val schemaVersion: Int,
    val lastUpgradeTime: Long
)
