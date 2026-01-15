package com.continuousauth.versioning

import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dagger.hilt.android.testing.HiltAndroidRule
import dagger.hilt.android.testing.HiltAndroidTest
import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import javax.inject.Inject

/**
 * Version manager tests.
 *
 * Covers app version upgrades and schema migrations.
 */
@HiltAndroidTest
@RunWith(AndroidJUnit4::class)
class VersionManagerTest {

    @get:Rule
    var hiltRule = HiltAndroidRule(this)

    @Inject
    lateinit var versionManager: VersionManager

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Before
    fun init() {
        hiltRule.inject()
    }

    /**
     * Tests first-run upgrade behavior.
     */
    @Test
    fun testFirstRunUpgrade() = runTest {
        // Clear any previous state.
        clearVersionPreferences()
        
        // Run upgrade.
        val result = versionManager.checkAndPerformUpgrade()
        
        // Assertions.
        assertTrue("Should be recognized as first run", result.wasFirstRun)
        assertTrue("Should include upgrade messages", result.upgrades.isNotEmpty())
        
        // Version info should be updated.
        val versionInfo = versionManager.getCurrentVersionInfo()
        assertEquals("App version code should match", VersionManager.CURRENT_APP_VERSION_CODE, versionInfo.appVersionCode)
        assertEquals("App version name should match", VersionManager.CURRENT_APP_VERSION_NAME, versionInfo.appVersionName)
        assertEquals("Schema version should match", VersionManager.CURRENT_SCHEMA_VERSION, versionInfo.schemaVersion)
    }

    /**
     * Tests that repeated upgrades are not re-applied.
     */
    @Test
    fun testNoUpgradeWhenVersionsMatch() = runTest {
        // Run upgrade once.
        versionManager.checkAndPerformUpgrade()
        
        // Run again.
        val result = versionManager.checkAndPerformUpgrade()
        
        // Assertions.
        assertFalse("Should not be recognized as first run", result.wasFirstRun)
        assertTrue("Should not include upgrade messages", result.upgrades.isEmpty())
    }

    /**
     * Tests whether an upgrade is needed.
     */
    @Test
    fun testNeedsUpgradeCheck() = runTest {
        // Clear any previous state.
        clearVersionPreferences()
        
        // Fresh install should need upgrade.
        assertTrue("Fresh install should require upgrade", versionManager.needsUpgrade())
        
        // Perform upgrade.
        versionManager.checkAndPerformUpgrade()
        
        // Re-check.
        assertFalse("Should not require upgrade after upgrade", versionManager.needsUpgrade())
    }

    /**
     * Tests data packet schema version compatibility.
     */
    @Test
    fun testDataPacketSchemaCompatibility() {
        val currentVersion = versionManager.getDataPacketSchemaVersion()
        
        // Current version should be compatible.
        assertTrue("Current version should be compatible", 
            versionManager.isDataPacketSchemaCompatible(currentVersion))
        
        // Lower versions should be compatible.
        assertTrue("Lower version should be compatible", 
            versionManager.isDataPacketSchemaCompatible(currentVersion - 1))
        
        // Higher versions should not be compatible.
        assertFalse("Higher version should not be compatible", 
            versionManager.isDataPacketSchemaCompatible(currentVersion + 1))
    }

    /**
     * Tests retrieving version info.
     */
    @Test
    fun testGetVersionInfo() = runTest {
        // Ensure version info exists.
        versionManager.checkAndPerformUpgrade()
        
        val versionInfo = versionManager.getCurrentVersionInfo()
        
        // Assertions.
        assertEquals("App version code should match", 
            VersionManager.CURRENT_APP_VERSION_CODE, versionInfo.appVersionCode)
        assertEquals("App version name should match", 
            VersionManager.CURRENT_APP_VERSION_NAME, versionInfo.appVersionName)
        assertEquals("Schema version should match", 
            VersionManager.CURRENT_SCHEMA_VERSION, versionInfo.schemaVersion)
        assertTrue("Should have a last upgrade time", versionInfo.lastUpgradeTime > 0)
    }

    /**
     * Tests a simulated version upgrade scenario.
     */
    @Test
    fun testSimulatedVersionUpgrade() = runTest {
        // Simulate an old version state.
        setMockOldVersion(appVersionCode = 0, schemaVersion = 0)
        
        // Run upgrade.
        val result = versionManager.checkAndPerformUpgrade()
        
        // Assertions.
        assertTrue("Should include app version upgrade", 
            result.upgrades.any { it.contains("App version", ignoreCase = true) })
        assertTrue("Should include schema upgrade", 
            result.upgrades.any { it.contains("Schema", ignoreCase = true) })
    }

    /**
     * Tests upgrade timestamps.
     */
    @Test
    fun testUpgradeTimestamp() = runTest {
        clearVersionPreferences()
        
        val beforeTime = System.currentTimeMillis()
        val result = versionManager.checkAndPerformUpgrade()
        val afterTime = System.currentTimeMillis()
        
        // Timestamp should be within the execution time window.
        assertTrue("Upgrade timestamp should be within the execution window", 
            result.timestamp >= beforeTime && result.timestamp <= afterTime)
        
        // Version info timestamp should match.
        val versionInfo = versionManager.getCurrentVersionInfo()
        assertEquals("Version info timestamp should match upgrade result", 
            result.timestamp, versionInfo.lastUpgradeTime)
    }

    /**
     * Clears version-related preferences.
     */
    private suspend fun clearVersionPreferences() {
        context.dataStore.edit { preferences ->
            preferences.clear()
        }
    }

    /**
     * Sets mocked old version data.
     */
    private suspend fun setMockOldVersion(appVersionCode: Int, schemaVersion: Int) {
        val STORED_APP_VERSION_CODE = intPreferencesKey("stored_app_version_code")
        val STORED_APP_VERSION_NAME = stringPreferencesKey("stored_app_version_name")
        val STORED_SCHEMA_VERSION = intPreferencesKey("stored_schema_version")
        
        context.dataStore.edit { preferences ->
            preferences[STORED_APP_VERSION_CODE] = appVersionCode
            preferences[STORED_APP_VERSION_NAME] = "0.9.0"
            preferences[STORED_SCHEMA_VERSION] = schemaVersion
        }
    }

    /**
     * Tests retrieving the data packet schema version.
     */
    @Test
    fun testGetDataPacketSchemaVersion() {
        val schemaVersion = versionManager.getDataPacketSchemaVersion()
        
        assertEquals("Schema version should match constant", 
            VersionManager.CURRENT_SCHEMA_VERSION, schemaVersion)
        assertTrue("Schema version should be > 0", schemaVersion > 0)
    }
}
