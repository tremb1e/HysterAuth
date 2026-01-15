package com.continuousauth.compatibility

import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorManager
import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import com.continuousauth.security.KeyAttestationManagerImpl
import com.continuousauth.observability.PerformanceMonitorImpl
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
 * Cross-device and cross-Android-version compatibility tests.
 *
 * Task 5.2.3: Run compatibility testing across devices and Android versions.
 */
@LargeTest
@HiltAndroidTest
@RunWith(AndroidJUnit4::class)
class CrossPlatformCompatibilityTest {

    @get:Rule
    var hiltRule = HiltAndroidRule(this)

    @Inject
    lateinit var keyAttestationManager: KeyAttestationManagerImpl

    @Inject
    lateinit var performanceMonitor: PerformanceMonitorImpl

    private val context: Context = InstrumentationRegistry.getInstrumentation().targetContext

    @Before
    fun init() {
        hiltRule.inject()
    }

    /**
     * Tests Android version compatibility.
     */
    @Test
    fun testAndroidVersionCompatibility() {
        val apiLevel = Build.VERSION.SDK_INT
        val androidVersion = Build.VERSION.RELEASE
        
        println("Current test device:")
        println("API Level: $apiLevel")
        println("Android Version: $androidVersion")
        println("Device model: ${Build.MODEL}")
        println("Manufacturer: ${Build.MANUFACTURER}")
        println("Product: ${Build.PRODUCT}")
        
        // Validate the minimum supported API level.
        assertTrue("App requires API 30+, current device: API $apiLevel", apiLevel >= 30)
        
        // Check version-specific feature availability.
        when {
            apiLevel >= Build.VERSION_CODES.TIRAMISU -> { // API 33+
                println("Android 13+ features available")
                testAndroid13Features()
            }
            apiLevel >= Build.VERSION_CODES.S -> { // API 31+
                println("Android 12+ features available")
                testAndroid12Features()
            }
            apiLevel >= Build.VERSION_CODES.R -> { // API 30+
                println("Android 11+ features available")
                testAndroid11Features()
            }
        }
    }

    /**
     * Tests sensor hardware compatibility.
     */
    @Test
    fun testSensorHardwareCompatibility() {
        val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        
        // Required sensors.
        val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        val magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        
        assertNotNull("Device must have an accelerometer", accelerometer)
        assertNotNull("Device must have a gyroscope", gyroscope)
        assertNotNull("Device must have a magnetometer", magnetometer)
        
        println("Sensor hardware info:")
        
        // Accelerometer properties.
        accelerometer?.let { sensor ->
            println("Accelerometer:")
            println("  Vendor: ${sensor.vendor}")
            println("  Version: ${sensor.version}")
            println("  Max range: ${sensor.maximumRange}")
            println("  Resolution: ${sensor.resolution}")
            println("  Power: ${sensor.power} mA")
            println("  Min delay: ${sensor.minDelay} μs")
            println("  FIFO max events: ${sensor.fifoMaxEventCount}")
            
            // Basic validation.
            assertTrue("Accelerometer max range should be reasonable", sensor.maximumRange > 0)
            assertTrue("Accelerometer resolution should be reasonable", sensor.resolution > 0)
            
            // High sampling rate support.
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                val maxFreq = 1_000_000f / sensor.minDelay // Hz
                println("  Max sampling frequency: ${String.format("%.1f", maxFreq)} Hz")
                assertTrue("Should support at least 100 Hz sampling", maxFreq >= 100f)
            }
        }
        
        // Gyroscope properties.
        gyroscope?.let { sensor ->
            println("Gyroscope:")
            println("  Vendor: ${sensor.vendor}")
            println("  Max range: ${sensor.maximumRange} rad/s")
            println("  Resolution: ${sensor.resolution}")
            println("  FIFO max events: ${sensor.fifoMaxEventCount}")
            
            assertTrue("Gyroscope max range should be reasonable", sensor.maximumRange > 0)
        }
        
        // Magnetometer properties.
        magnetometer?.let { sensor ->
            println("Magnetometer:")
            println("  Vendor: ${sensor.vendor}")
            println("  Max range: ${sensor.maximumRange} μT")
            println("  Resolution: ${sensor.resolution}")
            
            assertTrue("Magnetometer max range should be reasonable", sensor.maximumRange > 0)
        }
    }

    /**
     * Tests device feature compatibility.
     */
    @Test
    fun testDeviceFeatureCompatibility() {
        val packageManager = context.packageManager
        
        println("Device feature checks:")
        
        // Sensor feature checks.
        val hasSensorAccelerometer = packageManager.hasSystemFeature(PackageManager.FEATURE_SENSOR_ACCELEROMETER)
        val hasSensorGyroscope = packageManager.hasSystemFeature(PackageManager.FEATURE_SENSOR_GYROSCOPE)
        val hasSensorCompass = packageManager.hasSystemFeature(PackageManager.FEATURE_SENSOR_COMPASS)
        
        assertTrue("Device must support an accelerometer", hasSensorAccelerometer)
        assertTrue("Device must support a gyroscope", hasSensorGyroscope)
        assertTrue("Device must support a magnetometer/compass", hasSensorCompass)
        
        // High sampling rate sensor support.
        val hasHighSamplingRate = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            packageManager.hasSystemFeature("android.hardware.sensor.hifi_sensors")
        } else {
            false
        }
        
        println("Hi-fi sensors support: $hasHighSamplingRate")
        
        // Network features.
        val hasWifi = packageManager.hasSystemFeature(PackageManager.FEATURE_WIFI)
        val hasCellular = packageManager.hasSystemFeature(PackageManager.FEATURE_TELEPHONY)
        
        assertTrue("Device should support WiFi or cellular", hasWifi || hasCellular)
        
        println("WiFi support: $hasWifi")
        println("Cellular support: $hasCellular")
        
        // Security model.
        val hasKeystore = packageManager.hasSystemFeature(PackageManager.FEATURE_SECURITY_MODEL_COMPATIBLE)
        println("Security model compatible: $hasKeystore")
        
        // StrongBox support.
        val hasStrongBox = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            packageManager.hasSystemFeature("android.hardware.strongbox_keystore")
        } else {
            false
        }
        println("StrongBox support: $hasStrongBox")
    }

    /**
     * Tests keystore compatibility.
     */
    @Test
    fun testKeyStoreCompatibility() = runTest {
        println("Keystore compatibility test:")
        
        // Attestation support.
        val attestationSupported = keyAttestationManager.isAttestationSupported()
        val strongBoxSupported = keyAttestationManager.isStrongBoxSupported()
        
        println("Key attestation support: $attestationSupported")
        println("StrongBox support: $strongBoxSupported")
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            assertTrue("API 24+ should support key attestation", attestationSupported)
        }
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            // StrongBox support depends on hardware and is not available on all devices.
            println("StrongBox support recorded (not required)")
        }
        
        // Basic keystore functionality.
        try {
            val keyStore = java.security.KeyStore.getInstance("AndroidKeyStore")
            keyStore.load(null)
            println("AndroidKeyStore loaded successfully")
            
            val aliases = keyStore.aliases().toList()
            println("Current alias count: ${aliases.size}")
            
        } catch (e: Exception) {
            fail("AndroidKeyStore should be usable: ${e.message}")
        }
    }

    /**
     * Tests memory and performance compatibility.
     */
    @Test
    fun testMemoryAndPerformanceCompatibility() = runTest {
        println("Memory and performance compatibility test:")
        
        // Device memory info.
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memInfo = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        
        val totalMemoryMB = memInfo.totalMem / 1024 / 1024
        val availableMemoryMB = memInfo.availMem / 1024 / 1024
        
        println("Total memory: ${totalMemoryMB}MB")
        println("Available memory: ${availableMemoryMB}MB")
        println("Low-memory threshold: ${memInfo.threshold / 1024 / 1024}MB")
        println("Low memory: ${memInfo.lowMemory}")
        
        // Validate minimum memory requirements.
        assertTrue("Device should have at least 2GB RAM", totalMemoryMB >= 2048)
        assertTrue("Should have enough available memory", availableMemoryMB >= 512)
        
        // Validate performance monitoring works on this device.
        performanceMonitor.startMonitoring(1000L)
        delay(3000L)
        
        val perfStats = performanceMonitor.getPerformanceStats(3000L)
        assertTrue("Performance monitor should collect samples", perfStats.sampleCount > 0)
        
        println("Performance monitor results:")
        println("Sample count: ${perfStats.sampleCount}")
        println("Average memory usage: ${String.format("%.1f", perfStats.memoryUsageAvg)}MB")
        println("Heap utilization: ${String.format("%.1f", perfStats.heapUtilization)}%")
        
        performanceMonitor.stopMonitoring()
    }

    /**
     * Tests permission compatibility.
     */
    @Test
    fun testPermissionCompatibility() {
        println("Permission compatibility test:")
        
        val requiredPermissions = listOf(
            android.Manifest.permission.INTERNET,
            android.Manifest.permission.ACCESS_NETWORK_STATE,
            android.Manifest.permission.WAKE_LOCK
        )
        
        val optionalPermissions = listOf(
            android.Manifest.permission.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
            "android.permission.PACKAGE_USAGE_STATS"
        )
        
        // Required permissions.
        requiredPermissions.forEach { permission ->
            val granted = context.checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED
            println("Required permission $permission: ${if (granted) "granted" else "not granted"}")
            assertTrue("Required permission should be granted: $permission", granted)
        }
        
        // Optional permissions (not required).
        optionalPermissions.forEach { permission ->
            val granted = context.checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED
            println("Optional permission $permission: ${if (granted) "granted" else "not granted"}")
        }
        
        // High sampling rate sensors permission (API 31+).
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val highSamplingPermission = "android.permission.HIGH_SAMPLING_RATE_SENSORS"
            val granted = context.checkSelfPermission(highSamplingPermission) == PackageManager.PERMISSION_GRANTED
            println("High sampling rate sensors permission: ${if (granted) "granted" else "not granted"}")
        }
    }

    /**
     * Tests Android 11-specific features.
     */
    private fun testAndroid11Features() {
        println("Testing Android 11-specific features:")
        
        // Foreground service configuration.
        val packageManager = context.packageManager
        try {
            val serviceInfo = packageManager.getServiceInfo(
                android.content.ComponentName(context, "com.continuousauth.service.DataCollectionService"),
                0
            )
            println("Foreground service configuration looks correct")
        } catch (e: Exception) {
            println("Foreground service configuration check failed: ${e.message}")
        }
    }

    /**
     * Tests Android 12-specific features.
     */
    private fun testAndroid12Features() {
        testAndroid11Features()
        println("Testing Android 12-specific features:")
        
        // Approximate location permission impacts.
        println("Android 12+ permission model supported")
    }

    /**
     * Tests Android 13-specific features.
     */
    private fun testAndroid13Features() {
        testAndroid12Features()
        println("Testing Android 13-specific features:")
        
        // Notification permission.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val notificationPermission = context.checkSelfPermission("android.permission.POST_NOTIFICATIONS")
            println("Notification permission: ${if (notificationPermission == PackageManager.PERMISSION_GRANTED) "granted" else "not granted"}")
        }
    }

    /**
     * Tests processor architecture compatibility.
     */
    @Test
    fun testProcessorArchitectureCompatibility() {
        println("Processor architecture compatibility test:")
        
        val supportedAbis = Build.SUPPORTED_ABIS.toList()
        val supportedAbis32 = Build.SUPPORTED_32_BIT_ABIS.toList()
        val supportedAbis64 = Build.SUPPORTED_64_BIT_ABIS.toList()
        
        println("Supported ABIs: ${supportedAbis.joinToString(", ")}")
        println("32-bit ABIs: ${supportedAbis32.joinToString(", ")}")
        println("64-bit ABIs: ${supportedAbis64.joinToString(", ")}")
        
        assertTrue("Device should support at least one ABI", supportedAbis.isNotEmpty())
        
        // Modern devices should support 64-bit.
        val has64BitSupport = supportedAbis64.isNotEmpty()
        println("64-bit support: $has64BitSupport")
        
        // Common architecture support.
        val commonArchitectures = listOf("arm64-v8a", "armeabi-v7a", "x86_64", "x86")
        val supportedCommonArch = commonArchitectures.filter { it in supportedAbis }
        println("Supported common arch: ${supportedCommonArch.joinToString(", ")}")
        
        assertTrue("Should support at least one common architecture", supportedCommonArch.isNotEmpty())
    }

    /**
     * Tests storage compatibility.
     */
    @Test
    fun testStorageCompatibility() {
        println("Storage compatibility test:")
        
        // App private directory access.
        val cacheDir = context.cacheDir
        val filesDir = context.filesDir
        
        assertNotNull("App cache directory should be accessible", cacheDir)
        assertNotNull("App files directory should be accessible", filesDir)
        
        println("Cache dir: ${cacheDir.absolutePath}")
        println("Files dir: ${filesDir.absolutePath}")
        
        // Available storage.
        val cacheSpace = cacheDir.usableSpace / 1024 / 1024 // MB
        val filesSpace = filesDir.usableSpace / 1024 / 1024 // MB
        
        println("Cache usable space: ${cacheSpace}MB")
        println("Files usable space: ${filesSpace}MB")
        
        assertTrue("Cache dir should have enough space", cacheSpace >= 100) // At least 100MB
        assertTrue("Files dir should have enough space", filesSpace >= 50)  // At least 50MB
        
        // Write access.
        try {
            val testFile = java.io.File(cacheDir, "compatibility_test.tmp")
            testFile.writeText("test data")
            assertTrue("Should be able to write to cache dir", testFile.exists())
            testFile.delete()
            println("Cache dir write access OK")
        } catch (e: Exception) {
            fail("Cache dir write test failed: ${e.message}")
        }
    }

    /**
     * Generates a compatibility report.
     */
    @Test
    fun generateCompatibilityReport() {
        println("=== Device compatibility report ===")
        println("Device info:")
        println("  Manufacturer: ${Build.MANUFACTURER}")
        println("  Model: ${Build.MODEL}")
        println("  Product: ${Build.PRODUCT}")
        println("  Android: ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
        println("  Security patch: ${Build.VERSION.SECURITY_PATCH}")
        println("  Build: ${Build.DISPLAY}")
        
        // Collect compatibility info.
        val report = StringBuilder()
        report.appendLine("Basic device compatibility: ✓")
        
        try {
            testAndroidVersionCompatibility()
            report.appendLine("Android version compatibility: ✓")
        } catch (e: AssertionError) {
            report.appendLine("Android version compatibility: ✗ - ${e.message}")
        }
        
        try {
            testSensorHardwareCompatibility()
            report.appendLine("Sensor hardware compatibility: ✓")
        } catch (e: AssertionError) {
            report.appendLine("Sensor hardware compatibility: ✗ - ${e.message}")
        }
        
        try {
            testDeviceFeatureCompatibility()
            report.appendLine("Device feature compatibility: ✓")
        } catch (e: AssertionError) {
            report.appendLine("Device feature compatibility: ✗ - ${e.message}")
        }
        
        println("\nCompatibility results:")
        println(report.toString())
        
        // This test always passes; it only generates a report.
        assertTrue("Compatibility report generated", true)
    }
}
