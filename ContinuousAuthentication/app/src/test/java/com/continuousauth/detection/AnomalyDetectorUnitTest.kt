package com.continuousauth.detection

import android.app.usage.UsageStatsManager
import android.content.Context
import io.mockk.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.test.*
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlin.math.sqrt

/**
 * Unit tests for the anomaly detector.
 *
 * Validates core logic and edge cases of [AnomalyDetectorImpl].
 */
@ExperimentalCoroutinesApi
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [30])
class AnomalyDetectorUnitTest {

    private lateinit var mockContext: Context
    private lateinit var mockUsageStatsManager: UsageStatsManager
    private lateinit var anomalyDetector: AnomalyDetectorImpl
    private lateinit var testDispatcher: TestDispatcher
    private lateinit var testScope: TestScope

    @Before
    fun setup() {
        testDispatcher = StandardTestDispatcher()
        testScope = TestScope(testDispatcher)
        Dispatchers.setMain(testDispatcher)
        
        // Create mocks
        mockContext = mockk(relaxed = true)
        mockUsageStatsManager = mockk(relaxed = true)
        
        // Configure Context mocks
        every { mockContext.getSystemService(Context.USAGE_STATS_SERVICE) } returns mockUsageStatsManager
        every { mockContext.registerReceiver(any(), any()) } just Runs
        every { mockContext.unregisterReceiver(any()) } just Runs
        
        anomalyDetector = AnomalyDetectorImpl(mockContext)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    /**
     * Verify initial detector state.
     */
    @Test
    fun testInitialState() {
        assertFalse("Initial state should not be detecting", anomalyDetector.isDetecting())
    }

    /**
     * Verify start/stop detection flow.
     */
    @Test
    fun testStartStopDetection() = testScope.runTest {
        // Start
        anomalyDetector.startDetection()
        assertTrue("After start, detector should be active", anomalyDetector.isDetecting())
        
        // Verify receiver registration
        verify { mockContext.registerReceiver(any(), any()) }
        
        // Stop
        anomalyDetector.stopDetection()
        assertFalse("After stop, detector should be inactive", anomalyDetector.isDetecting())
        
        // Verify receiver unregistration
        verify { mockContext.unregisterReceiver(any()) }
    }

    /**
     * Verify repeated start/stop calls are safe.
     */
    @Test
    fun testRepeatedStartStop() = testScope.runTest {
        // Repeated start should be safe
        anomalyDetector.startDetection()
        anomalyDetector.startDetection()
        assertTrue("After repeated start, detector should be active", anomalyDetector.isDetecting())
        
        // Repeated stop should be safe
        anomalyDetector.stopDetection()
        anomalyDetector.stopDetection()
        assertFalse("After repeated stop, detector should be inactive", anomalyDetector.isDetecting())
    }

    /**
     * Verify policy updates.
     */
    @Test
    fun testPolicyUpdate() = testScope.runTest {
        val initialPolicy = DetectionPolicy(
            accelerometerSpikeThreshold = 2.0f,
            sensitiveApps = setOf("com.test.app1")
        )
        
        anomalyDetector.updatePolicy(initialPolicy)
        
        val updatedPolicy = DetectionPolicy(
            accelerometerSpikeThreshold = 5.0f,
            sensitiveApps = setOf("com.test.app1", "com.test.app2"),
            deviceUnlockEnabled = false
        )
        
        anomalyDetector.updatePolicy(updatedPolicy)
        
        // Policy update should succeed (no exception)
        assertTrue("Policy update should succeed", true)
    }

    /**
     * Verify policy updates while detecting.
     */
    @Test
    fun testPolicyUpdateWhileDetecting() = testScope.runTest {
        // Start detection
        anomalyDetector.startDetection()
        assertTrue("Detector should be active", anomalyDetector.isDetecting())
        
        // Update policy - disable device unlock detection
        val updatedPolicy = DetectionPolicy(deviceUnlockEnabled = false)
        anomalyDetector.updatePolicy(updatedPolicy)
        
        // Should still be detecting
        assertTrue("Detector should remain active after policy update", anomalyDetector.isDetecting())
        
        anomalyDetector.stopDetection()
    }

    /**
     * Verify accelerometer processing - normal data.
     */
    @Test
    fun testAccelerometerDataProcessing_NormalData() = testScope.runTest {
        var anomalyDetected = false
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                anomalyDetected = true
            }
        }
        
        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()
        
        // Send normal accelerometer data
        repeat(10) { index ->
            val x = 0.1f + index * 0.01f
            val y = 0.2f + index * 0.01f
            val z = 9.8f + index * 0.01f
            anomalyDetector.processSensorData(x, y, z, System.nanoTime() + index * 1000000L)
        }
        
        advanceUntilIdle()
        
        // Normal data should not trigger anomalies
        assertFalse("Normal data should not trigger anomalies", anomalyDetected)
        
        anomalyDetector.stopDetection()
    }

    /**
     * Verify accelerometer processing - spike data.
     */
    @Test
    fun testAccelerometerDataProcessing_SpikeData() = testScope.runTest {
        var anomalyDetected = false
        var detectedTrigger: AnomalyTrigger? = null
        
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                anomalyDetected = true
                detectedTrigger = trigger
            }
        }
        
        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()
        
        // Build baseline
        repeat(15) { index ->
            val x = 0.1f
            val y = 0.2f
            val z = 9.8f
            anomalyDetector.processSensorData(x, y, z, System.nanoTime() + index * 1000000L)
        }
        
        // Send spike data
        val spikeX = 10.0f
        val spikeY = 8.0f
        val spikeZ = 15.0f
        anomalyDetector.processSensorData(spikeX, spikeY, spikeZ, System.nanoTime() + 16 * 1000000L)
        
        advanceUntilIdle()
        
        // Spike data should trigger an anomaly
        assertTrue("Spike data should trigger an anomaly", anomalyDetected)
        assertTrue("Should detect accelerometer spike", detectedTrigger is AnomalyTrigger.AccelerometerSpike)
        
        val spike = detectedTrigger as AnomalyTrigger.AccelerometerSpike
        val expectedMagnitude = sqrt(spikeX * spikeX + spikeY * spikeY + spikeZ * spikeZ)
        assertEquals("Spike magnitude should be computed correctly", expectedMagnitude, spike.magnitude, 0.01f)
        
        anomalyDetector.stopDetection()
    }

    /**
     * Verify accelerometer cooldown behavior.
     */
    @Test
    fun testAccelerometerCooldownPeriod() = testScope.runTest {
        var anomalyCount = 0
        
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                if (trigger is AnomalyTrigger.AccelerometerSpike) {
                    anomalyCount++
                }
            }
        }
        
        // Use a short cooldown for testing
        val testPolicy = DetectionPolicy(
            accelerometerCooldownMs = 100L,
            accelerometerSpikeThreshold = 2.0f
        )
        
        anomalyDetector.updatePolicy(testPolicy)
        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()
        
        // Build baseline
        repeat(15) { index ->
            anomalyDetector.processSensorData(0.1f, 0.2f, 9.8f, System.nanoTime() + index * 1000000L)
        }
        
        val baseTime = System.nanoTime()
        
        // Rapid spikes (should be limited by cooldown)
        anomalyDetector.processSensorData(10.0f, 8.0f, 15.0f, baseTime + 16 * 1000000L)
        anomalyDetector.processSensorData(10.0f, 8.0f, 15.0f, baseTime + 17 * 1000000L) // within cooldown
        anomalyDetector.processSensorData(10.0f, 8.0f, 15.0f, baseTime + 18 * 1000000L) // within cooldown
        
        advanceUntilIdle()
        
        // Should trigger only once due to cooldown
        assertEquals("Should trigger only one anomaly within cooldown", 1, anomalyCount)
        
        // After cooldown, it should trigger again
        delay(150L) // beyond cooldown
        anomalyDetector.processSensorData(10.0f, 8.0f, 15.0f, System.nanoTime())
        
        advanceUntilIdle()
        
        // Now it should trigger a second time
        assertEquals("Should trigger again after cooldown", 2, anomalyCount)
        
        anomalyDetector.stopDetection()
    }

    /**
     * Verify listener registration and callbacks.
     */
    @Test
    fun testAnomalyListenerCallback() = testScope.runTest {
        var callbackCount = 0
        var lastTrigger: AnomalyTrigger? = null
        
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                callbackCount++
                lastTrigger = trigger
            }
            
            override fun onAnomalyCleared(trigger: AnomalyTrigger) {
                // Optional callback
            }
        }
        
        // Set listener
        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()
        
        // Build baseline and send spike
        repeat(15) { index ->
            anomalyDetector.processSensorData(0.1f, 0.2f, 9.8f, System.nanoTime() + index * 1000000L)
        }
        anomalyDetector.processSensorData(10.0f, 8.0f, 15.0f, System.nanoTime() + 16 * 1000000L)
        
        advanceUntilIdle()
        
        assertEquals("Callback should be invoked once", 1, callbackCount)
        assertTrue("Callback trigger should be accelerometer spike", lastTrigger is AnomalyTrigger.AccelerometerSpike)
        
        // Clear listener
        anomalyDetector.setOnAnomalyListener(null)
        
        // Send spike again
        delay(200L) // wait for cooldown
        anomalyDetector.processSensorData(10.0f, 8.0f, 15.0f, System.nanoTime())
        
        advanceUntilIdle()
        
        // Callback count should not increase
        assertEquals("No new callbacks after clearing listener", 1, callbackCount)
        
        anomalyDetector.stopDetection()
    }

    /**
     * Verify different policy configurations.
     */
    @Test
    fun testDifferentPolicyConfigurations() = testScope.runTest {
        // Disable all detection
        val disabledPolicy = DetectionPolicy(enabled = false)
        anomalyDetector.updatePolicy(disabledPolicy)
        
        var anomalyDetected = false
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                anomalyDetected = true
            }
        }
        
        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()
        
        // Send an obvious spike
        repeat(15) { index ->
            anomalyDetector.processSensorData(0.1f, 0.2f, 9.8f, System.nanoTime() + index * 1000000L)
        }
        anomalyDetector.processSensorData(20.0f, 20.0f, 20.0f, System.nanoTime())
        
        advanceUntilIdle()
        
        // Should not detect anomalies while disabled
        assertFalse("No anomalies should be detected when disabled", anomalyDetected)
        
        anomalyDetector.stopDetection()
    }

    /**
     * Edge case: insufficient data.
     */
    @Test
    fun testEdgeCase_EmptyData() = testScope.runTest {
        var anomalyDetected = false
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                anomalyDetected = true
            }
        }
        
        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()
        
        // Send too few points to build a baseline
        anomalyDetector.processSensorData(10.0f, 10.0f, 10.0f, System.nanoTime())
        anomalyDetector.processSensorData(20.0f, 20.0f, 20.0f, System.nanoTime() + 1000000L)
        
        advanceUntilIdle()
        
        // Should not trigger with insufficient data
        assertFalse("Should not trigger anomalies with insufficient data", anomalyDetected)
        
        anomalyDetector.stopDetection()
    }

    /**
     * Edge case: zero-magnitude baseline.
     */
    @Test
    fun testEdgeCase_ZeroMagnitudeData() = testScope.runTest {
        var anomalyDetected = false
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                anomalyDetected = true
            }
        }
        
        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()
        
        // Build a zero-magnitude baseline
        repeat(20) { index ->
            anomalyDetector.processSensorData(0.0f, 0.0f, 0.0f, System.nanoTime() + index * 1000000L)
        }
        
        // Send a small change
        anomalyDetector.processSensorData(0.1f, 0.1f, 0.1f, System.nanoTime() + 21 * 1000000L)
        
        advanceUntilIdle()
        
        // A small change on a zero baseline should not trigger (stdDev check)
        assertFalse("Small changes on a zero baseline should not trigger anomalies", anomalyDetected)
        
        anomalyDetector.stopDetection()
    }

    /**
     * Verify different anomaly trigger types.
     */
    @Test
    fun testMultipleAnomalyTypes() {
        // Device unlock trigger
        val deviceUnlock = AnomalyTrigger.DeviceUnlocked
        assertTrue("Device unlock trigger should be correct type", deviceUnlock is AnomalyTrigger.DeviceUnlocked)
        
        // Accelerometer spike trigger
        val accelerometerSpike = AnomalyTrigger.AccelerometerSpike(
            magnitude = 15.0f,
            threshold = 10.0f,
            deviation = 2.5f
        )
        assertTrue("Accelerometer spike trigger should be correct type", accelerometerSpike is AnomalyTrigger.AccelerometerSpike)
        assertEquals("Magnitude should match", 15.0f, accelerometerSpike.magnitude, 0.01f)
        assertEquals("Threshold should match", 10.0f, accelerometerSpike.threshold, 0.01f)
        assertEquals("Deviation should match", 2.5f, accelerometerSpike.deviation, 0.01f)
        
        // Sensitive app trigger
        val sensitiveApp = AnomalyTrigger.SensitiveAppEntered(
            packageName = "com.test.app",
            appName = "Test App"
        )
        assertTrue("Sensitive app trigger should be correct type", sensitiveApp is AnomalyTrigger.SensitiveAppEntered)
        assertEquals("Package name should match", "com.test.app", sensitiveApp.packageName)
        assertEquals("App name should match", "Test App", sensitiveApp.appName)
    }

    /**
     * Verify default detection policy values.
     */
    @Test
    fun testDetectionPolicyDefaults() {
        val defaultPolicy = DetectionPolicy()
        
        assertEquals("Default accelerometer threshold should match", 3.0f, defaultPolicy.accelerometerSpikeThreshold, 0.01f)
        assertEquals("Default window size should match", 20, defaultPolicy.accelerometerWindowSize)
        assertEquals("Default cooldown should match", 2000L, defaultPolicy.accelerometerCooldownMs)
        assertTrue("Anomaly detection should be enabled by default", defaultPolicy.enabled)
        assertTrue("Device unlock detection should be enabled by default", defaultPolicy.deviceUnlockEnabled)
        assertFalse("Debug mode should be disabled by default", defaultPolicy.debugMode)
        assertTrue("Sensitive apps list should be empty by default", defaultPolicy.sensitiveApps.isEmpty())
    }

    /**
     * Verify custom detection policy values.
     */
    @Test
    fun testDetectionPolicyCustomValues() {
        val customPolicy = DetectionPolicy(
            accelerometerSpikeThreshold = 5.0f,
            accelerometerWindowSize = 30,
            accelerometerCooldownMs = 1000L,
            sensitiveApps = setOf("app1", "app2", "app3"),
            appCheckIntervalMs = 2000L,
            deviceUnlockEnabled = false,
            deviceUnlockCooldownMs = 3000L,
            enabled = true,
            debugMode = true
        )
        
        assertEquals("Custom accelerometer threshold should match", 5.0f, customPolicy.accelerometerSpikeThreshold, 0.01f)
        assertEquals("Custom window size should match", 30, customPolicy.accelerometerWindowSize)
        assertEquals("Custom cooldown should match", 1000L, customPolicy.accelerometerCooldownMs)
        assertEquals("Custom sensitive apps count should match", 3, customPolicy.sensitiveApps.size)
        assertTrue("Custom sensitive apps list should contain app1", customPolicy.sensitiveApps.contains("app1"))
        assertEquals("Custom app check interval should match", 2000L, customPolicy.appCheckIntervalMs)
        assertFalse("Custom device unlock detection should be disabled", customPolicy.deviceUnlockEnabled)
        assertEquals("Custom device unlock cooldown should match", 3000L, customPolicy.deviceUnlockCooldownMs)
        assertTrue("Custom anomaly detection should be enabled", customPolicy.enabled)
        assertTrue("Custom debug mode should be enabled", customPolicy.debugMode)
    }
}
