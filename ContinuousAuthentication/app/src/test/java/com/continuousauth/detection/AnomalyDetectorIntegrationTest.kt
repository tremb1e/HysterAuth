package com.continuousauth.detection

import android.app.usage.UsageStats
import android.app.usage.UsageStatsManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
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

/**
 * Integration-style unit tests for the anomaly detector.
 *
 * Covers integration with system services and more complex scenarios.
 */
@ExperimentalCoroutinesApi
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [30])
class AnomalyDetectorIntegrationTest {

    private lateinit var mockContext: Context
    private lateinit var mockUsageStatsManager: UsageStatsManager
    private lateinit var anomalyDetector: AnomalyDetectorImpl
    private lateinit var testDispatcher: TestDispatcher
    private lateinit var testScope: TestScope
    private lateinit var capturedReceiver: BroadcastReceiver

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
        
        // Capture the registered receiver
        every { mockContext.registerReceiver(capture(slot<BroadcastReceiver>()), any()) } answers {
            capturedReceiver = firstArg()
            mockk()
        }
        every { mockContext.unregisterReceiver(any()) } just Runs
        
        anomalyDetector = AnomalyDetectorImpl(mockContext)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    /**
     * Verify end-to-end device unlock flow.
     */
    @Test
    fun testDeviceUnlockCompleteFlow() = testScope.runTest {
        var anomalyDetected = false
        var detectedTrigger: AnomalyTrigger? = null
        
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                anomalyDetected = true
                detectedTrigger = trigger
            }
        }
        
        anomalyDetector.setOnAnomalyListener(listener)
        
        // Start detection (registers receiver)
        anomalyDetector.startDetection()
        
        advanceUntilIdle()
        
        // Verify receiver registered
        verify { mockContext.registerReceiver(any(), any()) }
        
        // Simulate unlock broadcast
        val unlockIntent = Intent(Intent.ACTION_USER_PRESENT)
        capturedReceiver.onReceive(mockContext, unlockIntent)
        
        advanceUntilIdle()
        
        // Verify anomaly detected
        assertTrue("Device unlock anomaly should be detected", anomalyDetected)
        assertTrue("Detected trigger should be device unlock", detectedTrigger is AnomalyTrigger.DeviceUnlocked)
        
        anomalyDetector.stopDetection()
        
        // Verify receiver unregistered
        verify { mockContext.unregisterReceiver(any()) }
    }

    /**
     * Verify device unlock cooldown behavior.
     */
    @Test
    fun testDeviceUnlockCooldown() = testScope.runTest {
        var anomalyCount = 0
        
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                if (trigger is AnomalyTrigger.DeviceUnlocked) {
                    anomalyCount++
                }
            }
        }
        
        // Short cooldown for testing
        val testPolicy = DetectionPolicy(deviceUnlockCooldownMs = 100L)
        anomalyDetector.updatePolicy(testPolicy)
        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()
        
        advanceUntilIdle()
        
        // First unlock
        val unlockIntent = Intent(Intent.ACTION_USER_PRESENT)
        capturedReceiver.onReceive(mockContext, unlockIntent)
        
        advanceUntilIdle()
        assertEquals("First unlock should be detected", 1, anomalyCount)
        
        // Immediate second unlock (should be blocked by cooldown)
        capturedReceiver.onReceive(mockContext, unlockIntent)
        
        advanceUntilIdle()
        assertEquals("Unlocks within cooldown should be ignored", 1, anomalyCount)
        
        // Wait for cooldown
        delay(150L)
        
        // Unlock after cooldown
        capturedReceiver.onReceive(mockContext, unlockIntent)
        
        advanceUntilIdle()
        assertEquals("Second unlock should be detected after cooldown", 2, anomalyCount)
        
        anomalyDetector.stopDetection()
    }

    /**
     * Verify foreground app monitoring.
     */
    @Test
    fun testForegroundAppMonitoring() = testScope.runTest {
        var anomalyDetected = false
        var detectedTrigger: AnomalyTrigger? = null
        
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                anomalyDetected = true
                detectedTrigger = trigger
            }
        }
        
        // Policy includes sensitive apps
        val testPolicy = DetectionPolicy(
            sensitiveApps = setOf("com.sensitive.app", "com.banking.app"),
            appCheckIntervalMs = 50L // Short interval for testing
        )
        
        // Mock UsageStats
        val mockUsageStats = mockk<UsageStats>()
        every { mockUsageStats.packageName } returns "com.sensitive.app"
        every { mockUsageStats.lastTimeUsed } returns System.currentTimeMillis()
        
        every { 
            mockUsageStatsManager.queryUsageStats(
                UsageStatsManager.INTERVAL_DAILY,
                any(),
                any()
            )
        } returns listOf(mockUsageStats)
        
        anomalyDetector.updatePolicy(testPolicy)
        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()
        
        // Wait for app check interval
        advanceTimeBy(100L)
        advanceUntilIdle()
        
        // Verify sensitive app anomaly detected
        assertTrue("Sensitive app entry anomaly should be detected", anomalyDetected)
        assertTrue("Detected trigger should be sensitive app entry", detectedTrigger is AnomalyTrigger.SensitiveAppEntered)
        
        val appTrigger = detectedTrigger as AnomalyTrigger.SensitiveAppEntered
        assertEquals("Detected package name should match", "com.sensitive.app", appTrigger.packageName)
        
        anomalyDetector.stopDetection()
    }

    /**
     * Verify error handling in foreground app monitoring.
     */
    @Test
    fun testForegroundAppMonitoringErrorHandling() = testScope.runTest {
        var anomalyDetected = false
        
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                anomalyDetected = true
            }
        }
        
        // Policy includes sensitive apps
        val testPolicy = DetectionPolicy(
            sensitiveApps = setOf("com.sensitive.app"),
            appCheckIntervalMs = 50L
        )
        
        // Simulate UsageStatsManager throwing an exception
        every { 
            mockUsageStatsManager.queryUsageStats(any(), any(), any())
        } throws SecurityException("Permission denied")
        
        anomalyDetector.updatePolicy(testPolicy)
        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()
        
        // Wait for app check interval
        advanceTimeBy(100L)
        advanceUntilIdle()
        
        // Should not crash, and should not report false positives
        assertFalse("Should not report anomalies when checks fail", anomalyDetected)
        
        anomalyDetector.stopDetection()
    }

    /**
     * Verify concurrent detection of multiple anomaly types.
     */
    @Test
    fun testConcurrentAnomalyDetection() = testScope.runTest {
        val detectedTriggers = mutableListOf<AnomalyTrigger>()
        
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                detectedTriggers.add(trigger)
            }
        }
        
        // Policy enabling multiple detection signals
        val testPolicy = DetectionPolicy(
            enabled = true,
            deviceUnlockEnabled = true,
            sensitiveApps = setOf("com.test.app"),
            appCheckIntervalMs = 50L,
            accelerometerCooldownMs = 100L
        )
        
        anomalyDetector.updatePolicy(testPolicy)
        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()
        
        advanceUntilIdle()
        
        // 1) Trigger accelerometer anomaly
        repeat(15) { index ->
            anomalyDetector.processSensorData(0.1f, 0.2f, 9.8f, System.nanoTime() + index * 1000000L)
        }
        anomalyDetector.processSensorData(10.0f, 8.0f, 15.0f, System.nanoTime())
        
        // 2) Trigger device unlock anomaly
        val unlockIntent = Intent(Intent.ACTION_USER_PRESENT)
        capturedReceiver.onReceive(mockContext, unlockIntent)
        
        advanceUntilIdle()
        
        // Verify multiple anomalies detected
        assertTrue("Should detect at least two anomalies", detectedTriggers.size >= 2)
        
        val hasAccelerometerAnomaly = detectedTriggers.any { it is AnomalyTrigger.AccelerometerSpike }
        val hasDeviceUnlockAnomaly = detectedTriggers.any { it is AnomalyTrigger.DeviceUnlocked }
        
        assertTrue("Should detect accelerometer anomaly", hasAccelerometerAnomaly)
        assertTrue("Should detect device unlock anomaly", hasDeviceUnlockAnomaly)
        
        anomalyDetector.stopDetection()
    }

    /**
     * Verify behavior when policy is disabled.
     */
    @Test
    fun testDisabledPolicyBehavior() = testScope.runTest {
        var anomalyDetected = false
        
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                anomalyDetected = true
            }
        }
        
        // Disabled policy
        val disabledPolicy = DetectionPolicy(
            enabled = false,
            deviceUnlockEnabled = false
        )
        
        anomalyDetector.updatePolicy(disabledPolicy)
        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()
        
        // When disabled, detection should not register receivers
        verify(exactly = 0) { mockContext.registerReceiver(any(), any()) }
        
        // Try processing sensor data
        repeat(15) { index ->
            anomalyDetector.processSensorData(0.1f, 0.2f, 9.8f, System.nanoTime() + index * 1000000L)
        }
        anomalyDetector.processSensorData(20.0f, 20.0f, 20.0f, System.nanoTime())
        
        advanceUntilIdle()
        
        // No anomalies should be detected while disabled
        assertFalse("Should not detect anomalies when policy is disabled", anomalyDetected)
        
        anomalyDetector.stopDetection()
    }

    /**
     * Verify broadcast receiver error handling.
     */
    @Test
    fun testBroadcastReceiverErrorHandling() = testScope.runTest {
        var anomalyDetected = false
        
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                anomalyDetected = true
            }
        }
        
        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()
        
        advanceUntilIdle()
        
        // Send invalid Intent (null)
        capturedReceiver.onReceive(mockContext, null)
        
        // Send wrong action
        val wrongIntent = Intent("wrong.action")
        capturedReceiver.onReceive(mockContext, wrongIntent)
        
        advanceUntilIdle()
        
        // Wrong broadcasts should not trigger detection
        assertFalse("Wrong broadcasts should not trigger anomalies", anomalyDetected)
        
        // Correct broadcast should work
        val correctIntent = Intent(Intent.ACTION_USER_PRESENT)
        capturedReceiver.onReceive(mockContext, correctIntent)
        
        advanceUntilIdle()
        
        assertTrue("Correct broadcast should trigger an anomaly", anomalyDetected)
        
        anomalyDetector.stopDetection()
    }

    /**
     * Verify resource cleanup.
     */
    @Test
    fun testResourceCleanup() = testScope.runTest {
        anomalyDetector.startDetection()
        
        advanceUntilIdle()
        
        // Verify resources allocated
        verify { mockContext.registerReceiver(any(), any()) }
        
        // Cleanup
        anomalyDetector.cleanup()
        
        advanceUntilIdle()
        
        // Verify detection stopped
        assertFalse("After cleanup, detection should be stopped", anomalyDetector.isDetecting())
        
        // Verify receiver unregistered
        verify { mockContext.unregisterReceiver(any()) }
    }

    /**
     * Verify behavior under high-volume sensor data.
     */
    @Test
    fun testHighVolumeDataProcessing() = testScope.runTest {
        var anomalyCount = 0
        
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                if (trigger is AnomalyTrigger.AccelerometerSpike) {
                    anomalyCount++
                }
            }
        }
        
        // Use a permissive policy to allow detection
        val testPolicy = DetectionPolicy(
            accelerometerSpikeThreshold = 2.0f,
            accelerometerCooldownMs = 50L
        )
        
        anomalyDetector.updatePolicy(testPolicy)
        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()
        
        // Send a lot of data points
        val startTime = System.currentTimeMillis()
        
        // Build baseline
        repeat(100) { index ->
            anomalyDetector.processSensorData(
                x = 0.1f + (index % 10) * 0.01f,
                y = 0.2f + (index % 10) * 0.01f,
                z = 9.8f + (index % 10) * 0.01f,
                timestamp = System.nanoTime() + index * 1000000L
            )
        }
        
        // Inject some spikes
        repeat(5) { spikeIndex ->
            delay(60L) // wait for cooldown
            anomalyDetector.processSensorData(
                x = 10.0f + spikeIndex,
                y = 10.0f + spikeIndex,
                z = 10.0f + spikeIndex,
                timestamp = System.nanoTime() + (100 + spikeIndex) * 1000000L
            )
        }
        
        advanceUntilIdle()
        
        val endTime = System.currentTimeMillis()
        val duration = endTime - startTime
        
        // Verify performance and correctness
        assertTrue("High-volume processing should finish in a reasonable time", duration < 1000L) // within 1s
        assertTrue("Should detect some anomalies", anomalyCount > 0)
        assertTrue("Should not have too many false positives", anomalyCount <= 5)
        
        anomalyDetector.stopDetection()
    }
}
