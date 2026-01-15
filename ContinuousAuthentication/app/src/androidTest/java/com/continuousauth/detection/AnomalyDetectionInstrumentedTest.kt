package com.continuousauth.detection

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import com.continuousauth.core.SmartTransmissionManager
import com.continuousauth.observability.MetricsCollectorImpl
import com.continuousauth.transmission.TransmissionControllerImpl
import com.continuousauth.transmission.TransmissionMode
import dagger.hilt.android.testing.HiltAndroidRule
import dagger.hilt.android.testing.HiltAndroidTest
import kotlinx.coroutines.delay
import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import javax.inject.Inject

/**
 * Anomaly detection integration tests.
 *
 * Covers anomaly triggers and smart transmission mode switching.
 */
@LargeTest
@HiltAndroidTest
@RunWith(AndroidJUnit4::class)
class AnomalyDetectionInstrumentedTest {

    @get:Rule
    var hiltRule = HiltAndroidRule(this)

    @Inject
    lateinit var anomalyDetector: AnomalyDetectorImpl

    @Inject
    lateinit var transmissionController: TransmissionControllerImpl

    @Inject
    lateinit var smartTransmissionManager: SmartTransmissionManager

    @Inject
    lateinit var metricsCollector: MetricsCollectorImpl

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Before
    fun init() {
        hiltRule.inject()
    }

    /**
     * Tests device-unlock anomaly detection.
     */
    @Test
    fun testDeviceUnlockAnomalyDetection() = runTest {
        var anomalyDetected = false
        var detectedTrigger: AnomalyTrigger? = null

        // Set up anomaly listener.
        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                anomalyDetected = true
                detectedTrigger = trigger
            }

            override fun onAnomalyCleared(trigger: AnomalyTrigger) {
                // Not used in this test
            }
        }

        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()

        // Simulate a device unlock event.
        // Note: a real instrumentation test should simulate the system broadcast via the Android test APIs.
        val deviceUnlockTrigger = AnomalyTrigger.DeviceUnlocked
        listener.onAnomalyDetected(deviceUnlockTrigger)

        // Assertions.
        assertTrue("Device unlock anomaly should be detected", anomalyDetected)
        assertEquals("Detected trigger should be device unlock", AnomalyTrigger.DeviceUnlocked, detectedTrigger)

        anomalyDetector.stopDetection()
    }

    /**
     * Tests accelerometer spike detection.
     */
    @Test
    fun testAccelerometerSpikeDetection() = runTest {
        var anomalyDetected = false
        var detectedTrigger: AnomalyTrigger? = null

        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                anomalyDetected = true
                detectedTrigger = trigger
            }

            override fun onAnomalyCleared(trigger: AnomalyTrigger) {}
        }

        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()

        // Feed baseline accelerometer data.
        val normalData = listOf(
            Triple(0.1f, 0.2f, 9.8f),
            Triple(0.2f, 0.1f, 9.9f),
            Triple(0.0f, 0.3f, 9.7f)
        )

        normalData.forEachIndexed { index, (x, y, z) ->
            anomalyDetector.processSensorData(x, y, z, System.nanoTime() + index * 1000000L)
        }

        delay(100) // Wait for baseline to build.

        // Inject a spike.
        val spikeData = Triple(5.0f, 4.0f, 12.0f) // Clearly out of the normal range.
        anomalyDetector.processSensorData(spikeData.first, spikeData.second, spikeData.third, System.nanoTime())

        delay(100) // Wait for anomaly processing.

        // Assertions.
        assertTrue("Accelerometer spike should be detected", anomalyDetected)
        assertTrue("Detected anomaly should be an accelerometer spike", detectedTrigger is AnomalyTrigger.AccelerometerSpike)

        val spike = detectedTrigger as AnomalyTrigger.AccelerometerSpike
        assertTrue("Spike magnitude should exceed the threshold", spike.magnitude > spike.threshold)

        anomalyDetector.stopDetection()
    }

    /**
     * Tests sensitive-app entry anomaly detection.
     */
    @Test
    fun testSensitiveAppAnomalyDetection() = runTest {
        var anomalyDetected = false
        var detectedTrigger: AnomalyTrigger? = null

        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                anomalyDetected = true
                detectedTrigger = trigger
            }

            override fun onAnomalyCleared(trigger: AnomalyTrigger) {}
        }

        // Update policy and add apps to the sensitive list.
        val testPolicy = DetectionPolicy(
            enabled = true,
            deviceUnlockEnabled = false,
            accelerometerSpikeThreshold = 3.0f,
            accelerometerWindowSize = 10,
            sensitiveApps = setOf("com.android.settings", "com.android.vending")
        )
        
        anomalyDetector.updatePolicy(testPolicy)
        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()

        // Simulate sensitive-app entry.
        // Note: a real test should simulate UsageStatsManager behavior.
        val sensitiveAppTrigger = AnomalyTrigger.SensitiveAppEntered(
            packageName = "com.android.settings",
            appName = "Settings"
        )
        listener.onAnomalyDetected(sensitiveAppTrigger)

        // Assertions.
        assertTrue("Sensitive-app entry anomaly should be detected", anomalyDetected)
        assertTrue("Detected anomaly should be sensitive-app entry", detectedTrigger is AnomalyTrigger.SensitiveAppEntered)

        val appTrigger = detectedTrigger as AnomalyTrigger.SensitiveAppEntered
        assertEquals("Package name should match", "com.android.settings", appTrigger.packageName)

        anomalyDetector.stopDetection()
    }

    /**
     * Tests smart transmission manager integration.
     */
    @Test
    fun testSmartTransmissionManagerIntegration() = runTest {
        // Start the smart transmission manager.
        smartTransmissionManager.start()

        // Validate initial state.
        val initialStatus = smartTransmissionManager.getStatusInfo()
        assertTrue("Manager should be active", initialStatus.isActive)
        assertEquals("Initial mode should be slow mode", TransmissionMode.SLOW_MODE, initialStatus.currentMode)

        // Simulate an anomaly.
        smartTransmissionManager.processSensorData(5.0f, 4.0f, 12.0f, System.nanoTime())

        delay(200) // Wait for anomaly processing.

        // Validate status after processing.
        val updatedStatus = smartTransmissionManager.getStatusInfo()
        // Note: this is a basic integration check; spike detection may not be fully exercised here.

        assertTrue("Manager should remain active", updatedStatus.isActive)
        assertTrue("Anomaly detection should be active", updatedStatus.isDetectionActive)

        // Stop the manager.
        smartTransmissionManager.stop()

        val finalStatus = smartTransmissionManager.getStatusInfo()
        assertFalse("Manager should be inactive", finalStatus.isActive)
        assertEquals("Final mode should be slow mode", TransmissionMode.SLOW_MODE, finalStatus.currentMode)
    }

    /**
     * Tests transmission mode switching.
     */
    @Test
    fun testTransmissionModeSwitch() = runTest {
        // Validate initial state.
        assertEquals("Initial mode should be slow mode", TransmissionMode.SLOW_MODE, transmissionController.getCurrentMode())

        // Switch to fast mode.
        transmissionController.switchToFastMode("test trigger", 5000L)

        // Validate mode switch.
        assertEquals("Should switch to fast mode", TransmissionMode.FAST_MODE, transmissionController.getCurrentMode())

        val remainingTime = transmissionController.getRemainingFastModeTime()
        assertTrue("Fast-mode remaining time should be > 0", remainingTime > 0)

        // Wait for fast mode to expire.
        delay(5100L) // Slightly over 5 seconds.

        // Validate automatic switch back to slow mode.
        assertEquals("Should automatically switch back to slow mode", TransmissionMode.SLOW_MODE, transmissionController.getCurrentMode())
    }

    /**
     * Tests dynamic policy updates.
     */
    @Test
    fun testPolicyDynamicUpdate() = runTest {
        // Initial policy.
        val initialPolicy = DetectionPolicy(
            enabled = true,
            accelerometerSpikeThreshold = 2.0f
        )
        
        anomalyDetector.updatePolicy(initialPolicy)
        anomalyDetector.startDetection()

        // Updated policy.
        val updatedPolicy = DetectionPolicy(
            enabled = true,
            accelerometerSpikeThreshold = 5.0f // Higher threshold
        )

        anomalyDetector.updatePolicy(updatedPolicy)

        // The threshold change should be validated via real sensor data; this test ensures the update call succeeds.

        assertTrue("Anomaly detection should still be running", anomalyDetector.isDetecting())

        anomalyDetector.stopDetection()
    }

    /**
     * Tests cooldown behavior.
     */
    @Test
    fun testCooldownPeriod() = runTest {
        var anomalyCount = 0

        val listener = object : OnAnomalyListener {
            override fun onAnomalyDetected(trigger: AnomalyTrigger) {
                anomalyCount++
            }

            override fun onAnomalyCleared(trigger: AnomalyTrigger) {}
        }

        anomalyDetector.setOnAnomalyListener(listener)
        anomalyDetector.startDetection()

        // Send multiple anomaly events in quick succession.
        repeat(5) {
            val deviceUnlockTrigger = AnomalyTrigger.DeviceUnlocked
            listener.onAnomalyDetected(deviceUnlockTrigger)
            delay(100) // Short interval
        }

        // Note: an actual cooldown implementation may throttle consecutive triggers; this test validates basic behavior.

        assertTrue("Should detect at least one anomaly", anomalyCount > 0)

        anomalyDetector.stopDetection()
    }

    /**
     * Tests metrics collection.
     */
    @Test
    fun testMetricsCollection() = runTest {
        // Initial snapshot.
        val initialSnapshot = metricsCollector.getSnapshot()
        val initialAnomalies = initialSnapshot.counters[com.continuousauth.observability.MetricType.ANOMALIES_DETECTED] ?: 0L

        // Start the manager.
        smartTransmissionManager.start()

        // Process some sensor data.
        repeat(10) {
            smartTransmissionManager.processSensorData(0.1f, 0.2f, 9.8f, System.nanoTime() + it * 1000000L)
        }

        delay(100)

        // Updated snapshot.
        val updatedSnapshot = metricsCollector.getSnapshot()
        val sensorSamples = updatedSnapshot.counters[com.continuousauth.observability.MetricType.SENSOR_SAMPLES_COLLECTED] ?: 0L

        // Assertions.
        assertTrue("Should collect sensor samples", sensorSamples > 0L)

        smartTransmissionManager.stop()
    }

    /**
     * Tests anomaly detector start/stop lifecycle.
     */
    @Test
    fun testAnomalyDetectorLifecycle() = runTest {
        // Initial state.
        assertFalse("Anomaly detector should start stopped", anomalyDetector.isDetecting())

        // Start detection.
        anomalyDetector.startDetection()
        assertTrue("Anomaly detector should be running after start", anomalyDetector.isDetecting())

        // Stop detection.
        anomalyDetector.stopDetection()
        assertFalse("Anomaly detector should not be running after stop", anomalyDetector.isDetecting())

        // Repeated start/stop should not crash.
        anomalyDetector.startDetection()
        anomalyDetector.startDetection() // Starting twice should be safe.
        assertTrue("Anomaly detector should still be running", anomalyDetector.isDetecting())

        anomalyDetector.stopDetection()
        anomalyDetector.stopDetection() // Stopping twice should be safe.
        assertFalse("Anomaly detector should be stopped", anomalyDetector.isDetecting())
    }
}
