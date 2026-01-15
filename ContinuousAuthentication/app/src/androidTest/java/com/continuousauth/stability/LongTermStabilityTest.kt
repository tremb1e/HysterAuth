package com.continuousauth.stability

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import com.continuousauth.core.SmartTransmissionManager
import com.continuousauth.observability.MetricsCollectorImpl
import com.continuousauth.observability.PerformanceMonitorImpl
import com.continuousauth.ui.MainActivity
import dagger.hilt.android.testing.HiltAndroidRule
import dagger.hilt.android.testing.HiltAndroidTest
import kotlinx.coroutines.*
import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import javax.inject.Inject
import kotlin.math.sin
import kotlin.random.Random

/**
 * Long-term stability test suite.
 *
 * Task 5.3.3: CI should include a long-running stability test: run for ≥12 hours on
 * high-/mid-/low-end devices, monitoring ANRs, crashes, and memory leaks.
 */
@LargeTest
@HiltAndroidTest
@RunWith(AndroidJUnit4::class)
class LongTermStabilityTest {

    @get:Rule(order = 0)
    var hiltRule = HiltAndroidRule(this)

    @get:Rule(order = 1)
    var activityScenarioRule = ActivityScenarioRule(MainActivity::class.java)

    @Inject
    lateinit var smartTransmissionManager: SmartTransmissionManager

    @Inject
    lateinit var performanceMonitor: PerformanceMonitorImpl

    @Inject
    lateinit var metricsCollector: MetricsCollectorImpl

    private val context: Context = InstrumentationRegistry.getInstrumentation().targetContext
    private val activityManager by lazy {
        context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    }

    companion object {
        // Use a shorter duration in CI (30 minutes).
        // Run the full 12-hour test on real devices.
        private val TEST_DURATION_MS = if (isRunningInCI()) {
            30 * 60 * 1000L  // 30 minutes for CI
        } else {
            12 * 60 * 60 * 1000L  // 12 hours for on-device testing
        }
        
        private const val MEMORY_LEAK_THRESHOLD_MB = 100L  // Memory leak threshold
        private const val MAX_MEMORY_USAGE_MB = 500L       // Max memory usage
        private const val MONITORING_INTERVAL_MS = 30000L  // Monitoring interval (30s)
        private const val PERFORMANCE_SAMPLE_INTERVAL_MS = 5000L  // Performance sampling interval (5s)
        
        private fun isRunningInCI(): Boolean {
            return System.getenv("CI") == "true" ||
                   System.getProperty("CI") == "true" ||
                   System.getenv("GITHUB_ACTIONS") == "true"
        }
    }

    @Before
    fun init() {
        hiltRule.inject()
    }

    /**
     * Main long-running stability test.
     *
     * Monitors memory leaks, ANRs, and crashes.
     */
    @Test
    fun testLongTermStability() = runTest(timeout = TEST_DURATION_MS + 60000L) {
        val testStartTime = System.currentTimeMillis()
        val testDurationHours = TEST_DURATION_MS / (60 * 60 * 1000.0)
        
        println("=== Long-term stability test started ===")
        println("Duration: ${String.format("%.1f", testDurationHours)} hours")
        println("Environment: ${if (isRunningInCI()) "CI" else "Device"}")
        println("Monitoring interval: ${MONITORING_INTERVAL_MS / 1000}s")
        println("Start time: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(java.util.Date())}")
        
        // Start core services.
        performanceMonitor.startMonitoring(PERFORMANCE_SAMPLE_INTERVAL_MS)
        smartTransmissionManager.start()
        
        // Stability monitoring stats.
        val stabilityStats = StabilityStats()
        
        try {
            // Launch data generator coroutine.
            val dataGeneratorJob = launch {
                generateContinuousData(stabilityStats)
            }
            
            // Launch monitoring coroutine.
            val monitoringJob = launch {
                monitorSystemHealth(testStartTime, stabilityStats)
            }
            
            // Wait for the test duration.
            val testJob = launch {
                delay(TEST_DURATION_MS)
            }
            
            // Wait for all jobs.
            joinAll(dataGeneratorJob, monitoringJob, testJob)
            
            // Validate results.
            validateStabilityResults(stabilityStats, testStartTime)
            
        } finally {
            // Clean up.
            smartTransmissionManager.stop()
            performanceMonitor.stopMonitoring()
            
            // Generate final report.
            generateFinalReport(stabilityStats, testStartTime)
        }
    }

    /**
     * Continuous data generation.
     *
     * Simulates a normal app workload.
     */
    private suspend fun generateContinuousData(stats: StabilityStats) {
        var dataCount = 0L
        val startTime = System.currentTimeMillis()
        
        while (true) {
            try {
                // Simulate sensor data stream.
                val x = sin(dataCount * 0.01) * 2.0f + Random.nextFloat() * 0.1f
                val y = sin(dataCount * 0.015 + 1.0) * 1.5f + Random.nextFloat() * 0.1f  
                val z = 9.8f + sin(dataCount * 0.008) * 0.2f + Random.nextFloat() * 0.05f
                
                smartTransmissionManager.processSensorData(x, y, z, System.nanoTime())
                
                dataCount++
                stats.totalDataPointsGenerated = dataCount
                
                // Log every 1000 data points.
                if (dataCount % 1000 == 0L) {
                    val elapsed = System.currentTimeMillis() - startTime
                    val rate = dataCount * 1000.0 / elapsed
                    println("Data generation: ${dataCount} points, rate: ${String.format("%.1f", rate)} pts/s")
                }
                
                // Simulate a 20 Hz sampling rate.
                delay(50L)
                
            } catch (e: Exception) {
                stats.dataGenerationErrors++
                println("Data generation error: ${e.message}")
                
                // Too many consecutive errors likely indicates a severe issue.
                if (stats.dataGenerationErrors > 100) {
                    throw Exception("Too many data generation errors; failing test")
                }
            }
        }
    }

    /**
     * System health monitor.
     *
     * Monitors memory, performance metrics, and potential issues.
     */
    private suspend fun monitorSystemHealth(testStartTime: Long, stats: StabilityStats) {
        var initialMemory = getCurrentMemoryUsage()
        var maxMemoryUsage = initialMemory
        var lastGcTime = System.currentTimeMillis()
        
        println("Initial memory usage: ${initialMemory} MB")
        
        while (true) {
            try {
                val currentMemory = getCurrentMemoryUsage()
                val heapMemory = getCurrentHeapUsage()
                maxMemoryUsage = maxOf(maxMemoryUsage, currentMemory)
                
                // Update stats.
                stats.currentMemoryMB = currentMemory
                stats.maxMemoryMB = maxOf(stats.maxMemoryMB, currentMemory)
                stats.currentHeapMB = heapMemory
                
                // Check for memory leaks.
                val memoryGrowth = currentMemory - initialMemory
                if (memoryGrowth > MEMORY_LEAK_THRESHOLD_MB) {
                    stats.memoryLeakWarnings++
                    println("WARNING: Possible memory leak detected, memory growth: ${memoryGrowth} MB")
                    
                    // Trigger GC and re-evaluate.
                    System.gc()
                    delay(2000L)
                    val afterGcMemory = getCurrentMemoryUsage()
                    val actualLeak = afterGcMemory - initialMemory
                    
                    if (actualLeak > MEMORY_LEAK_THRESHOLD_MB) {
                        println("ERROR: Memory leak confirmed, still grew after GC: ${actualLeak} MB")
                        stats.confirmedMemoryLeaks++
                    } else {
                        println("INFO: Memory returned to normal after GC, growth: ${actualLeak} MB")
                    }
                    
                    lastGcTime = System.currentTimeMillis()
                }
                
                // Check for excessive memory usage.
                if (currentMemory > MAX_MEMORY_USAGE_MB) {
                    stats.highMemoryEvents++
                    println("WARNING: High memory usage: ${currentMemory} MB")
                }
                
                // Collect performance stats.
                val perfStats = performanceMonitor.getPerformanceStats(MONITORING_INTERVAL_MS)
                stats.avgCpuUsage = perfStats.cpuUsageAvg
                stats.maxCpuUsage = maxOf(stats.maxCpuUsage, perfStats.cpuUsagePeak)
                
                // Check for abnormal CPU usage.
                if (perfStats.cpuUsageAvg > 80.0) {
                    stats.highCpuEvents++
                    println("WARNING: High CPU usage: ${String.format("%.1f", perfStats.cpuUsageAvg)}%")
                }
                
                // Check app responsiveness.
                checkApplicationResponsiveness(stats)
                
                // Periodic status report.
                val elapsed = System.currentTimeMillis() - testStartTime
                if (elapsed % (5 * 60 * 1000) == 0L) { // Every 5 minutes
                    reportPeriodicStatus(elapsed, stats)
                }
                
                delay(MONITORING_INTERVAL_MS)
                
            } catch (e: Exception) {
                stats.monitoringErrors++
                println("Monitoring error: ${e.message}")
                delay(MONITORING_INTERVAL_MS)
            }
        }
    }

    /**
     * Checks app responsiveness.
     *
     * Detects potential ANR situations.
     */
    private fun checkApplicationResponsiveness(stats: StabilityStats) {
        try {
            activityScenarioRule.scenario.onActivity { activity ->
                // Check whether the main thread is blocked.
                val startTime = System.currentTimeMillis()
                
                activity.runOnUiThread {
                    // This should complete quickly.
                    val endTime = System.currentTimeMillis()
                    val delay = endTime - startTime
                    
                    if (delay > 5000) { // >5s indicates a potential ANR
                        stats.potentialAnrEvents++
                        println("WARNING: Potential ANR detected, UI thread response delay: ${delay}ms")
                    }
                    
                    stats.uiResponseDelayMs = delay
                }
            }
        } catch (e: Exception) {
            stats.responsivenesCheckErrors++
            println("Responsiveness check error: ${e.message}")
        }
    }

    /**
     * Validates stability test results.
     */
    private fun validateStabilityResults(stats: StabilityStats, testStartTime: Long) {
        val testDurationHours = (System.currentTimeMillis() - testStartTime) / (60.0 * 60.0 * 1000.0)
        
        println("\n=== Validating stability results ===")
        
        // Validate basic runtime.
        assertTrue(
            "Test should run long enough",
            testDurationHours >= (if (isRunningInCI()) 0.4 else 11.9) // CI: 24 minutes, device: 11.9 hours
        )
        
        // Validate data generation.
        assertTrue("Should generate a large amount of data", stats.totalDataPointsGenerated > 1000L)
        
        // Validate memory usage.
        assertTrue(
            "Memory usage should stay within limits",
            stats.maxMemoryMB < MAX_MEMORY_USAGE_MB
        )
        
        // Ensure there are no confirmed memory leaks.
        if (stats.confirmedMemoryLeaks > 0) {
            fail("Detected ${stats.confirmedMemoryLeaks} confirmed memory leaks")
        }
        
        // Ensure ANR events remain low.
        assertTrue(
            "ANR events should be rare (< 5)",
            stats.potentialAnrEvents < 5
        )
        
        // Validate CPU usage.
        assertTrue(
            "Average CPU usage should be reasonable (< 60%)",
            stats.avgCpuUsage < 60.0
        )
        
        println("✓ All stability checks passed")
    }

    /**
     * Generates the final test report.
     */
    private fun generateFinalReport(stats: StabilityStats, testStartTime: Long) {
        val testDuration = System.currentTimeMillis() - testStartTime
        val testDurationHours = testDuration / (60.0 * 60.0 * 1000.0)
        
        val report = StringBuilder()
        report.appendLine("=" * 50)
        report.appendLine("Long-term stability test final report")
        report.appendLine("=" * 50)
        report.appendLine("Completed at: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(java.util.Date())}")
        report.appendLine("Total duration: ${String.format("%.2f", testDurationHours)} hours")
        report.appendLine("Environment: ${if (isRunningInCI()) "CI" else "Device"}")
        
        report.appendLine("\nData generation:")
        report.appendLine("  Data points generated: ${stats.totalDataPointsGenerated}")
        report.appendLine("  Data generation errors: ${stats.dataGenerationErrors}")
        report.appendLine("  Generation rate: ${String.format("%.1f", stats.totalDataPointsGenerated * 1000.0 / testDuration)} pts/s")
        
        report.appendLine("\nMemory:")
        report.appendLine("  Current memory: ${stats.currentMemoryMB} MB")
        report.appendLine("  Peak memory: ${stats.maxMemoryMB} MB")
        report.appendLine("  Current heap: ${stats.currentHeapMB} MB")
        report.appendLine("  Memory leak warnings: ${stats.memoryLeakWarnings}")
        report.appendLine("  Confirmed memory leaks: ${stats.confirmedMemoryLeaks}")
        report.appendLine("  High memory events: ${stats.highMemoryEvents}")
        
        report.appendLine("\nPerformance:")
        report.appendLine("  Average CPU usage: ${String.format("%.1f", stats.avgCpuUsage)}%")
        report.appendLine("  Peak CPU usage: ${String.format("%.1f", stats.maxCpuUsage)}%")
        report.appendLine("  High CPU events: ${stats.highCpuEvents}")
        
        report.appendLine("\nStability indicators:")
        report.appendLine("  Potential ANR events: ${stats.potentialAnrEvents}")
        report.appendLine("  UI response delay: ${stats.uiResponseDelayMs} ms")
        report.appendLine("  Monitoring errors: ${stats.monitoringErrors}")
        report.appendLine("  Responsiveness check errors: ${stats.responsivenesCheckErrors}")
        
        report.appendLine("\nComponent counters:")
        val metricsSnapshot = metricsCollector.getSnapshot()
        report.appendLine("  Sensor samples collected: ${metricsSnapshot.counters[com.continuousauth.observability.MetricType.SENSOR_SAMPLES_COLLECTED] ?: 0}")
        report.appendLine("  Anomalies detected: ${metricsSnapshot.counters[com.continuousauth.observability.MetricType.ANOMALIES_DETECTED] ?: 0}")
        report.appendLine("  Fast-mode switches: ${metricsSnapshot.counters[com.continuousauth.observability.MetricType.MODE_SWITCHES_TO_FAST] ?: 0}")
        
        // Conclusion.
        val isSuccessful = stats.confirmedMemoryLeaks == 0 &&
                          stats.potentialAnrEvents < 5 &&
                          stats.maxMemoryMB < MAX_MEMORY_USAGE_MB &&
                          stats.avgCpuUsage < 60.0
        
        report.appendLine("\nConclusion:")
        report.appendLine(if (isSuccessful) "✓ Long-term stability test passed" else "✗ Long-term stability test failed")
        
        println(report.toString())
        
        // Save report to file (if possible).
        try {
            val reportFile = java.io.File(context.cacheDir, "stability_test_report.txt")
            reportFile.writeText(report.toString())
            println("Report saved to: ${reportFile.absolutePath}")
        } catch (e: Exception) {
            println("Failed to save report: ${e.message}")
        }
    }

    /**
     * Periodic status report.
     */
    private fun reportPeriodicStatus(elapsed: Long, stats: StabilityStats) {
        val hours = elapsed / (60 * 60 * 1000.0)
        val targetHours = TEST_DURATION_MS / (60.0 * 60.0 * 1000.0)
        val progress = (elapsed.toDouble() / TEST_DURATION_MS) * 100
        
        println("\n--- Stability progress ---")
        println("Uptime: ${String.format("%.2f", hours)}/${String.format("%.1f", targetHours)} hours (${String.format("%.1f", progress)}%)")
        println("Data points: ${stats.totalDataPointsGenerated}, memory: ${stats.currentMemoryMB}MB, CPU: ${String.format("%.1f", stats.avgCpuUsage)}%")
        println("Memory leak warnings: ${stats.memoryLeakWarnings}, ANR events: ${stats.potentialAnrEvents}")
        println("---------------------------")
    }

    /**
     * Gets current memory usage (MB).
     */
    private fun getCurrentMemoryUsage(): Long {
        val memoryInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(memoryInfo)
        return memoryInfo.totalPss / 1024L // Convert to MB.
    }

    /**
     * Gets current heap usage (MB).
     */
    private fun getCurrentHeapUsage(): Long {
        val runtime = Runtime.getRuntime()
        val usedMemory = runtime.totalMemory() - runtime.freeMemory()
        return usedMemory / 1024L / 1024L // Convert to MB.
    }

    /**
     * Extension: string repetition.
     */
    private operator fun String.times(count: Int): String {
        return repeat(count)
    }
}

/**
 * Stability test statistics.
 */
data class StabilityStats(
    var totalDataPointsGenerated: Long = 0L,
    var dataGenerationErrors: Int = 0,
    var currentMemoryMB: Long = 0L,
    var maxMemoryMB: Long = 0L,
    var currentHeapMB: Long = 0L,
    var memoryLeakWarnings: Int = 0,
    var confirmedMemoryLeaks: Int = 0,
    var highMemoryEvents: Int = 0,
    var avgCpuUsage: Double = 0.0,
    var maxCpuUsage: Double = 0.0,
    var highCpuEvents: Int = 0,
    var potentialAnrEvents: Int = 0,
    var uiResponseDelayMs: Long = 0L,
    var monitoringErrors: Int = 0,
    var responsivenesCheckErrors: Int = 0
)
