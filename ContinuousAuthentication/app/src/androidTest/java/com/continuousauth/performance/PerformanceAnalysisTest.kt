package com.continuousauth.performance

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import com.continuousauth.observability.PerformanceMonitorImpl
import com.continuousauth.core.SmartTransmissionManager
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
 * Performance monitoring and analysis tests.
 *
 * Evaluates CPU and memory usage under different workloads.
 *
 * Task 5.2.2: Use Android Profiler and Battery Historian to analyze power consumption.
 */
@LargeTest
@HiltAndroidTest
@RunWith(AndroidJUnit4::class)
class PerformanceAnalysisTest {

    @get:Rule
    var hiltRule = HiltAndroidRule(this)

    @Inject
    lateinit var performanceMonitor: PerformanceMonitorImpl

    @Inject
    lateinit var smartTransmissionManager: SmartTransmissionManager

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Before
    fun init() {
        hiltRule.inject()
    }

    /**
     * Tests baseline performance monitoring.
     */
    @Test
    fun testBaselinePerformanceMonitoring() = runTest {
        // Start monitoring.
        performanceMonitor.startMonitoring(1000L) // Sample every second

        // Collect baseline data.
        delay(5000L)

        // Fetch stats.
        val stats = performanceMonitor.getPerformanceStats(5000L)
        
        // Basic validation.
        assertTrue("Should have samples", stats.sampleCount > 0)
        assertTrue("Memory usage should be > 0", stats.memoryUsageAvg > 0.0)
        assertTrue("CPU usage should be >= 0", stats.cpuUsageAvg >= 0.0)
        assertTrue("Heap usage should be > 0", stats.heapUsageAvg > 0.0)
        assertTrue("Heap utilization should be within range", stats.heapUtilization >= 0.0 && stats.heapUtilization <= 100.0)

        // Peak values should stay within limits.
        assertTrue("Peak memory usage should stay within limits", stats.memoryUsagePeak < 500L) // 500MB threshold
        assertTrue("Peak CPU usage should be <= 80%", stats.cpuUsagePeak <= 80.0)

        performanceMonitor.stopMonitoring()
    }

    /**
     * Tests performance under high load.
     */
    @Test
    fun testHighLoadPerformance() = runTest {
        performanceMonitor.startMonitoring(500L) // More frequent sampling

        // Start transmission manager to generate load.
        smartTransmissionManager.start()

        // Process a large amount of sensor data.
        val startTime = System.currentTimeMillis()
        repeat(1000) { index ->
            // Generate changing sensor data.
            val x = (index % 10) * 0.1f + Math.random().toFloat() * 0.5f
            val y = (index % 8) * 0.12f + Math.random().toFloat() * 0.3f
            val z = 9.8f + Math.random().toFloat() * 0.2f
            
            smartTransmissionManager.processSensorData(x, y, z, System.nanoTime() + index * 10000000L)
            
            // Pause periodically to avoid excessive load.
            if (index % 100 == 0) {
                delay(10L)
            }
        }

        val processingTime = System.currentTimeMillis() - startTime
        
        // Wait for monitoring data to accumulate.
        delay(2000L)

        val stats = performanceMonitor.getPerformanceStats(10000L)
        
        // Assertions.
        assertTrue("Should have enough samples under high load", stats.sampleCount > 5)
        
        // Memory usage should stay within limits even under high load.
        assertTrue("Peak memory should be < 1GB under high load", stats.memoryUsagePeak < 1000L)
        
        // CPU usage can be high but should not stay pegged.
        assertTrue("Average CPU usage should be < 95%", stats.cpuUsageAvg < 95.0)
        
        // Processing 1000 samples should complete in reasonable time.
        assertTrue("Processing time should be within bounds", processingTime < 30000L) // Within 30s

        println("High load performance results:")
        println("Processing time: ${processingTime}ms")
        println("Average memory usage: ${String.format("%.1f", stats.memoryUsageAvg)}MB")
        println("Peak memory usage: ${stats.memoryUsagePeak}MB")
        println("Average CPU usage: ${String.format("%.1f", stats.cpuUsageAvg)}%")
        println("Peak CPU usage: ${String.format("%.1f", stats.cpuUsagePeak)}%")

        smartTransmissionManager.stop()
        performanceMonitor.stopMonitoring()
    }

    /**
     * Tests behavior under memory pressure.
     */
    @Test
    fun testMemoryPressureHandling() = runTest {
        performanceMonitor.startMonitoring(1000L)
        
        val initialStats = performanceMonitor.getPerformanceStats(1000L)
        val initialMemory = performanceMonitor.getCurrentSnapshot().memoryUsageMB

        // Create some memory pressure (careful to avoid OOM).
        val memoryConsumers = mutableListOf<ByteArray>()
        
        try {
            // Gradually allocate memory blocks and observe changes.
            repeat(20) { // Limit iterations to avoid OOM.
                val memoryBlock = ByteArray(1024 * 1024 * 5) // 5MB block
                memoryConsumers.add(memoryBlock)
                
                delay(100L) // Brief pause
                
                val currentSnapshot = performanceMonitor.getCurrentSnapshot()
                
                // Memory usage should increase.
                assertTrue(
                    "Memory usage should increase with allocations",
                    currentSnapshot.memoryUsageMB >= initialMemory
                )
                
                // Exit early if memory usage is too high to avoid OOM.
                if (currentSnapshot.memoryUsageMB > 200L) {
                    break
                }
            }
            
            delay(2000L) // Allow sampling
            
            val pressureStats = performanceMonitor.getPerformanceStats(5000L)
            
            // Assertions.
            assertTrue("Peak memory should increase under memory pressure", 
                pressureStats.memoryUsagePeak > initialStats.memoryUsagePeak)
            
            println("Memory pressure test results:")
            println("Initial memory: ${initialMemory}MB")
            println("Peak memory under pressure: ${pressureStats.memoryUsagePeak}MB")
            println("Heap utilization: ${String.format("%.1f", pressureStats.heapUtilization)}%")
            
        } finally {
            // Clean up memory and help GC.
            memoryConsumers.clear()
            System.gc()
            delay(1000L) // Wait for GC
        }

        performanceMonitor.stopMonitoring()
    }

    /**
     * Tests stability during longer runs.
     */
    @Test
    fun testLongRunningStability() = runTest {
        performanceMonitor.startMonitoring(2000L) // Longer sampling interval

        smartTransmissionManager.start()

        val startTime = System.currentTimeMillis()
        val targetDuration = 30000L // 30s test (use 12h in production)

        var iterationCount = 0
        
        while (System.currentTimeMillis() - startTime < targetDuration) {
            // Simulate normal sensor data stream.
            val x = Math.sin(iterationCount * 0.1) * 0.5f
            val y = Math.cos(iterationCount * 0.1) * 0.3f
            val z = 9.8f + Math.sin(iterationCount * 0.05) * 0.1f
            
            smartTransmissionManager.processSensorData(x.toFloat(), y.toFloat(), z, System.nanoTime())
            
            iterationCount++
            delay(50L) // Simulate 20 Hz sampling rate
        }

        val finalStats = performanceMonitor.getPerformanceStats(targetDuration)
        
        // Assertions.
        assertTrue("Should have enough samples in a long run", finalStats.sampleCount > 10)
        
        // Memory should be relatively stable (no obvious leaks).
        val memoryGrowth = finalStats.memoryUsagePeak - finalStats.memoryUsageAvg
        assertTrue("Memory growth should be within limits", memoryGrowth < 100L) // 100MB threshold
        
        // CPU usage should stay reasonable.
        assertTrue("Average CPU usage should be reasonable over time", finalStats.cpuUsageAvg < 50.0)
        
        println("Long-running stability test results:")
        println("Duration: ${targetDuration / 1000}s")
        println("Processed samples: $iterationCount")
        println("Average memory: ${String.format("%.1f", finalStats.memoryUsageAvg)}MB")
        println("Peak memory: ${finalStats.memoryUsagePeak}MB")
        println("Memory delta: ${String.format("%.1f", memoryGrowth)}MB")
        println("Average CPU: ${String.format("%.1f", finalStats.cpuUsageAvg)}%")

        smartTransmissionManager.stop()
        performanceMonitor.stopMonitoring()
    }

    /**
     * Tests the overhead of the performance monitor itself.
     */
    @Test
    fun testPerformanceMonitorOverhead() = runTest {
        // Baseline without monitoring.
        val baselineStart = System.currentTimeMillis()
        
        repeat(1000) {
            // Basic work.
            val x = Math.random().toFloat()
            val y = Math.random().toFloat()
            val z = 9.8f + Math.random().toFloat() * 0.1f
            
            // Pure computation, no monitoring.
            val magnitude = Math.sqrt((x * x + y * y + z * z).toDouble())
        }
        
        val baselineTime = System.currentTimeMillis() - baselineStart

        // With monitoring enabled.
        performanceMonitor.startMonitoring(100L) // High-frequency monitoring for overhead test
        
        val monitoredStart = System.currentTimeMillis()
        
        repeat(1000) {
            val x = Math.random().toFloat()
            val y = Math.random().toFloat()
            val z = 9.8f + Math.random().toFloat() * 0.1f
            
            // Same computation under monitoring.
            val magnitude = Math.sqrt((x * x + y * y + z * z).toDouble())
            
            // Snapshot reads (simulates monitoring overhead).
            if (it % 100 == 0) {
                performanceMonitor.getCurrentSnapshot()
            }
        }
        
        val monitoredTime = System.currentTimeMillis() - monitoredStart
        
        performanceMonitor.stopMonitoring()

        // Monitoring overhead.
        val overhead = monitoredTime - baselineTime
        val overheadPercent = (overhead.toDouble() / baselineTime) * 100.0

        println("Performance monitor overhead:")
        println("Baseline time: ${baselineTime}ms")
        println("Monitored time: ${monitoredTime}ms")
        println("Overhead: ${overhead}ms (${String.format("%.1f", overheadPercent)}%)")

        // Overhead should be acceptable.
        assertTrue("Performance monitor overhead should be < 50%", overheadPercent < 50.0)
    }

    /**
     * Tests ring buffer behavior.
     */
    @Test
    fun testRingBufferPerformance() = runTest {
        performanceMonitor.startMonitoring(100L) // High-frequency sampling to exercise the buffer

        // Run long enough to fill the buffer.
        delay(15000L) // 15s, should produce ~150 samples

        val samples = performanceMonitor.getRecentSamples(100)
        
        // Assertions.
        assertTrue("Should retrieve samples", samples.isNotEmpty())
        assertTrue("Sample count should not exceed the requested size", samples.size <= 100)
        
        // Timestamps should be non-decreasing.
        if (samples.size > 1) {
            for (i in 1 until samples.size) {
                assertTrue("Sample timestamps should be non-decreasing", 
                    samples[i].timestamp >= samples[i-1].timestamp)
            }
        }

        performanceMonitor.stopMonitoring()
    }

    /**
     * Tests performance under different load patterns.
     */
    @Test
    fun testVariousLoadPatterns() = runTest {
        performanceMonitor.startMonitoring(500L)
        smartTransmissionManager.start()

        // Pattern 1: burst load.
        println("Testing burst load pattern...")
        repeat(5) {
            // Process a burst of data quickly.
            repeat(200) { index ->
                smartTransmissionManager.processSensorData(
                    Math.random().toFloat(),
                    Math.random().toFloat(),
                    9.8f + Math.random().toFloat() * 0.1f,
                    System.nanoTime()
                )
            }
            delay(1000L) // Rest
        }

        val burstStats = performanceMonitor.getPerformanceStats(10000L)
        println("Burst load - peak CPU: ${String.format("%.1f", burstStats.cpuUsagePeak)}%, peak memory: ${burstStats.memoryUsagePeak}MB")

        // Pattern 2: steady load.
        println("Testing steady load pattern...")
        repeat(500) { index ->
            smartTransmissionManager.processSensorData(
                Math.sin(index * 0.1).toFloat(),
                Math.cos(index * 0.1).toFloat(),
                9.8f,
                System.nanoTime()
            )
            delay(20L) // Steady 50 Hz
        }

        val steadyStats = performanceMonitor.getPerformanceStats(15000L)
        println("Steady load - avg CPU: ${String.format("%.1f", steadyStats.cpuUsageAvg)}%, avg memory: ${String.format("%.1f", steadyStats.memoryUsageAvg)}MB")

        // Burst peak CPU should exceed steady average CPU.
        assertTrue("Burst peak CPU should be higher than steady avg CPU", 
            burstStats.cpuUsagePeak > steadyStats.cpuUsageAvg)

        smartTransmissionManager.stop()
        performanceMonitor.stopMonitoring()
    }
}
