package com.continuousauth.ui

import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.*
import androidx.test.espresso.assertion.ViewAssertions.*
import androidx.test.espresso.matcher.ViewMatchers.*
import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import com.continuousauth.R
import com.continuousauth.detection.AnomalyTrigger
import com.continuousauth.network.ConnectionStatus
import com.continuousauth.network.NetworkState
import dagger.hilt.android.testing.HiltAndroidRule
import dagger.hilt.android.testing.HiltAndroidTest
import kotlinx.coroutines.test.runTest
import org.hamcrest.Matchers.*
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import javax.inject.Inject

/**
 * Main screen UI tests.
 *
 * Covers common user interactions and state changes in MainActivity.
 */
@LargeTest
@HiltAndroidTest
@RunWith(AndroidJUnit4::class)
class MainActivityInstrumentedTest {

    @get:Rule(order = 0)
    var hiltRule = HiltAndroidRule(this)

    @get:Rule(order = 1)
    var activityScenarioRule = ActivityScenarioRule(MainActivity::class.java)

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Before
    fun init() {
        hiltRule.inject()
    }

    /**
     * Verifies that basic UI elements are displayed.
     */
    @Test
    fun testBasicUIElements() {
        // Main UI elements.
        onView(withId(R.id.btnToggleCollection)).check(matches(isDisplayed()))
        onView(withId(R.id.btnToggleChart)).check(matches(isDisplayed()))
        onView(withId(R.id.tvCollectionStatus)).check(matches(isDisplayed()))
        onView(withId(R.id.tvConnectionStatus)).check(matches(isDisplayed()))
        onView(withId(R.id.tvNetworkQuality)).check(matches(isDisplayed()))
        
        // Sensor status.
        onView(withId(R.id.tvAccelerometerStatus)).check(matches(isDisplayed()))
        onView(withId(R.id.tvGyroscopeStatus)).check(matches(isDisplayed()))
        onView(withId(R.id.tvMagnetometerStatus)).check(matches(isDisplayed()))
        
        // Transmission stats.
        onView(withId(R.id.tvTransmissionMode)).check(matches(isDisplayed()))
        onView(withId(R.id.tvPacketsSent)).check(matches(isDisplayed()))
        onView(withId(R.id.tvPacketsPending)).check(matches(isDisplayed()))
    }

    /**
     * Tests starting/stopping data collection.
     */
    @Test
    fun testDataCollectionToggle() {
        // Initial state: collection should be stopped.
        onView(withId(R.id.tvCollectionStatus))
            .check(matches(withText(containsString(context.getString(R.string.status_stopped)))))
        
        onView(withId(R.id.btnToggleCollection))
            .check(matches(withText(context.getString(R.string.start_collection))))

        // Start collection.
        onView(withId(R.id.btnToggleCollection)).perform(click())
        
        // Wait for UI update.
        Thread.sleep(1000)
        
        // Verify state change.
        onView(withId(R.id.btnToggleCollection))
            .check(matches(withText(context.getString(R.string.stop_collection))))
        
        // Stop collection.
        onView(withId(R.id.btnToggleCollection)).perform(click())
        
        // Wait for UI update.
        Thread.sleep(1000)
        
        // Verify state reset.
        onView(withId(R.id.btnToggleCollection))
            .check(matches(withText(context.getString(R.string.start_collection))))
    }

    /**
     * Tests chart show/hide behavior.
     */
    @Test
    fun testChartToggle() {
        // Initial state: chart should be hidden.
        onView(withId(R.id.chartContainer))
            .check(matches(not(isDisplayed())))
        
        onView(withId(R.id.btnToggleChart))
            .check(matches(withText(context.getString(R.string.show_chart))))

        // Show chart.
        onView(withId(R.id.btnToggleChart)).perform(click())
        
        // Verify chart is visible.
        onView(withId(R.id.chartContainer))
            .check(matches(isDisplayed()))
        
        onView(withId(R.id.btnToggleChart))
            .check(matches(withText(context.getString(R.string.hide_chart))))

        // Hide chart.
        onView(withId(R.id.btnToggleChart)).perform(click())
        
        // Verify chart is hidden.
        onView(withId(R.id.chartContainer))
            .check(matches(not(isDisplayed())))
    }

    /**
     * Tests menu options.
     */
    @Test
    fun testMenuOptions() {
        // Open options menu.
        onView(isRoot()).perform(pressMenuKey())
        
        // Toggle debug mode.
        onView(withId(R.id.action_debug)).perform(click())
        
        // Verify debug card is shown.
        onView(withId(R.id.cardDebugInfo))
            .check(matches(isDisplayed()))
        
        // Toggle off debug mode.
        onView(isRoot()).perform(pressMenuKey())
        onView(withId(R.id.action_debug)).perform(click())
        
        // Verify debug card is hidden.
        onView(withId(R.id.cardDebugInfo))
            .check(matches(not(isDisplayed())))
    }

    /**
     * Tests visualization mode toggle.
     */
    @Test
    fun testVisualizationToggle() {
        // Open options menu.
        onView(isRoot()).perform(pressMenuKey())
        
        // Toggle visualization mode.
        onView(withId(R.id.action_visualization)).perform(click())
        
        // Verify visualization card is shown.
        onView(withId(R.id.cardVisualization))
            .check(matches(isDisplayed()))
        
        // Toggle again.
        onView(isRoot()).perform(pressMenuKey())
        onView(withId(R.id.action_visualization)).perform(click())
        
        // Verify visualization card is hidden.
        onView(withId(R.id.cardVisualization))
            .check(matches(not(isDisplayed())))
    }

    /**
     * Tests network state display.
     */
    @Test
    fun testNetworkStatusDisplay() = runTest {
        activityScenarioRule.scenario.onActivity { activity ->
            val viewModel = activity.viewModel
            
            // Simulate different network states.
            val networkStates = listOf(
                NetworkState.WIFI_EXCELLENT,
                NetworkState.WIFI_GOOD,
                NetworkState.WIFI_POOR,
                NetworkState.CELLULAR_EXCELLENT,
                NetworkState.CELLULAR_GOOD,
                NetworkState.CELLULAR_POOR,
                NetworkState.DISCONNECTED
            )
            
            networkStates.forEach { state ->
                // Set network state via reflection (test-only simulation).
                val networkStateField = viewModel.javaClass.getDeclaredField("_networkState")
                networkStateField.isAccessible = true
                val mutableLiveData = networkStateField.get(viewModel) as androidx.lifecycle.MutableLiveData<NetworkState>
                
                activity.runOnUiThread {
                    mutableLiveData.value = state
                }
                
                Thread.sleep(500) // Wait for UI update.
                
                // Verify displayed text.
                val expectedText = when (state) {
                    NetworkState.WIFI_EXCELLENT -> context.getString(R.string.wifi_excellent)
                    NetworkState.WIFI_GOOD -> context.getString(R.string.wifi_good)
                    NetworkState.WIFI_POOR -> context.getString(R.string.wifi_poor)
                    NetworkState.CELLULAR_EXCELLENT -> context.getString(R.string.cellular_excellent)
                    NetworkState.CELLULAR_GOOD -> context.getString(R.string.cellular_good)
                    NetworkState.CELLULAR_POOR -> context.getString(R.string.cellular_poor)
                    NetworkState.DISCONNECTED -> context.getString(R.string.disconnected)
                    else -> "-"
                }
                
                onView(withId(R.id.tvNetworkQuality))
                    .check(matches(withText(expectedText)))
            }
        }
    }

    /**
     * Tests connection status display.
     */
    @Test
    fun testConnectionStatusDisplay() = runTest {
        activityScenarioRule.scenario.onActivity { activity ->
            val viewModel = activity.viewModel
            
            val connectionStates = listOf(
                ConnectionStatus.CONNECTED,
                ConnectionStatus.DISCONNECTED,
                ConnectionStatus.CONNECTING,
                ConnectionStatus.RECONNECTING
            )
            
            connectionStates.forEach { status ->
                // Set connection status via reflection (test-only simulation).
                val connectionStatusField = viewModel.javaClass.getDeclaredField("_connectionStatus")
                connectionStatusField.isAccessible = true
                val mutableLiveData = connectionStatusField.get(viewModel) as androidx.lifecycle.MutableLiveData<ConnectionStatus>
                
                activity.runOnUiThread {
                    mutableLiveData.value = status
                }
                
                Thread.sleep(500) // Wait for UI update.
                
                // Verify displayed text.
                val expectedText = when (status) {
                    ConnectionStatus.CONNECTED -> context.getString(R.string.connected)
                    ConnectionStatus.DISCONNECTED -> context.getString(R.string.disconnected)
                    ConnectionStatus.CONNECTING -> context.getString(R.string.connecting)
                    ConnectionStatus.RECONNECTING -> context.getString(R.string.reconnecting)
                    else -> status.name
                }
                
                onView(withId(R.id.tvConnectionStatus))
                    .check(matches(withText(expectedText)))
            }
        }
    }

    /**
     * Tests sensor status display.
     */
    @Test
    fun testSensorStatusDisplay() = runTest {
        activityScenarioRule.scenario.onActivity { activity ->
            val viewModel = activity.viewModel
            
            // Simulate active sensors.
            val activeSensorStatus = mapOf(
                "accelerometer" to true,
                "gyroscope" to true,
                "magnetometer" to true
            )
            
            val inactiveSensorStatus = mapOf(
                "accelerometer" to false,
                "gyroscope" to false,
                "magnetometer" to false
            )
            
            listOf(activeSensorStatus, inactiveSensorStatus).forEach { sensorStatus ->
                // Set sensor state via reflection (test-only simulation).
                val sensorStatusField = viewModel.javaClass.getDeclaredField("_sensorStatus")
                sensorStatusField.isAccessible = true
                val mutableLiveData = sensorStatusField.get(viewModel) as androidx.lifecycle.MutableLiveData<Map<String, Boolean>>
                
                activity.runOnUiThread {
                    mutableLiveData.value = sensorStatus
                }
                
                Thread.sleep(500) // Wait for UI update.
                
                val expectedText = if (sensorStatus.values.first()) 
                    context.getString(R.string.sensor_active) 
                else 
                    context.getString(R.string.sensor_inactive)
                
                // Verify sensor status labels.
                onView(withId(R.id.tvAccelerometerStatus))
                    .check(matches(withText(expectedText)))
                onView(withId(R.id.tvGyroscopeStatus))
                    .check(matches(withText(expectedText)))
                onView(withId(R.id.tvMagnetometerStatus))
                    .check(matches(withText(expectedText)))
            }
        }
    }

    /**
     * Tests error message handling.
     */
    @Test
    fun testErrorMessageHandling() = runTest {
        activityScenarioRule.scenario.onActivity { activity ->
            val viewModel = activity.viewModel
            
            // Trigger an error message via reflection (test-only).
            val errorMessageField = viewModel.javaClass.getDeclaredField("_errorMessage")
            errorMessageField.isAccessible = true
            val mutableLiveData = errorMessageField.get(viewModel) as androidx.lifecycle.MutableLiveData<String?>
            
            val testErrorMessage = "Test error message"
            
            activity.runOnUiThread {
                mutableLiveData.value = testErrorMessage
            }
            
            Thread.sleep(500) // Wait for handling.
            
            // Note: showErrorMessage currently only logs; this mainly verifies that it does not crash.
            // A production-ready test should assert Snackbar/Dialog visibility.
        }
    }

    /**
     * Tests behavior under different permission states.
     */
    @Test
    fun testPermissionHandling() {
        // This test needs more advanced permission simulation; keep a minimal sanity check.
        // A full test should use PermissionTestRule or UiAutomator to grant/deny permissions.
        
        activityScenarioRule.scenario.onActivity { activity ->
            // Activity should start without permission-related crashes.
            assertThat(activity, not(nullValue()))
            
            // UI elements should exist.
            onView(withId(R.id.btnToggleCollection)).check(matches(isDisplayed()))
        }
    }

    /**
     * Tests Activity lifecycle handling.
     */
    @Test
    fun testActivityLifecycle() {
        // Simulate pause/resume.
        activityScenarioRule.scenario.onActivity { activity ->
            // Activity should be running.
            assertThat(activity, not(nullValue()))
        }
        
        // Move to background-like state.
        activityScenarioRule.scenario.moveToState(androidx.lifecycle.Lifecycle.State.STARTED)
        
        // Return to resumed state.
        activityScenarioRule.scenario.moveToState(androidx.lifecycle.Lifecycle.State.RESUMED)
        
        // UI should still work.
        onView(withId(R.id.btnToggleCollection)).check(matches(isDisplayed()))
    }
}
