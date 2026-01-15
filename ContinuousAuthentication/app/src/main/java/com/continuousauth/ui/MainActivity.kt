package com.continuousauth.ui

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.continuousauth.R
import com.continuousauth.databinding.ActivityMainBinding
import com.continuousauth.network.ConnectionStatus
import com.continuousauth.network.NetworkState
import com.continuousauth.ui.dialogs.PrivacyAgreementDialog
import com.continuousauth.ui.dialogs.BatteryOptimizationDialog
import com.continuousauth.ui.dialogs.UsageStatsPermissionDialog
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.launch

/**
 * Main activity.
 *
 * Displays the main UI and coordinates major modules.
 */
@AndroidEntryPoint
class MainActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "MainActivity"
        private const val PRIVACY_AGREEMENT_SHOWN = "privacy_agreement_shown"
        private const val BATTERY_OPTIMIZATION_REQUESTED = "battery_optimization_requested"
    }
    
    private lateinit var binding: ActivityMainBinding
    private lateinit var viewModel: MainViewModel
    
    // Permission launchers
    private val usageStatsPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        // Check whether USAGE_STATS permission has been granted.
        viewModel.checkUsageStatsPermission()
    }
    
    private val batteryOptimizationLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        // Record that the user has handled the battery optimization prompt.
        getSharedPreferences("app_prefs", MODE_PRIVATE)
            .edit()
            .putBoolean(BATTERY_OPTIMIZATION_REQUESTED, true)
            .apply()
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize ViewBinding.
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // Set ActionBar title.
        supportActionBar?.title = getString(R.string.app_name)
        
        // Initialize ViewModel.
        viewModel = ViewModelProvider(this)[MainViewModel::class.java]
        
        // Wire up UI listeners.
        setupUIListeners()
        
        // Observe ViewModel state.
        observeViewModelData()
        
        // Check and run first-launch flow.
        checkFirstLaunchFlow()
        
        Log.i(TAG, "MainActivity initialized")
    }
    
    override fun onResume() {
        super.onResume()
        // Refresh status information.
        viewModel.refreshStatus()
    }
    
    /**
     * Sets up UI event listeners.
     */
    private fun setupUIListeners() {
        // Start/stop collection button
        binding.btnToggleCollection.setOnClickListener {
            if (viewModel.isCollectionRunning.value == true) {
                viewModel.stopCollection()
            } else {
                viewModel.startCollection()
            }
        }
        
        // Chart show/hide button
        binding.btnToggleChart.setOnClickListener {
            val isVisible = binding.chartContainer.visibility == View.VISIBLE
            if (isVisible) {
                binding.chartContainer.visibility = View.GONE
                binding.btnToggleChart.text = getString(R.string.show_chart)
            } else {
                binding.chartContainer.visibility = View.VISIBLE
                binding.btnToggleChart.text = getString(R.string.hide_chart)
                viewModel.startVisualization()
            }
        }
    }
    
    /**
     * Observes ViewModel updates.
     */
    private fun observeViewModelData() {
        // Collection status
        viewModel.collectionStatus.observe(this) { status ->
            updateCollectionStatus(status)
        }
        
        // Collection running state
        viewModel.isCollectionRunning.observe(this) { isRunning ->
            updateCollectionButton(isRunning)
        }
        
        // Sensor status
        viewModel.sensorStatus.observe(this) { sensorStatus ->
            updateSensorStatus(sensorStatus)
        }
        
        // Network status
        viewModel.networkState.observe(this) { networkState ->
            updateNetworkStatus(networkState)
        }
        
        // Connection status
        viewModel.connectionStatus.observe(this) { connectionStatus ->
            updateConnectionStatus(connectionStatus)
        }
        
        // Transmission stats
        viewModel.transmissionStats.observe(this) { stats ->
            updateTransmissionStats(stats)
        }
        
        // Debug visibility
        viewModel.debugModeEnabled.observe(this) { debugEnabled ->
            binding.cardDebugInfo.visibility = if (debugEnabled) View.VISIBLE else View.GONE
        }
        
        // Visualization visibility
        viewModel.visualizationEnabled.observe(this) { visualizationEnabled ->
            binding.cardVisualization.visibility = if (visualizationEnabled) View.VISIBLE else View.GONE
        }
        
        // Error messages
        viewModel.errorMessage.observe(this) { errorMessage ->
            if (!errorMessage.isNullOrEmpty()) {
                // Show an error dialog or Snackbar.
                showErrorMessage(errorMessage)
            }
        }
        
        // Debug info
        viewModel.debugInfo.observe(this) { debugInfo ->
            binding.tvDebugInfo.text = debugInfo
        }
    }
    
    /**
     * Updates collection status UI.
     */
    private fun updateCollectionStatus(status: String) {
        binding.tvCollectionStatus.text = when (status) {
            "RUNNING" -> getString(R.string.status_running)
            "STOPPED" -> getString(R.string.status_stopped)
            "PAUSED" -> getString(R.string.status_paused)
            "ERROR" -> getString(R.string.status_error)
            else -> status
        }
        
        // Update text color based on status.
        val colorRes = when (status) {
            "RUNNING" -> R.color.status_running
            "STOPPED" -> R.color.status_stopped
            "PAUSED" -> R.color.status_paused
            "ERROR" -> R.color.status_error
            else -> R.color.secondary_text
        }
        binding.tvCollectionStatus.setTextColor(ContextCompat.getColor(this, colorRes))
    }
    
    /**
     * Updates collection button UI.
     */
    private fun updateCollectionButton(isRunning: Boolean) {
        if (isRunning) {
            binding.btnToggleCollection.text = getString(R.string.stop_collection)
            binding.btnToggleCollection.icon = ContextCompat.getDrawable(this, R.drawable.ic_stop)
        } else {
            binding.btnToggleCollection.text = getString(R.string.start_collection)
            binding.btnToggleCollection.icon = ContextCompat.getDrawable(this, R.drawable.ic_play_arrow)
        }
    }
    
    /**
     * Updates sensor status UI.
     */
    private fun updateSensorStatus(sensorStatus: Map<String, Boolean>) {
        // Accelerometer
        val accelerometerActive = sensorStatus["accelerometer"] == true
        binding.tvAccelerometerStatus.text = if (accelerometerActive) 
            getString(R.string.sensor_active) else getString(R.string.sensor_inactive)
        binding.tvAccelerometerStatus.setTextColor(
            ContextCompat.getColor(this, 
                if (accelerometerActive) R.color.sensor_active else R.color.sensor_inactive
            )
        )
        
        // Gyroscope
        val gyroscopeActive = sensorStatus["gyroscope"] == true
        binding.tvGyroscopeStatus.text = if (gyroscopeActive) 
            getString(R.string.sensor_active) else getString(R.string.sensor_inactive)
        binding.tvGyroscopeStatus.setTextColor(
            ContextCompat.getColor(this, 
                if (gyroscopeActive) R.color.sensor_active else R.color.sensor_inactive
            )
        )
        
        // Magnetometer
        val magnetometerActive = sensorStatus["magnetometer"] == true
        binding.tvMagnetometerStatus.text = if (magnetometerActive) 
            getString(R.string.sensor_active) else getString(R.string.sensor_inactive)
        binding.tvMagnetometerStatus.setTextColor(
            ContextCompat.getColor(this, 
                if (magnetometerActive) R.color.sensor_active else R.color.sensor_inactive
            )
        )
    }
    
    /**
     * Updates network status UI.
     */
    private fun updateNetworkStatus(networkState: NetworkState) {
        val qualityText = when (networkState) {
            NetworkState.WIFI_EXCELLENT -> getString(R.string.wifi_excellent)
            NetworkState.WIFI_GOOD -> getString(R.string.wifi_good)
            NetworkState.WIFI_POOR -> getString(R.string.wifi_poor)
            NetworkState.CELLULAR_EXCELLENT -> getString(R.string.cellular_excellent)
            NetworkState.CELLULAR_GOOD -> getString(R.string.cellular_good)
            NetworkState.CELLULAR_POOR -> getString(R.string.cellular_poor)
            NetworkState.DISCONNECTED -> getString(R.string.disconnected)
            else -> "-"
        }
        
        binding.tvNetworkQuality.text = qualityText
        
        // Set color based on network quality.
        val colorRes = when (networkState) {
            NetworkState.WIFI_EXCELLENT, NetworkState.CELLULAR_EXCELLENT -> R.color.network_excellent
            NetworkState.WIFI_GOOD, NetworkState.CELLULAR_GOOD -> R.color.network_good
            NetworkState.WIFI_POOR, NetworkState.CELLULAR_POOR -> R.color.network_poor
            NetworkState.DISCONNECTED -> R.color.network_disconnected
            else -> R.color.secondary_text
        }
        binding.tvNetworkQuality.setTextColor(ContextCompat.getColor(this, colorRes))
    }
    
    /**
     * Updates connection status UI.
     */
    private fun updateConnectionStatus(connectionStatus: ConnectionStatus) {
        val statusText = when (connectionStatus) {
            ConnectionStatus.CONNECTED -> getString(R.string.connected)
            ConnectionStatus.DISCONNECTED -> getString(R.string.disconnected)
            ConnectionStatus.CONNECTING -> getString(R.string.connecting)
            ConnectionStatus.RECONNECTING -> getString(R.string.reconnecting)
            else -> connectionStatus.name
        }
        
        binding.tvConnectionStatus.text = statusText
        
        // Set color based on connection state.
        val colorRes = when (connectionStatus) {
            ConnectionStatus.CONNECTED -> R.color.status_running
            ConnectionStatus.DISCONNECTED -> R.color.status_error
            ConnectionStatus.CONNECTING, ConnectionStatus.RECONNECTING -> R.color.status_paused
            else -> R.color.secondary_text
        }
        binding.tvConnectionStatus.setTextColor(ContextCompat.getColor(this, colorRes))
    }
    
    /**
     * Updates transmission stats UI.
     */
    private fun updateTransmissionStats(stats: TransmissionStats) {
        binding.tvTransmissionMode.text = if (stats.isFastMode) 
            getString(R.string.fast_mode) else getString(R.string.slow_mode)
        binding.tvPacketsSent.text = stats.packetsSent.toString()
        binding.tvPacketsPending.text = stats.packetsPending.toString()
    }
    
    /**
     * Checks the first-launch flow.
     */
    private fun checkFirstLaunchFlow() {
        val prefs = getSharedPreferences("app_prefs", MODE_PRIVATE)
        
        // Check whether privacy agreement has been shown.
        if (!prefs.getBoolean(PRIVACY_AGREEMENT_SHOWN, false)) {
            showPrivacyAgreement()
        } else {
            // Check permissions.
            checkPermissionsAndGuidance()
        }
    }
    
    /**
     * Shows the privacy agreement.
     */
    private fun showPrivacyAgreement() {
        val dialog = PrivacyAgreementDialog { accepted ->
            if (accepted) {
                // User accepted privacy agreement.
                getSharedPreferences("app_prefs", MODE_PRIVATE)
                    .edit()
                    .putBoolean(PRIVACY_AGREEMENT_SHOWN, true)
                    .apply()
                
                // Record user consent via PrivacyManager.
                viewModel.grantPrivacyConsent()
                
                checkPermissionsAndGuidance()
            } else {
                // User rejected; close the app.
                finish()
            }
        }
        dialog.show(supportFragmentManager, "privacy_agreement")
    }
    
    /**
     * Checks permissions and guidance flow.
     */
    private fun checkPermissionsAndGuidance() {
        lifecycleScope.launch {
            // Usage Stats permission
            if (!viewModel.hasUsageStatsPermission()) {
                showUsageStatsPermissionDialog()
            } else {
                checkBatteryOptimization()
            }
        }
    }
    
    /**
     * Shows the Usage Stats permission dialog.
     */
    private fun showUsageStatsPermissionDialog() {
        val dialog = UsageStatsPermissionDialog { granted ->
            if (granted) {
                // Navigate to settings page.
                val intent = Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)
                usageStatsPermissionLauncher.launch(intent)
            } else {
                // Continue with battery optimization check.
                checkBatteryOptimization()
            }
        }
        dialog.show(supportFragmentManager, "usage_stats_permission")
    }
    
    /**
     * Checks battery optimization settings.
     */
    private fun checkBatteryOptimization() {
        val prefs = getSharedPreferences("app_prefs", MODE_PRIVATE)
        
        // Check whether the prompt has already been shown.
        if (!prefs.getBoolean(BATTERY_OPTIMIZATION_REQUESTED, false)) {
            showBatteryOptimizationDialog()
        }
    }
    
    /**
     * Shows the battery optimization dialog.
     */
    private fun showBatteryOptimizationDialog() {
        val dialog = BatteryOptimizationDialog { disableOptimization ->
            if (disableOptimization && Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                // Navigate to battery optimization settings.
                val intent = Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
                batteryOptimizationLauncher.launch(intent)
            } else {
                // User skipped or the system does not support it.
                getSharedPreferences("app_prefs", MODE_PRIVATE)
                    .edit()
                    .putBoolean(BATTERY_OPTIMIZATION_REQUESTED, true)
                    .apply()
            }
        }
        dialog.show(supportFragmentManager, "battery_optimization")
    }
    
    /**
     * Shows an error message.
     */
    private fun showErrorMessage(message: String) {
        // This can be implemented with Snackbar or AlertDialog.
        Log.e(TAG, "Error: $message")
        // TODO: Implement a user-friendly error UI.
    }
    
    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }
    
    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_settings -> {
                // Open settings screen.
                // TODO: Implement Settings activity.
                true
            }
            R.id.action_debug -> {
                // Toggle debug mode.
                viewModel.toggleDebugMode()
                true
            }
            R.id.action_visualization -> {
                // Toggle visualization mode.
                viewModel.toggleVisualization()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        Log.i(TAG, "MainActivity destroyed")
    }
}

/**
 * Transmission stats.
 */
data class TransmissionStats(
    val isFastMode: Boolean = false,
    val packetsSent: Long = 0L,
    val packetsPending: Int = 0,
    val lastAckLatency: Long? = null
)
