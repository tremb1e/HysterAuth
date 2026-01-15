package com.continuousauth.ui.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.continuousauth.databinding.FragmentSensorDataBinding
import com.continuousauth.ui.adapters.RecentAppsAdapter
import com.continuousauth.ui.chart.ChartManager
import com.continuousauth.ui.chart.SensorChartView
import javax.inject.Inject
import com.continuousauth.ui.viewmodels.SensorDataViewModel
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

/**
 * Sensor data fragment.
 *
 * Displays real-time sensor data, collapsible charts, and a list of recent apps.
 */
@AndroidEntryPoint
class SensorDataFragment : Fragment() {

    private var _binding: FragmentSensorDataBinding? = null
    private val binding get() = _binding!!
    
    private val viewModel: SensorDataViewModel by viewModels()
    
    @Inject
    lateinit var chartManager: ChartManager
    
    private lateinit var accelerometerChart: SensorChartView
    private lateinit var gyroscopeChart: SensorChartView
    private lateinit var magnetometerChart: SensorChartView
    private lateinit var recentAppsAdapter: RecentAppsAdapter
    
    private val dateTimeFormatter = java.text.SimpleDateFormat("HH:mm:ss.SSS", java.util.Locale.getDefault())
    
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSensorDataBinding.inflate(inflater, container, false)
        return binding.root
    }
    
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        setupViews()
        observeViewModel()
        
        // Start sensor collection.
        viewModel.startSensorCollection()
    }
    
    private fun setupViews() {
        // Initialize chart views.
        accelerometerChart = SensorChartView(requireContext())
        gyroscopeChart = SensorChartView(requireContext())
        magnetometerChart = SensorChartView(requireContext())
        
        // Set distinct color schemes per sensor.
        accelerometerChart.setColorScheme(
            android.graphics.Color.RED,     // X axis - red
            android.graphics.Color.GREEN,   // Y axis - green
            android.graphics.Color.BLUE     // Z axis - blue
        )
        
        gyroscopeChart.setColorScheme(
            android.graphics.Color.rgb(255, 140, 0),    // X axis - orange
            android.graphics.Color.rgb(0, 191, 255),    // Y axis - deep sky blue
            android.graphics.Color.rgb(148, 0, 211)     // Z axis - purple
        )
        
        magnetometerChart.setColorScheme(
            android.graphics.Color.rgb(255, 20, 147),   // X axis - deep pink
            android.graphics.Color.rgb(34, 139, 34),    // Y axis - forest green
            android.graphics.Color.rgb(70, 130, 180)    // Z axis - steel blue
        )
        
        // Add to containers and enable click-to-expand behavior (detailed view).
        binding.accelerometerChartContainer.addView(accelerometerChart)
        binding.gyroscopeChartContainer.addView(gyroscopeChart)
        binding.magnetometerChartContainer.addView(magnetometerChart)
        
        // Enable chart interactions.
        accelerometerChart.enableInteractiveMode()
        gyroscopeChart.enableInteractiveMode()
        magnetometerChart.enableInteractiveMode()
        
        // Charts are collapsed by default.
        binding.chartContainer.visibility = View.GONE
        binding.tvChartToggle.text = getString(com.continuousauth.R.string.show_chart)
        
        // Toggle expand/collapse.
        binding.layoutChartHeader.setOnClickListener {
            toggleChartVisibility()
        }
        
        // Recent apps list.
        recentAppsAdapter = RecentAppsAdapter()
        binding.rvRecentApps.apply {
            layoutManager = LinearLayoutManager(requireContext())
            adapter = recentAppsAdapter
            setHasFixedSize(true)
        }
    }
    
    private fun observeViewModel() {
        // Observe accelerometer data.
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.accelerometerData.collectLatest { data ->
                val currentTime = System.currentTimeMillis()
                binding.tvAccelerometerData.text = 
                    "X: %.3f\nY: %.3f\nZ: %.3f".format(data.x, data.y, data.z)
                binding.tvAccelerometerTimestamp.text = 
                    "Time: ${dateTimeFormatter.format(java.util.Date(currentTime))}"
                
                // Update chart if visible.
                if (binding.chartContainer.visibility == View.VISIBLE) {
                    accelerometerChart.addDataPoint(data.x, data.y, data.z, currentTime)
                }
            }
        }
        
        // Observe gyroscope data.
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.gyroscopeData.collectLatest { data ->
                val currentTime = System.currentTimeMillis()
                binding.tvGyroscopeData.text = 
                    "X: %.3f\nY: %.3f\nZ: %.3f".format(data.x, data.y, data.z)
                binding.tvGyroscopeTimestamp.text = 
                    "Time: ${dateTimeFormatter.format(java.util.Date(currentTime))}"
                
                // Update chart if visible.
                if (binding.chartContainer.visibility == View.VISIBLE) {
                    gyroscopeChart.addDataPoint(data.x, data.y, data.z, currentTime)
                }
            }
        }
        
        // Observe magnetometer data.
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.magnetometerData.collectLatest { data ->
                val currentTime = System.currentTimeMillis()
                binding.tvMagnetometerData.text = 
                    "X: %.3f\nY: %.3f\nZ: %.3f".format(data.x, data.y, data.z)
                binding.tvMagnetometerTimestamp.text = 
                    "Time: ${dateTimeFormatter.format(java.util.Date(currentTime))}"
                
                // Update chart if visible.
                if (binding.chartContainer.visibility == View.VISIBLE) {
                    magnetometerChart.addDataPoint(data.x, data.y, data.z, currentTime)
                }
            }
        }
        
        // Observe recent apps (live updates).
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.recentApps.collectLatest { apps ->
                // Limit to the most recent 10 apps.
                val recentTenApps = apps.take(10)
                recentAppsAdapter.submitList(recentTenApps)
            }
        }
        
        // Observe sensor status.
        viewModel.sensorsActive.observe(viewLifecycleOwner) { isActive ->
            updateSensorStatus(isActive)
        }
    }
    
    private fun toggleChartVisibility() {
        if (binding.chartContainer.visibility == View.VISIBLE) {
            // Collapse charts.
            binding.chartContainer.visibility = View.GONE
            binding.tvChartToggle.text = getString(com.continuousauth.R.string.show_chart)
            binding.ivChartToggle.rotation = 0f
            
            // Pause chart updates.
            accelerometerChart.pauseUpdates()
            gyroscopeChart.pauseUpdates()
            magnetometerChart.pauseUpdates()
        } else {
            // Expand charts.
            binding.chartContainer.visibility = View.VISIBLE
            binding.tvChartToggle.text = getString(com.continuousauth.R.string.hide_chart)
            binding.ivChartToggle.rotation = 180f
            
            // Resume chart updates.
            accelerometerChart.resumeUpdates()
            gyroscopeChart.resumeUpdates()
            magnetometerChart.resumeUpdates()
        }
    }
    
    private fun updateSensorStatus(isActive: Boolean) {
        val statusText = if (isActive) {
            getString(com.continuousauth.R.string.sensor_active)
        } else {
            getString(com.continuousauth.R.string.sensor_inactive)
        }
        
        val statusColor = if (isActive) {
            requireContext().getColor(com.continuousauth.R.color.sensor_active)
        } else {
            requireContext().getColor(com.continuousauth.R.color.sensor_inactive)
        }
        
        binding.viewStatusIndicator.setBackgroundColor(statusColor)
        binding.tvSensorStatus.text = statusText
        binding.tvSensorStatus.setTextColor(statusColor)
    }
    
    override fun onDestroyView() {
        super.onDestroyView()
        
        // Release chart resources.
        accelerometerChart.cleanup()
        gyroscopeChart.cleanup()
        magnetometerChart.cleanup()
        
        _binding = null
    }
}
