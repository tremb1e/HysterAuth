package com.continuousauth.ui.fragments

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.lifecycle.lifecycleScope
import com.continuousauth.R
import com.continuousauth.databinding.FragmentServerConfigBinding
import com.continuousauth.ui.viewmodels.ServerConfigViewModel
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.launch

/**
 * Server configuration fragment.
 *
 * Manages server connection settings and upload controls.
 */
@AndroidEntryPoint
class ServerConfigFragment : Fragment() {

    private var _binding: FragmentServerConfigBinding? = null
    private val binding get() = _binding!!
    
    private val viewModel: ServerConfigViewModel by viewModels()
    
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentServerConfigBinding.inflate(inflater, container, false)
        return binding.root
    }
    
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        setupViews()
        observeViewModel()
        
        // Load persisted config.
        viewModel.loadServerConfig()
    }
    
    private fun setupViews() {
        // Device ID copy.
        binding.layoutDeviceId.setOnClickListener {
            copyDeviceIdToClipboard()
        }
        
        // Server connection test button.
        binding.btnTestConnection.setOnClickListener {
            testServerConnection()
        }
        
        // Start/stop upload button.
        binding.btnToggleUpload.setOnClickListener {
            toggleUpload()
        }
        
        // Persist on focus loss.
        binding.etServerIp.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) {
                saveServerConfig()
            }
        }
        
        binding.etServerPort.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) {
                saveServerConfig()
            }
        }
    }
    
    private fun observeViewModel() {
        // Observe device ID.
        viewModel.deviceId.observe(viewLifecycleOwner) { deviceId ->
            binding.tvDeviceId.text = deviceId
        }
        
        // Observe server config.
        viewModel.serverConfig.observe(viewLifecycleOwner) { config ->
            if (binding.etServerIp.text.toString() != config.ip) {
                binding.etServerIp.setText(config.ip)
            }
            if (binding.etServerPort.text.toString() != config.port.toString()) {
                binding.etServerPort.setText(config.port.toString())
            }
        }
        
        // Observe connection state.
        viewModel.connectionStatus.observe(viewLifecycleOwner) { status ->
            updateConnectionStatus(status)
        }
        
        // Observe upload state.
        viewModel.isUploading.observe(viewLifecycleOwner) { isUploading ->
            updateUploadButton(isUploading)
        }
        
        // Observe error messages.
        viewModel.errorMessage.observe(viewLifecycleOwner) { message ->
            message?.let {
                showError(it)
            }
        }
        
        // Observe success messages.
        viewModel.successMessage.observe(viewLifecycleOwner) { message ->
            message?.let {
                showSuccess(it)
            }
        }
    }
    
    private fun copyDeviceIdToClipboard() {
        val clipboard = requireContext().getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText("Device ID", binding.tvDeviceId.text)
        clipboard.setPrimaryClip(clip)
        
        Toast.makeText(requireContext(), getString(R.string.device_id_copied), Toast.LENGTH_SHORT).show()
    }
    
    private fun testServerConnection() {
        // Persist current config first.
        saveServerConfig()
        
        // Show progress.
        binding.progressConnection.visibility = View.VISIBLE
        binding.btnTestConnection.isEnabled = false
        binding.btnTestConnection.text = getString(R.string.testing_connection)
        
        // Run connectivity test.
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.testConnection()
            
            // Hide progress.
            binding.progressConnection.visibility = View.GONE
            binding.btnTestConnection.isEnabled = true
            binding.btnTestConnection.text = getString(R.string.test_connection)
        }
    }
    
    private fun toggleUpload() {
        if (viewModel.isUploading.value == true) {
            // Confirm before stopping upload.
            showStopUploadConfirmation()
        } else {
            // Validate config before starting upload.
            if (validateServerConfig()) {
                startUpload()
            }
        }
    }
    
    private fun showStopUploadConfirmation() {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle(getString(R.string.stop_upload_title))
            .setMessage(getString(R.string.stop_upload_message))
            .setPositiveButton(getString(R.string.stop)) { _, _ ->
                stopUpload()
            }
            .setNegativeButton(getString(R.string.cancel), null)
            .show()
    }
    
    private fun startUpload() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.startUpload()
        }
    }
    
    private fun stopUpload() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.stopUpload()
        }
    }
    
    private fun saveServerConfig() {
        val ip = binding.etServerIp.text.toString().trim()
        val portText = binding.etServerPort.text.toString().trim()
        
        if (ip.isNotEmpty() && portText.isNotEmpty()) {
            try {
                val port = portText.toInt()
                viewModel.saveServerConfig(ip, port)
            } catch (e: NumberFormatException) {
                showError(getString(R.string.invalid_port))
            }
        }
    }
    
    private fun validateServerConfig(): Boolean {
        val ip = binding.etServerIp.text.toString().trim()
        val portText = binding.etServerPort.text.toString().trim()
        
        if (ip.isEmpty()) {
            binding.tilServerIp.error = getString(R.string.error_empty_ip)
            return false
        }
        
        if (portText.isEmpty()) {
            binding.tilServerPort.error = getString(R.string.error_empty_port)
            return false
        }
        
        try {
            val port = portText.toInt()
            if (port < 1 || port > 65535) {
                binding.tilServerPort.error = getString(R.string.error_invalid_port_range)
                return false
            }
        } catch (e: NumberFormatException) {
            binding.tilServerPort.error = getString(R.string.error_invalid_port)
            return false
        }
        
        // Clear errors.
        binding.tilServerIp.error = null
        binding.tilServerPort.error = null
        
        return true
    }
    
    private fun updateConnectionStatus(status: ServerConfigViewModel.ConnectionStatus) {
        val (text, color) = when (status) {
            ServerConfigViewModel.ConnectionStatus.CONNECTED -> {
                Pair(getString(R.string.connected), R.color.status_connected)
            }
            ServerConfigViewModel.ConnectionStatus.DISCONNECTED -> {
                Pair(getString(R.string.disconnected), R.color.status_disconnected)
            }
            ServerConfigViewModel.ConnectionStatus.CONNECTING -> {
                Pair(getString(R.string.connecting), R.color.status_connecting)
            }
            ServerConfigViewModel.ConnectionStatus.ERROR -> {
                Pair(getString(R.string.connection_error), R.color.status_error)
            }
        }
        
        binding.tvConnectionStatus.text = text
        binding.tvConnectionStatus.setTextColor(requireContext().getColor(color))
        binding.viewConnectionIndicator.setBackgroundColor(requireContext().getColor(color))
    }
    
    private fun updateUploadButton(isUploading: Boolean) {
        if (isUploading) {
            binding.btnToggleUpload.text = getString(R.string.stop_upload)
            binding.btnToggleUpload.setIconResource(R.drawable.ic_stop)
            binding.btnToggleUpload.setBackgroundColor(requireContext().getColor(R.color.button_stop))
        } else {
            binding.btnToggleUpload.text = getString(R.string.start_upload)
            binding.btnToggleUpload.setIconResource(R.drawable.ic_play_arrow)
            binding.btnToggleUpload.setBackgroundColor(requireContext().getColor(R.color.button_start))
        }
    }
    
    private fun showError(message: String) {
        Toast.makeText(requireContext(), message, Toast.LENGTH_LONG).show()
    }
    
    private fun showSuccess(message: String) {
        Toast.makeText(requireContext(), message, Toast.LENGTH_SHORT).show()
    }
    
    override fun onDestroyView() {
        super.onDestroyView()
        // Persist the latest config.
        saveServerConfig()
        _binding = null
    }
}
