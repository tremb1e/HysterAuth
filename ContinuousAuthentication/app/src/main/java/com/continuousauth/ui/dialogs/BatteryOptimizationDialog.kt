package com.continuousauth.ui.dialogs

import android.app.Dialog
import android.os.Bundle
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.DialogFragment
import com.continuousauth.R
import com.google.android.material.dialog.MaterialAlertDialogBuilder

/**
 * Battery optimization dialog.
 *
 * Guides the user to disable battery optimizations for the app to keep background collection stable.
 */
class BatteryOptimizationDialog(
    private val onResult: (disableOptimization: Boolean) -> Unit
) : DialogFragment() {
    
    companion object {
        private const val TAG = "BatteryOptimizationDialog"
    }
    
    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        val context = requireContext()
        
        return MaterialAlertDialogBuilder(context)
            .setTitle(getString(R.string.battery_optimization_title))
            .setMessage(getBatteryOptimizationText())
            .setIcon(R.drawable.ic_battery_optimization)
            .setPositiveButton(getString(R.string.disable_optimization)) { dialog, _ ->
                dialog.dismiss()
                onResult(true)
            }
            .setNegativeButton(getString(R.string.skip)) { dialog, _ ->
                dialog.dismiss()
                onResult(false)
            }
            .setCancelable(true)
            .create()
    }
    
    /**
     * Builds battery optimization explanation text.
     */
    private fun getBatteryOptimizationText(): String {
        return buildString {
            appendLine(getString(R.string.battery_optimization_explanation))
            appendLine()
            appendLine(getString(R.string.battery_optimization_benefits))
            appendLine()
            appendLine(getString(R.string.battery_optimization_how_to))
        }
    }
    
    override fun onStart() {
        super.onStart()
        
        // Adjust dialog width.
        dialog?.window?.let { window ->
            val attributes = window.attributes
            attributes.width = (resources.displayMetrics.widthPixels * 0.9).toInt()
            window.attributes = attributes
        }
    }
}
