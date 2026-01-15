package com.continuousauth.ui.dialogs

import android.app.Dialog
import android.os.Bundle
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.DialogFragment
import com.continuousauth.R
import com.google.android.material.dialog.MaterialAlertDialogBuilder

/**
 * Usage Stats permission dialog.
 *
 * Guides the user to grant PACKAGE_USAGE_STATS permission, used to detect the foreground app package name.
 */
class UsageStatsPermissionDialog(
    private val onResult: (grantPermission: Boolean) -> Unit
) : DialogFragment() {
    
    companion object {
        private const val TAG = "UsageStatsPermissionDialog"
    }
    
    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        val context = requireContext()
        
        return MaterialAlertDialogBuilder(context)
            .setTitle(getString(R.string.usage_stats_permission_title))
            .setMessage(getUsageStatsPermissionText())
            .setIcon(R.drawable.ic_permission)
            .setPositiveButton(getString(R.string.grant_permission)) { dialog, _ ->
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
     * Builds Usage Stats permission explanation text.
     */
    private fun getUsageStatsPermissionText(): String {
        return buildString {
            appendLine(getString(R.string.usage_stats_permission_explanation))
            appendLine()
            appendLine(getString(R.string.usage_stats_permission_purpose))
            appendLine()
            appendLine(getString(R.string.usage_stats_permission_privacy))
            appendLine()
            appendLine(getString(R.string.usage_stats_permission_steps))
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
