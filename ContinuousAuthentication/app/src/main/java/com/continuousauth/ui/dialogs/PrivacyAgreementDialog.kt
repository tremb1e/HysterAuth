package com.continuousauth.ui.dialogs

import android.app.Dialog
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.DialogFragment
import com.continuousauth.R
import com.google.android.material.dialog.MaterialAlertDialogBuilder

/**
 * Privacy agreement dialog.
 *
 * Shown on first launch; the user must accept to continue using the app.
 */
class PrivacyAgreementDialog(
    private val onResult: (accepted: Boolean) -> Unit
) : DialogFragment() {
    
    companion object {
        private const val TAG = "PrivacyAgreementDialog"
    }
    
    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        val context = requireContext()
        
        return MaterialAlertDialogBuilder(context)
            .setTitle(getString(R.string.privacy_agreement_title))
            .setMessage(getPrivacyAgreementText())
            .setPositiveButton(getString(R.string.agree)) { dialog, _ ->
                dialog.dismiss()
                onResult(true)
            }
            .setNegativeButton(getString(R.string.disagree)) { dialog, _ ->
                dialog.dismiss()
                onResult(false)
            }
            .setCancelable(false) // User must explicitly choose
            .create()
    }
    
    /**
     * Builds privacy agreement text.
     *
     * Must clearly describe data collection, usage, storage, and sharing policies.
     */
    private fun getPrivacyAgreementText(): String {
        return buildString {
            appendLine(getString(R.string.privacy_welcome))
            appendLine()
            
            appendLine(getString(R.string.privacy_data_collection_title))
            appendLine(getString(R.string.privacy_data_collection_content))
            appendLine()
            
            appendLine(getString(R.string.privacy_data_usage_title))
            appendLine(getString(R.string.privacy_data_usage_content))
            appendLine()
            
            appendLine(getString(R.string.privacy_data_storage_title))
            appendLine(getString(R.string.privacy_data_storage_content))
            appendLine()
            
            appendLine(getString(R.string.privacy_data_sharing_title))
            appendLine(getString(R.string.privacy_data_sharing_content))
            appendLine()
            
            appendLine(getString(R.string.privacy_user_rights_title))
            appendLine(getString(R.string.privacy_user_rights_content))
            appendLine()
            
            appendLine(getString(R.string.privacy_contact_title))
            appendLine(getString(R.string.privacy_contact_content))
        }
    }
    
    override fun onStart() {
        super.onStart()
        
        // Make dialog message text scrollable.
        dialog?.findViewById<android.widget.TextView>(android.R.id.message)?.apply {
            movementMethod = ScrollingMovementMethod()
            maxHeight = resources.displayMetrics.heightPixels / 2 // Half of screen height
        }
    }
}
