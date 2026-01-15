package com.continuousauth.utils
import android.app.AppOpsManager
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Process
import android.provider.Settings
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentActivity

object UsageStatsHelper {

    /**
     * Returns whether usage access permission is granted.
     */
    fun hasUsageStatsPermission(context: Context): Boolean {
        val appOps = context.getSystemService(Context.APP_OPS_SERVICE) as AppOpsManager
        val packageName = context.packageName
        val uid = Process.myUid()

        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val mode = appOps.unsafeCheckOpNoThrow(
                AppOpsManager.OPSTR_GET_USAGE_STATS,
                uid,
                packageName
            )
            mode == AppOpsManager.MODE_ALLOWED
        } else {
            @Suppress("DEPRECATION")
            val mode = appOps.checkOpNoThrow(
                AppOpsManager.OPSTR_GET_USAGE_STATS,
                uid,
                packageName
            )
            mode == AppOpsManager.MODE_ALLOWED
        }
    }

    /**
     * Open the Settings page for enabling usage access permission.
     */
    fun requestUsageStatsPermission(context: Context) {
        try {
            val intent = Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(intent)
            showPermissionGuideToast(context)
        } catch (e: Exception) {
            e.printStackTrace()
            showErrorToast(context)
        }
    }

    fun showPermissionGuideToast(context: Context) {
        Toast.makeText(
            context,
            "Find \"${getAppName(context)}\" in Settings and enable the toggle.",
            Toast.LENGTH_LONG
        ).show()
    }

    fun showErrorToast(context: Context) {
        Toast.makeText(
            context,
            "Unable to open Settings automatically. Please go to: Settings → Apps → Special app access → Usage access.",
            Toast.LENGTH_LONG
        ).show()
    }

    private fun getAppName(context: Context): String {
        return try {
            val packageManager = context.packageManager
            val applicationInfo = packageManager.getApplicationInfo(context.packageName, 0)
            packageManager.getApplicationLabel(applicationInfo).toString()
        } catch (e: Exception) {
            "This app"
        }
    }
}

/**
 * Helper using registerForActivityResult to request usage access permission.
 */
class UsageStatsPermissionLauncher private constructor() {

    // Activity Result Launcher
    private lateinit var usageStatsLauncher: androidx.activity.result.ActivityResultLauncher<Intent>

    // Callbacks
    private var onPermissionGranted: (() -> Unit)? = null
    private var onPermissionDenied: (() -> Unit)? = null

    companion object {
        /**
         * Create the launcher in an Activity.
         */
        fun create(activity: FragmentActivity): UsageStatsPermissionLauncher {
            return UsageStatsPermissionLauncher().apply {
                initLauncher(activity)
            }
        }

        /**
         * Create the launcher in a Fragment.
         */
        fun create(fragment: Fragment): UsageStatsPermissionLauncher {
            return UsageStatsPermissionLauncher().apply {
                initLauncher(fragment)
            }
        }
    }

    private fun initLauncher(owner: Any) {
        val contract = ActivityResultContracts.StartActivityForResult()

        usageStatsLauncher = when (owner) {
            is FragmentActivity -> owner.registerForActivityResult(contract) { result ->
                handleActivityResult(owner, result.resultCode)
            }
            is Fragment -> owner.registerForActivityResult(contract) { result ->
                handleActivityResult(owner.requireContext(), result.resultCode)
            }
            else -> throw IllegalArgumentException("Unsupported owner type")
        }
    }

    private fun handleActivityResult(context: Context, resultCode: Int) {
        if (UsageStatsHelper.hasUsageStatsPermission(context)) {
            onPermissionGranted?.invoke()
        } else {
            onPermissionDenied?.invoke()
        }
    }

    /**
     * Request usage stats permission.
     */
    fun requestUsageStatsPermission(
        context: Context,
        onGranted: (() -> Unit)? = null,
        onDenied: (() -> Unit)? = null
    ) {
        this.onPermissionGranted = onGranted
        this.onPermissionDenied = onDenied

        try {
            val intent = Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)
            usageStatsLauncher.launch(intent)
            UsageStatsHelper.showPermissionGuideToast(context)
        } catch (e: Exception) {
            e.printStackTrace()
            UsageStatsHelper.showErrorToast(context)
            onDenied?.invoke()
        }
    }

    /**
     * Clear callbacks.
     */
    fun dispose() {
        onPermissionGranted = null
        onPermissionDenied = null
    }
}
