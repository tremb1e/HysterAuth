package com.continuousauth.ui

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController
import com.continuousauth.R
import com.continuousauth.databinding.ActivityMainNavBinding
import com.continuousauth.ui.dialogs.PrivacyAgreementDialog
import dagger.hilt.android.AndroidEntryPoint

/**
 * Main activity with bottom navigation.
 *
 * Manages navigation across the main screens.
 */
@AndroidEntryPoint
class MainActivityNav : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainNavBinding
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize ViewBinding.
        binding = ActivityMainNavBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // Set up navigation.
        setupNavigation()
        
        // Check first-launch flow.
        checkFirstLaunch()
    }
    
    /**
     * Sets up navigation components.
     */
    private fun setupNavigation() {
        // Get NavController.
        val navHostFragment = supportFragmentManager
            .findFragmentById(R.id.nav_host_fragment) as NavHostFragment
        val navController = navHostFragment.navController
        
        // Hook up bottom navigation.
        binding.bottomNavigation.setupWithNavController(navController)
        
        // Configure ActionBar.
        val appBarConfiguration = AppBarConfiguration(
            setOf(
                R.id.navigation_sensor_data,
                R.id.navigation_server_config,
                R.id.navigation_detailed_info
            )
        )
        setupActionBarWithNavController(navController, appBarConfiguration)
    }
    
    /**
     * Checks whether this is the first launch.
     */
    private fun checkFirstLaunch() {
        val prefs = getSharedPreferences("app_prefs", MODE_PRIVATE)
        val isFirstLaunch = prefs.getBoolean("is_first_launch", true)
        
        if (isFirstLaunch) {
            showPrivacyAgreement()
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
                    .putBoolean("is_first_launch", false)
                    .apply()
            } else {
                // User rejected; close the app.
                finish()
            }
        }
        dialog.show(supportFragmentManager, "privacy_agreement")
    }
}
