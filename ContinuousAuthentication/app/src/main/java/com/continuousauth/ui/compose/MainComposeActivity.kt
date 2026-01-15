package com.continuousauth.ui.compose

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.fragment.app.FragmentActivity
import androidx.navigation.NavController
import androidx.navigation.compose.*
import com.continuousauth.R
import com.continuousauth.ui.MainViewModel
import com.continuousauth.ui.viewmodels.SensorDataViewModel
import com.continuousauth.ui.compose.screens.*
import com.continuousauth.ui.theme.ContinuousAuthTheme
import com.continuousauth.ui.theme.ExtendedColors
import com.continuousauth.ui.compose.dialogs.PrivacyAgreementDialog
import com.continuousauth.privacy.ConsentState
import com.continuousauth.utils.UsageStatsHelper
import com.continuousauth.utils.UsageStatsPermissionLauncher
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.launch

/**
 * Main Compose activity.
 *
 * Uses Material 3 and bottom navigation.
 */
@OptIn(ExperimentalAnimationApi::class)
@AndroidEntryPoint
class MainComposeActivity : FragmentActivity() {
    private val TAG = "MainComposeActivity"
    private val viewModel: MainViewModel by viewModels()
    private val sensorViewModel: SensorDataViewModel by viewModels()

    // Permission launcher
    private lateinit var usageStatsLauncher: UsageStatsPermissionLauncher

    companion object {
        const val PREFS_NAME = "app_prefs"
        const val PRIVACY_AGREEMENT_SHOWN = "privacy_agreement_shown"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    setContent {
        ContinuousAuthTheme {
            MainApp(viewModel, sensorViewModel)
        }
    }

    // Initialize launcher
    try {
        usageStatsLauncher = UsageStatsPermissionLauncher.create(this)

        if (UsageStatsHelper.hasUsageStatsPermission(this)) {
            onUsageStatsPermissionGranted()
        } else {
            // Request permission via launcher.
            usageStatsLauncher.requestUsageStatsPermission(
                context = this,
                onGranted = {
                    runOnUiThread {
                        onUsageStatsPermissionGranted()
                    }
                },
                onDenied = {
                    runOnUiThread {
                        Toast.makeText(
                            this@MainComposeActivity,
                            "Usage access permission denied",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            )
        }
    } catch (e: Exception) {
        // Handle failures creating/using the permission launcher.
        runOnUiThread {
            Toast.makeText(
                this@MainComposeActivity,
                "Permission initialization failed. Please try again.",
                Toast.LENGTH_SHORT
            ).show()
        }
    }
}


        private fun onUsageStatsPermissionGranted() {
        Toast.makeText(this, "Usage access permission granted", Toast.LENGTH_SHORT).show()
        Log.d(TAG, "Usage access permission granted - ${System.currentTimeMillis()}")
    }


    /**
     * Main app composable.
     */
    @Composable
    fun MainApp(viewModel: MainViewModel, sensorViewModel: SensorDataViewModel) {
        val navController = rememberNavController()
        val currentRoute = navController.currentBackStackEntryAsState().value?.destination?.route
        val context = LocalContext.current

        // Check privacy agreement state.
        val prefs =
            context.getSharedPreferences(MainComposeActivity.PREFS_NAME, Context.MODE_PRIVATE)
        var showPrivacyDialog by remember {
            mutableStateOf(!prefs.getBoolean(MainComposeActivity.PRIVACY_AGREEMENT_SHOWN, false))
        }

        // Observe consent state.
        val consentStateLiveData = viewModel.consentState.observeAsState()
        val consentState = consentStateLiveData.value ?: ConsentState.UNKNOWN

        // Show privacy agreement dialog if needed.
        if (showPrivacyDialog) {
            PrivacyAgreementDialog(
                onAccept = {
                    // User accepted privacy agreement.
                    prefs.edit()
                        .putBoolean(MainComposeActivity.PRIVACY_AGREEMENT_SHOWN, true)
                        .apply()

                    // Record consent in ViewModel.
                    viewModel.grantPrivacyConsent()

                    showPrivacyDialog = false
                },
                onDecline = {
                    // User declined; exit the app.
                    (context as? ComponentActivity)?.finish()
                }
            )
        }

        // Show a blocking UI if consent is not granted.
        if (consentState != ConsentState.GRANTED && !showPrivacyDialog) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Card(
                    modifier = Modifier.padding(32.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "Please accept the privacy agreement first",
                            style = MaterialTheme.typography.headlineSmall,
                            color = MaterialTheme.colorScheme.onErrorContainer
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Button(
                            onClick = { showPrivacyDialog = true },
                            colors = ButtonDefaults.buttonColors(
                                containerColor = MaterialTheme.colorScheme.error
                            )
                        ) {
                            Text("View privacy agreement")
                        }
                    }
                }
            }
        } else {
            // Only show the main UI after consent is granted.
            Scaffold(
                bottomBar = {
                    AnimatedBottomBar(
                        currentRoute = currentRoute,
                        onNavigate = { route ->
                            navController.navigate(route) {
                                popUpTo(navController.graph.startDestinationId) {
                                    saveState = true
                                }
                                launchSingleTop = true
                                restoreState = true
                            }
                        }
                    )
                },
                containerColor = MaterialTheme.colorScheme.background
            ) { paddingValues ->
                NavHost(
                    navController = navController,
                    startDestination = Screen.Sensors.route,
                    modifier = Modifier.padding(paddingValues),
                    enterTransition = {
                        fadeIn(animationSpec = tween(300)) + slideInHorizontally(
                            initialOffsetX = { it },
                            animationSpec = tween(300)
                        )
                    },
                    exitTransition = {
                        fadeOut(animationSpec = tween(300)) + slideOutHorizontally(
                            targetOffsetX = { -it },
                            animationSpec = tween(300)
                        )
                    }
                ) {
                    composable(Screen.Sensors.route) {
                        SensorsScreen(viewModel, sensorViewModel)
                    }
                    composable(Screen.Server.route) {
                        ServerConfigScreen(viewModel)
                    }
                    composable(Screen.Details.route) {
                        DetailedInfoScreen(
                            onNavigateBack = { navController.popBackStack() }
                        )
                    }

                    composable(Screen.Privacy.route) {
                        PrivacySettingsScreen(
                            onNavigateBack = { navController.popBackStack() }
                        )
                    }
                }
            }
        } // End else branch
    }

    /**
     * Animated bottom navigation bar.
     */
    @OptIn(ExperimentalAnimationApi::class)
    @Composable
    fun AnimatedBottomBar(
        currentRoute: String?,
        onNavigate: (String) -> Unit
    ) {
        val context = LocalContext.current
        val screens = listOf(Screen.Sensors, Screen.Server, Screen.Details, Screen.Privacy)

        NavigationBar(
            modifier = Modifier
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            MaterialTheme.colorScheme.surface.copy(alpha = 0.95f),
                            MaterialTheme.colorScheme.surface
                        )
                    )
                ),
            containerColor = Color.Transparent,
            tonalElevation = 8.dp
        ) {
            screens.forEach { screen ->
                val selected = currentRoute == screen.route
                val screenTitle = getScreenTitle(context, screen)
                val animatedWeight by animateFloatAsState(
                    targetValue = if (selected) 1.5f else 1f,
                    animationSpec = spring(
                        dampingRatio = Spring.DampingRatioMediumBouncy,
                        stiffness = Spring.StiffnessLow
                    ),
                    label = "weight"
                )

                NavigationBarItem(
                    selected = selected,
                    onClick = { onNavigate(screen.route) },
                    icon = {
                        AnimatedContent(
                            targetState = selected,
                            transitionSpec = {
                                scaleIn(animationSpec = tween(200)) with scaleOut(
                                    animationSpec = tween(
                                        200
                                    )
                                )
                            },
                            label = "icon"
                        ) { isSelected ->
                            Icon(
                                imageVector = if (isSelected) screen.selectedIcon else screen.unselectedIcon,
                                contentDescription = screenTitle,
                                modifier = Modifier.size(if (isSelected) 28.dp else 24.dp)
                            )
                        }
                    },
                    label = {
                        AnimatedVisibility(
                            visible = selected,
                            enter = fadeIn() + expandVertically(),
                            exit = fadeOut() + shrinkVertically()
                        ) {
                            Text(
                                text = screenTitle,
                                style = MaterialTheme.typography.labelMedium,
                                modifier = Modifier.padding(top = 4.dp)
                            )
                        }
                    },
                    colors = NavigationBarItemDefaults.colors(
                        selectedIconColor = MaterialTheme.colorScheme.primary,
                        selectedTextColor = MaterialTheme.colorScheme.primary,
                        unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
                        indicatorColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
                    ),
                    modifier = Modifier.weight(animatedWeight)
                )
            }
        }
    }

    /**
     * Screen navigation definition.
     */
    @Composable
    fun getScreenTitle(context: Context, screen: Screen): String {
        return when (screen) {
            Screen.Sensors -> context.getString(R.string.nav_sensor_data)
            Screen.Server -> context.getString(R.string.nav_server_config)
            Screen.Details -> context.getString(R.string.nav_detailed_info)
            Screen.Privacy -> "Continuous Authentication"
        }
    }

    sealed class Screen(
        val route: String,
        val selectedIcon: ImageVector,
        val unselectedIcon: ImageVector
    ) {
        object Sensors : Screen(
            route = "sensors",
            selectedIcon = Icons.Filled.Sensors,
            unselectedIcon = Icons.Outlined.Sensors
        )

        object Server : Screen(
            route = "server",
            selectedIcon = Icons.Filled.CloudQueue,
            unselectedIcon = Icons.Outlined.CloudQueue
        )

        object Details : Screen(
            route = "details",
            selectedIcon = Icons.Filled.Dashboard,
            unselectedIcon = Icons.Outlined.Dashboard
        )

        object Privacy : Screen(
            route = "privacy",
            selectedIcon = Icons.Filled.Security,
            unselectedIcon = Icons.Outlined.Security
        )
    }
}
