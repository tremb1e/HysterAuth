package com.continuousauth.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

// Material 3 color schemes
private val DarkColorScheme = darkColorScheme(
    primary = Color(0xFF4DD0E1),           // Cyan primary
    onPrimary = Color(0xFF003940),
    primaryContainer = Color(0xFF00525B),
    onPrimaryContainer = Color(0xFF70F3FF),
    
    secondary = Color(0xFF8B9DC3),         // Blue-gray secondary
    onSecondary = Color(0xFF1F2E49),
    secondaryContainer = Color(0xFF364560),
    onSecondaryContainer = Color(0xFFC6D8FF),
    
    tertiary = Color(0xFFFF7597),          // Pink accent
    onTertiary = Color(0xFF561D30),
    tertiaryContainer = Color(0xFF723344),
    onTertiaryContainer = Color(0xFFFFD9E2),
    
    error = Color(0xFFFFB4AB),
    onError = Color(0xFF690005),
    errorContainer = Color(0xFF93000A),
    onErrorContainer = Color(0xFFFFDAD6),
    
    background = Color(0xFF0F1419),        // Dark background
    onBackground = Color(0xFFE1E2E8),
    
    surface = Color(0xFF1A1F27),           // Surface / cards
    onSurface = Color(0xFFE1E2E8),
    surfaceVariant = Color(0xFF40484F),
    onSurfaceVariant = Color(0xFFC0C8D0),
    
    surfaceTint = Color(0xFF4DD0E1),
    inverseSurface = Color(0xFFE1E2E8),
    inverseOnSurface = Color(0xFF2E3138),
    
    outline = Color(0xFF8A9299),
    outlineVariant = Color(0xFF40484F),
    scrim = Color(0xFF000000)
)

private val LightColorScheme = lightColorScheme(
    primary = Color(0xFF006874),           // Deep teal primary
    onPrimary = Color(0xFFFFFFFF),
    primaryContainer = Color(0xFF97F0FF),
    onPrimaryContainer = Color(0xFF001F24),
    
    secondary = Color(0xFF4C5F7B),         // Blue-gray secondary
    onSecondary = Color(0xFFFFFFFF),
    secondaryContainer = Color(0xFFD3E4FF),
    onSecondaryContainer = Color(0xFF051C35),
    
    tertiary = Color(0xFF984061),          // Magenta accent
    onTertiary = Color(0xFFFFFFFF),
    tertiaryContainer = Color(0xFFFFD9E2),
    onTertiaryContainer = Color(0xFF3E001D),
    
    error = Color(0xFFBA1A1A),
    onError = Color(0xFFFFFFFF),
    errorContainer = Color(0xFFFFDAD6),
    onErrorContainer = Color(0xFF410002),
    
    background = Color(0xFFFAFDFF),        // Light background
    onBackground = Color(0xFF191C1E),
    
    surface = Color(0xFFFFFBFF),           // Surface / cards
    onSurface = Color(0xFF191C1E),
    surfaceVariant = Color(0xFFDCE4E8),
    onSurfaceVariant = Color(0xFF40484C),
    
    surfaceTint = Color(0xFF006874),
    inverseSurface = Color(0xFF2E3133),
    inverseOnSurface = Color(0xFFF0F1F3),
    
    outline = Color(0xFF70787C),
    outlineVariant = Color(0xFFC0C8CC),
    scrim = Color(0xFF000000)
)

/**
 * Application theme.
 *
 * Supports dark theme and dynamic color (Android 12+).
 */
@Composable
fun ContinuousAuthTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true, // Enable dynamic colors by default (Android 12+)
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }
    
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.primary.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }
    
    MaterialTheme(
        colorScheme = colorScheme,
        typography = AppTypography,
        content = content
    )
}

/**
 * Extended colors for special state rendering.
 */
object ExtendedColors {
    // Sensor status colors
    val sensorActive = Color(0xFF4CAF50)
    val sensorInactive = Color(0xFF757575)
    val sensorError = Color(0xFFFF5252)
    
    // Network quality colors
    val networkExcellent = Color(0xFF4CAF50)
    val networkGood = Color(0xFFFFC107)
    val networkPoor = Color(0xFFFF9800)
    val networkOffline = Color(0xFFFF5252)
    
    // Transmission mode colors
    val fastMode = Color(0xFFFF6B6B)
    val slowMode = Color(0xFF4ECDC4)
    
    // Chart colors (softer palette)
    val chartAccelerometer = Color(0xFF7E9FC2)  // Soft blue-gray
    val chartGyroscope = Color(0xFF8FBC8F)       // Soft green
    val chartMagnetometer = Color(0xFFD4A76A)    // Soft ochre
    
    // Gradient colors
    val gradientStart = Color(0xFF667EEA)
    val gradientEnd = Color(0xFF764BA2)
    
    // Success / warning / error
    val success = Color(0xFF4CAF50)
    val warning = Color(0xFFFFC107)
    val error = Color(0xFFF44336)
    val info = Color(0xFF2196F3)
}
