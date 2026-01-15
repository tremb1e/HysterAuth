package com.continuousauth.time

import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Enhanced time synchronization module.
 *
 * Uses NTP to synchronize time and provides accurate UTC timestamps.
 */
@Singleton
class EnhancedTimeSync @Inject constructor(
    private val coroutineScope: CoroutineScope
) {
    
    companion object {
        private const val TAG = "EnhancedTimeSync"
        private const val NTP_SERVER = "pool.ntp.org"
        private const val NTP_PORT = 123
        private const val NTP_PACKET_SIZE = 48
        private const val NTP_TIMESTAMP_OFFSET = 2208988800L // Seconds between 1900 and 1970 epochs
        private const val SYNC_INTERVAL_MS = 3600000L // Sync once per hour
        private const val RETRY_DELAY_MS = 5000L // Retry delay
        private const val MAX_RETRIES = 3
        private const val TIMEOUT_MS = 10000 // 10s timeout
    }
    
    // NTP offset (ms)
    private val _ntpOffset = MutableStateFlow(0L)
    val ntpOffset: StateFlow<Long> = _ntpOffset.asStateFlow()
    
    // Last sync time
    private var lastSyncTime = 0L
    
    // Sync status
    private val _syncStatus = MutableStateFlow(SyncStatus.IDLE)
    val syncStatus: StateFlow<SyncStatus> = _syncStatus.asStateFlow()
    
    // Sync accuracy (ms)
    private val _syncAccuracy = MutableStateFlow(0L)
    val syncAccuracy: StateFlow<Long> = _syncAccuracy.asStateFlow()
    
    private var syncJob: Job? = null
    
    /**
     * Starts periodic time synchronization.
     */
    fun startPeriodicSync() {
        stopPeriodicSync()
        
        syncJob = coroutineScope.launch {
            while (isActive) {
                syncTime()
                delay(SYNC_INTERVAL_MS)
            }
        }
        
        Log.i(TAG, "Starting periodic NTP time sync")
    }
    
    /**
     * Stops periodic time synchronization.
     */
    fun stopPeriodicSync() {
        syncJob?.cancel()
        syncJob = null
        Log.i(TAG, "Stopping periodic NTP time sync")
    }
    
    /**
     * Triggers a sync immediately.
     */
    suspend fun syncTime(): Boolean {
        _syncStatus.value = SyncStatus.SYNCING
        
        var retryCount = 0
        while (retryCount < MAX_RETRIES) {
            try {
                val offset = performNtpSync()
                _ntpOffset.value = offset
                lastSyncTime = System.currentTimeMillis()
                _syncStatus.value = SyncStatus.SUCCESS
                
                Log.i(TAG, "NTP sync succeeded; offset: ${offset}ms")
                return true
                
            } catch (e: Exception) {
                retryCount++
                Log.w(TAG, "NTP sync failed (attempt $retryCount/$MAX_RETRIES): ${e.message}")
                
                if (retryCount < MAX_RETRIES) {
                    delay(RETRY_DELAY_MS)
                } else {
                    _syncStatus.value = SyncStatus.ERROR
                    Log.e(TAG, "NTP sync failed; max retries reached")
                }
            }
        }
        
        return false
    }
    
    /**
     * Performs an NTP sync.
     */
    private suspend fun performNtpSync(): Long = withContext(Dispatchers.IO) {
        val socket = DatagramSocket()
        socket.soTimeout = TIMEOUT_MS
        
        try {
            // Build NTP request packet.
            val ntpData = ByteArray(NTP_PACKET_SIZE)
            ntpData[0] = 0x1B // LI = 0, VN = 3, Mode = 3 (Client)
            
            // Capture send time.
            val requestTime = System.currentTimeMillis()
            val requestTicks = SystemClock.elapsedRealtime()
            
            // Send request.
            val address = InetAddress.getByName(NTP_SERVER)
            val packet = DatagramPacket(ntpData, ntpData.size, address, NTP_PORT)
            socket.send(packet)
            
            // Receive response.
            val response = DatagramPacket(ntpData, ntpData.size)
            socket.receive(response)
            
            // Capture receive time.
            val responseTime = System.currentTimeMillis()
            val responseTicks = SystemClock.elapsedRealtime()
            
            // Parse NTP timestamp.
            val ntpTime = parseNtpTime(ntpData)
            
            // Compute round-trip delay.
            val roundTripTime = responseTicks - requestTicks
            
            // Compute clock offset.
            // offset = ((T2 - T1) + (T3 - T4)) / 2
            // Simplified: offset = ntpTime - localTime - roundTripTime/2
            val localTime = requestTime + roundTripTime / 2
            val offset = ntpTime - localTime
            
            // Update sync accuracy.
            _syncAccuracy.value = roundTripTime / 2
            
            Log.d(TAG, "NTP sync details - RTT: ${roundTripTime}ms, accuracy: ${roundTripTime / 2}ms")
            
            offset
            
        } finally {
            socket.close()
        }
    }
    
    /**
     * Parses NTP timestamp.
     */
    private fun parseNtpTime(data: ByteArray): Long {
        // Transmit timestamp is located at bytes 40-47.
        val transmitTimeOffset = 40
        
        // Read seconds part (32-bit).
        var seconds = 0L
        for (i in 0..3) {
            seconds = (seconds shl 8) or (data[transmitTimeOffset + i].toLong() and 0xFF)
        }
        
        // Read fractional part (32-bit).
        var fraction = 0L
        for (i in 4..7) {
            fraction = (fraction shl 8) or (data[transmitTimeOffset + i].toLong() and 0xFF)
        }
        
        // Convert to milliseconds since Unix epoch.
        val unixSeconds = seconds - NTP_TIMESTAMP_OFFSET
        val milliseconds = (fraction * 1000L) / 0x100000000L
        
        return unixSeconds * 1000L + milliseconds
    }
    
    /**
     * Returns current wall time corrected by NTP offset.
     */
    fun getCorrectedWallTime(): Long {
        return System.currentTimeMillis() + _ntpOffset.value
    }
    
    /**
     * Returns current NTP offset.
     */
    fun getNtpOffset(): Long {
        return _ntpOffset.value
    }
    
    /**
     * Returns whether the current NTP sync is considered valid.
     */
    fun isNtpSyncValid(): Boolean {
        val timeSinceLastSync = System.currentTimeMillis() - lastSyncTime
        return _syncStatus.value == SyncStatus.SUCCESS && 
               timeSinceLastSync < SYNC_INTERVAL_MS * 2 // Valid within 2x sync interval
    }
    
    /**
     * Returns sync information snapshot.
     */
    fun getSyncInfo(): SyncInfo {
        return SyncInfo(
            offset = _ntpOffset.value,
            lastSyncTime = lastSyncTime,
            accuracy = _syncAccuracy.value,
            status = _syncStatus.value
        )
    }
    
    /**
     * Converts a monotonic timestamp to an absolute UTC time.
     *
     * @param elapsedNanos Elapsed time (ns) since boot for the event
     * @param baseElapsedNanos Base elapsed time (ns) captured for the batch
     * @param baseWallMs Base wall-clock time (ms) captured for the batch (already NTP-corrected)
     */
    fun convertToAbsoluteTime(
        elapsedNanos: Long,
        baseElapsedNanos: Long,
        baseWallMs: Long
    ): Long {
        val deltaNanos = elapsedNanos - baseElapsedNanos
        val deltaMs = deltaNanos / 1_000_000
        return baseWallMs + deltaMs
    }
    
    /**
     * Cleans up resources.
     */
    fun cleanup() {
        stopPeriodicSync()
    }
}

/**
 * Sync status.
 */
enum class SyncStatus {
    IDLE,       // Idle
    SYNCING,    // Syncing
    SUCCESS,    // Success
    ERROR       // Error
}

/**
 * Sync information.
 */
data class SyncInfo(
    val offset: Long,        // NTP offset (ms)
    val lastSyncTime: Long,  // Last sync time (ms)
    val accuracy: Long,      // Sync accuracy (ms)
    val status: SyncStatus   // Sync status
)
