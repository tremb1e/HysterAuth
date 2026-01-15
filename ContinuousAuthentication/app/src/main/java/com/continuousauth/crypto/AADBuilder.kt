package com.continuousauth.crypto

import android.content.Context
import android.os.BatteryManager
import android.os.Build
import com.google.protobuf.ByteString
import dagger.hilt.android.qualifiers.ApplicationContext
import java.nio.charset.StandardCharsets
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Builds AAD (associated data) for payload encryption.
 *
 * Serializes non-secret metadata (packet id, device hash, device model, app version, battery level,
 * network type, etc.) into AAD to strengthen tamper detection.
 */
@Singleton  
class AADBuilder @Inject constructor(
    @ApplicationContext private val context: Context,
    private val envelopeCryptoBox: EnvelopeCryptoBox
) {
    
    /**
     * Build associated data (AAD).
     *
     * Uses hashed identifiers instead of plaintext sensitive values and keeps a deterministic field order.
     */
    fun buildAAD(
        packetId: String,
        packetSeqNo: Long,
        dekKeyId: String,
        transmissionProfile: String = "UNRESTRICTED",
        appVersion: String,
        sampleCount: Int,
        keyVersion: String = "v1"
    ): ByteArray {
        
        return try {
            // Use hashed device id instead of plaintext.
            val deviceIdHash = envelopeCryptoBox.getDeviceIdHash()
            
            // Build AAD in a deterministic order.
            val aadData = AADData.newBuilder()
                // Core fields (fixed order)
                .setPacketId(packetId)              // 1. packet_id
                .setDeviceIdHash(deviceIdHash)      // 2. device_id_hash (HMAC)
                .setPacketSeqNo(packetSeqNo)       // 3. packet_seq_no
                .setDekKeyId(dekKeyId)              // 4. dek_key_id
                .setKeyVersion(keyVersion)          // 5. key_version
                // Additional metadata fields
                .setDeviceModel(Build.MODEL)
                .setDeviceManufacturer(Build.MANUFACTURER)
                .setAppVersion(appVersion)
                .setAndroidApiLevel(Build.VERSION.SDK_INT)
                .setTransmissionMode(transmissionProfile)
                .setSampleCount(sampleCount)
                .setBatteryLevel(getBatteryLevel())
                .setNetworkType(getNetworkType())
                .setTimestamp(System.currentTimeMillis())
                .build()
                
            aadData.toByteArray()
            
        } catch (e: Exception) {
            android.util.Log.e("AADBuilder", "Failed to build detailed AAD; falling back to basic AAD", e)
            // Fall back to basic AAD (still keeps core field order).
            buildBasicAAD(packetId, deviceIdHash = envelopeCryptoBox.getDeviceIdHash(), 
                         packetSeqNo = packetSeqNo, dekKeyId = dekKeyId, keyVersion = keyVersion)
        }
    }
    
    /**
     * Build basic AAD (used when detailed AAD construction fails).
     *
     * Keeps core field order.
     */
    private fun buildBasicAAD(
        packetId: String,
        deviceIdHash: String,
        packetSeqNo: Long,
        dekKeyId: String,
        keyVersion: String
    ): ByteArray {
        // Concatenate core fields in a fixed order.
        val basicData = StringBuilder()
            .append(packetId).append("|")           // 1. packet_id
            .append(deviceIdHash).append("|")       // 2. device_id_hash
            .append(packetSeqNo).append("|")        // 3. packet_seq_no
            .append(dekKeyId).append("|")           // 4. dek_key_id
            .append(keyVersion)                     // 5. key_version
            .toString()
        
        return basicData.toByteArray(StandardCharsets.UTF_8)
    }
    
    /**
     * Get battery level.
     */
    private fun getBatteryLevel(): Int {
        return try {
            val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager
            batteryManager?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) ?: -1
        } catch (e: Exception) {
            android.util.Log.w("AADBuilder", "Failed to get battery level", e)
            -1
        }
    }
    
    /**
     * Get network transport type.
     */
    private fun getNetworkType(): String {
        return try {
            val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) 
                as? android.net.ConnectivityManager
            
            val activeNetwork = connectivityManager?.activeNetwork
            val capabilities = connectivityManager?.getNetworkCapabilities(activeNetwork)
            
            when {
                capabilities?.hasTransport(android.net.NetworkCapabilities.TRANSPORT_WIFI) == true -> "WIFI"
                capabilities?.hasTransport(android.net.NetworkCapabilities.TRANSPORT_CELLULAR) == true -> "CELLULAR"
                capabilities?.hasTransport(android.net.NetworkCapabilities.TRANSPORT_ETHERNET) == true -> "ETHERNET"
                else -> "UNKNOWN"
            }
        } catch (e: Exception) {
            android.util.Log.w("AADBuilder", "Failed to get network type", e)
            "UNKNOWN"
        }
    }
}

/**
 * Simple serialization structure for AAD data.
 *
 * Uses a key/value format instead of Protobuf to avoid cyclic dependencies.
 */
private class AADData private constructor(
    private val data: MutableMap<String, Any> = mutableMapOf()
) {
    
    companion object {
        fun newBuilder() = Builder()
    }
    
    class Builder {
        private val data = mutableMapOf<String, Any>()
        
        // Core fields (fixed order)
        fun setPacketId(value: String) = apply { data["01_packet_id"] = value }
        fun setDeviceIdHash(value: String) = apply { data["02_device_id_hash"] = value }
        fun setPacketSeqNo(value: Long) = apply { data["03_packet_seq_no"] = value }
        fun setDekKeyId(value: String) = apply { data["04_dek_key_id"] = value }
        fun setKeyVersion(value: String) = apply { data["05_key_version"] = value }
        
        // Additional metadata fields
        fun setDeviceModel(value: String) = apply { data["device_model"] = value }
        fun setDeviceManufacturer(value: String) = apply { data["device_manufacturer"] = value }
        fun setAppVersion(value: String) = apply { data["app_version"] = value }
        fun setAndroidApiLevel(value: Int) = apply { data["android_api_level"] = value }
        fun setTransmissionMode(value: String) = apply { data["transmission_mode"] = value }
        fun setSampleCount(value: Int) = apply { data["sample_count"] = value }
        fun setBatteryLevel(value: Int) = apply { data["battery_level"] = value }
        fun setNetworkType(value: String) = apply { data["network_type"] = value }
        fun setTimestamp(value: Long) = apply { data["timestamp"] = value }
        
        fun build() = AADData(data)
    }
    
    fun toByteArray(): ByteArray {
        // Simple key/value serialization.
        val sortedEntries = data.toSortedMap() // Keep ordering stable
        val serialized = sortedEntries.entries.joinToString("|") { "${it.key}=${it.value}" }
        return serialized.toByteArray(StandardCharsets.UTF_8)
    }
}
