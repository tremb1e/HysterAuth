package com.continuousauth.data

import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.provider.Settings
import com.continuousauth.model.SensorSample
import com.continuousauth.proto.DataPacket
import com.continuousauth.proto.Metadata
import com.continuousauth.proto.SerializedSensorBatch
import com.continuousauth.time.EnhancedTimeSync
import dagger.hilt.android.qualifiers.ApplicationContext
import android.util.Log
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * DataPacket builder.
 *
 * Creates packets with batch timestamps and metadata.
 */
@Singleton
class DataPacketBuilder @Inject constructor(
    @ApplicationContext private val context: Context,
    private val enhancedTimeSync: EnhancedTimeSync,
    private val envelopeCryptoBox: com.continuousauth.crypto.EnvelopeCryptoBox
) {
    
    companion object {
        private const val SCHEMA_VERSION = "1.0" // proto uses string type
        private const val TAG = "DataPacketBuilder"
    }
    
    private val deviceId by lazy { generateDeviceId() }
    private var packetSeqNo = 0L
    
    /**
     * Build a [DataPacket].
     *
     * Records batch timestamps: base_wall_ms and device_uptime_ns.
     */
    fun buildDataPacket(
        sensorSamples: List<SensorSample>,
        encryptedPayload: ByteArray,
        transmissionProfile: String = "UNRESTRICTED",
        userId: String,
        sessionId: String,
        encryptedDek: ByteArray? = null,
        dekKeyId: String = "",
        sha256: ByteArray? = null,
        compressionType: String = "gzip"
    ): DataPacket {
        
        // Capture key timestamps at batch creation.
        val baseElapsedNs = SystemClock.elapsedRealtimeNanos()
        val baseWallMs = enhancedTimeSync.getCorrectedWallTime()
        val ntpOffsetMs = enhancedTimeSync.getNtpOffset()
        
        // Generate a unique packet ID.
        val packetId = UUID.randomUUID().toString()
        
        // Build metadata.
        val metadata = buildMetadata(transmissionProfile, compressionType)
        
        // Get HMAC hash of the device ID.
        val deviceIdHash = envelopeCryptoBox.getDeviceIdHash()
        
        // Get the next packet sequence number.
        packetSeqNo = envelopeCryptoBox.getNextPacketSeqNo()
        
        // Build packet.
        val builder = DataPacket.newBuilder()
            .setPacketId(packetId)
            .setDeviceIdHash(deviceIdHash) // HMAC hash
            .setBaseWallMs(baseWallMs)
            .setDeviceUptimeNs(baseElapsedNs)
            .setEncryptedSensorPayload(com.google.protobuf.ByteString.copyFrom(encryptedPayload))
            .setPacketSeqNo(packetSeqNo) // sequence number
            .setMetadata(metadata)
        
        // Envelope encryption fields.
        if (encryptedDek != null) {
            builder.setEncryptedDek(com.google.protobuf.ByteString.copyFrom(encryptedDek))
        }
        if (dekKeyId.isNotEmpty()) {
            builder.setDekKeyId(dekKeyId)
        }
        if (sha256 != null) {
            builder.setSha256(com.google.protobuf.ByteString.copyFrom(sha256))
        }
        
        // If NTP sync is valid, add the offset.
        if (enhancedTimeSync.isNtpSyncValid()) {
            builder.setNtpOffsetMs(ntpOffsetMs)
        }
        
        android.util.Log.d(
            TAG,
            "Built DataPacket - id: $packetId, samples: ${sensorSamples.size}, " +
                "base_wall_ms: $baseWallMs, device_uptime_ns: $baseElapsedNs, " +
                "ntp_offset_ms: $ntpOffsetMs"
        )
        
        return builder.build()
    }
    
    /**
     * Build plaintext sensor batch (pre-encryption).
     */
    fun buildSensorBatch(
        sensorSamples: List<SensorSample>,
        userId: String,
        sessionId: String
    ): SerializedSensorBatch {
        
        // Convert model samples to proto format.
        val protoSamples = sensorSamples.map { sample ->
            val builder = com.continuousauth.proto.SensorSample.newBuilder()
                .setType(convertSensorType(sample.type))
                .setEventTimestampNs(sample.eventTimestampNs)
                .setX(sample.x)
                .setY(sample.y)
                .setZ(sample.z)
                .setAccuracy(sample.accuracy)
                .setSeqNo(sample.seqNo)
            
            // HMAC hash the foreground app package name.
            if (sample.foregroundApp.isNotEmpty()) {
                val appHash = envelopeCryptoBox.getAppPackageHash(sample.foregroundApp)
                builder.setForegroundAppHash(appHash)
            }
            
            builder.build()
        }
        
        // HMAC hash the user ID.
        val userIdHash = envelopeCryptoBox.getUserIdHash(userId)
        
        return SerializedSensorBatch.newBuilder()
            .addAllSamples(protoSamples)
            .setUserIdHash(userIdHash) // HMAC hash
            .setSessionId(sessionId)
            .build()
    }
    
    /**
     * Build metadata.
     */
    private fun buildMetadata(transmissionProfile: String, compressionType: String = "gzip"): Metadata {
        return Metadata.newBuilder()
            .setAppVersion(getAppVersion())
            .setAndroidApiLevel(Build.VERSION.SDK_INT) // proto type is int32
            .setSchemaVersion(SCHEMA_VERSION)
            .setTransmissionProfile(transmissionProfile)
            .setCompression(compressionType) // compression type
            .setEncryptionScheme("Envelope-AES256GCM") // envelope encryption scheme
            .setKeyVersion(envelopeCryptoBox.getDekKeyId()) // key version
            .build()
    }
    
    /**
     * Convert sensor type.
     */
    private fun convertSensorType(sensorType: com.continuousauth.model.SensorType): com.continuousauth.proto.SensorType {
        return when (sensorType) {
            com.continuousauth.model.SensorType.ACCELEROMETER -> com.continuousauth.proto.SensorType.ACCELEROMETER
            com.continuousauth.model.SensorType.GYROSCOPE -> com.continuousauth.proto.SensorType.GYROSCOPE
            com.continuousauth.model.SensorType.MAGNETOMETER -> com.continuousauth.proto.SensorType.MAGNETOMETER
        }
    }
    
    /**
     * Generate device ID.
     */
    private fun generateDeviceId(): String {
        return try {
            Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID)
                ?: UUID.randomUUID().toString()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get ANDROID_ID; using random UUID", e)
            UUID.randomUUID().toString()
        }
    }
    
    /**
     * Get app version.
     */
    private fun getAppVersion(): String {
        return try {
            val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
            "${packageInfo.versionName} (${packageInfo.longVersionCode})"
        } catch (e: Exception) {
            "unknown"
        }
    }
}
