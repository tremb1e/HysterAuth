package com.continuousauth.pool

import com.continuousauth.model.SensorSample
import com.continuousauth.model.SensorType
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Sensor event object pool.
 *
 * Reduces allocations and GC pressure during high-frequency sampling.
 */
@Singleton
class SensorEventPool @Inject constructor() {
    
    companion object {
        private const val DEFAULT_POOL_SIZE = 1000
        private const val MAX_POOL_SIZE = 5000
        private const val TAG = "SensorEventPool"
    }
    
    // Pool queue
    private val pool = ConcurrentLinkedQueue<PooledSensorSample>()
    private val currentPoolSize = AtomicInteger(0)
    
    // Stats
    private val totalAcquired = AtomicLong(0L)
    private val totalReleased = AtomicLong(0L)
    private val totalCreated = AtomicLong(0L)
    
    init {
        // Pre-fill the pool.
        prefillPool()
    }
    
    /**
     * Acquires a pooled sensor sample wrapper from the pool.
     */
    fun acquire(): PooledSensorSample {
        val pooledSample = pool.poll()
        
        return if (pooledSample != null) {
            currentPoolSize.decrementAndGet()
            totalAcquired.incrementAndGet()
            pooledSample.reset() // Reset state
            pooledSample
        } else {
            // No available object in the pool; create a new one.
            totalCreated.incrementAndGet()
            totalAcquired.incrementAndGet()
            android.util.Log.v(TAG, "No available object in pool; creating a new one")
            PooledSensorSample(this)
        }
    }
    
    /**
     * Releases an object back to the pool.
     */
    fun release(pooledSample: PooledSensorSample) {
        if (currentPoolSize.get() < MAX_POOL_SIZE) {
            pooledSample.reset()
            pool.offer(pooledSample)
            currentPoolSize.incrementAndGet()
            totalReleased.incrementAndGet()
        } else {
            // Pool is full; let the object be collected by GC.
            totalReleased.incrementAndGet()
            android.util.Log.v(TAG, "Pool is full; object will be collected by GC")
        }
    }
    
    /**
     * Returns current pool status.
     */
    fun getPoolStatus(): PoolStatus {
        return PoolStatus(
            currentSize = currentPoolSize.get(),
            maxSize = MAX_POOL_SIZE,
            totalAcquired = totalAcquired.get(),
            totalReleased = totalReleased.get(),
            totalCreated = totalCreated.get(),
            hitRate = if (totalAcquired.get() > 0) {
                (totalAcquired.get() - totalCreated.get()).toDouble() / totalAcquired.get()
            } else 0.0
        )
    }
    
    /**
     * Pre-fills the pool with default capacity.
     */
    private fun prefillPool() {
        repeat(DEFAULT_POOL_SIZE) {
            pool.offer(PooledSensorSample(this))
            currentPoolSize.incrementAndGet()
        }
        android.util.Log.i(TAG, "Pool prefill complete - initial size: $DEFAULT_POOL_SIZE")
    }
}

/**
 * Pooled sensor sample wrapper.
 */
class PooledSensorSample(private val pool: SensorEventPool) {
    
    private var type: SensorType = SensorType.ACCELEROMETER
    private var eventTimestampNs: Long = 0L
    private var x: Float = 0f
    private var y: Float = 0f
    private var z: Float = 0f
    private var accuracy: Int = 0
    private var seqNo: Long = 0L
    private var foregroundApp: String = ""
    
    /**
     * Sets sensor data.
     */
    fun setSensorData(
        type: SensorType,
        eventTimestampNs: Long,
        x: Float,
        y: Float,
        z: Float,
        accuracy: Int,
        seqNo: Long,
        foregroundApp: String
    ) {
        this.type = type
        this.eventTimestampNs = eventTimestampNs
        this.x = x
        this.y = y
        this.z = z
        this.accuracy = accuracy
        this.seqNo = seqNo
        this.foregroundApp = foregroundApp
    }
    
    /**
     * Converts to {@link SensorSample}.
     */
    fun toSensorSample(): SensorSample {
        return SensorSample(
            type = type,
            eventTimestampNs = eventTimestampNs,
            x = x,
            y = y,
            z = z,
            accuracy = accuracy,
            seqNo = seqNo,
            foregroundApp = foregroundApp
        )
    }
    
    /**
     * Resets object state.
     */
    fun reset() {
        type = SensorType.ACCELEROMETER
        eventTimestampNs = 0L
        x = 0f
        y = 0f
        z = 0f
        accuracy = 0
        seqNo = 0L
        foregroundApp = ""
    }
    
    /**
     * Releases the object back to the pool.
     */
    fun release() {
        pool.release(this)
    }
}

/**
 * Object pool status.
 */
data class PoolStatus(
    val currentSize: Int,       // Current pool size
    val maxSize: Int,          // Max pool size
    val totalAcquired: Long,   // Total acquire operations
    val totalReleased: Long,   // Total release operations
    val totalCreated: Long,    // Total objects created
    val hitRate: Double        // Hit rate (acquired from pool / total acquired)
)
