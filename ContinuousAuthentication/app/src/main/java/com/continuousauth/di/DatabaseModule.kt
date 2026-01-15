package com.continuousauth.di

import android.content.Context
import com.continuousauth.database.BatchMetadataDao
import com.continuousauth.database.ContinuousAuthDatabase
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Database module DI configuration.
 */
@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {
    
    /**
     * Provides database instance.
     */
    @Provides
    @Singleton
    fun provideContinuousAuthDatabase(
        @ApplicationContext context: Context
    ): ContinuousAuthDatabase {
        return ContinuousAuthDatabase.getInstance(context)
    }
    
    /**
     * Provides {@link BatchMetadataDao}.
     */
    @Provides
    @Singleton
    fun provideBatchMetadataDao(
        database: ContinuousAuthDatabase
    ): BatchMetadataDao {
        return database.batchMetadataDao()
    }
}
