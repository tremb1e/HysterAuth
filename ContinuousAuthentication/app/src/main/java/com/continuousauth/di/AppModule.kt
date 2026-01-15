package com.continuousauth.di

import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import javax.inject.Singleton

/**
 * Application-level dependency injection module.
 */
@Module
@InstallIn(SingletonComponent::class)
object AppModule {
    
    /**
     * Provides a global coroutine scope.
     *
     * Uses a SupervisorJob so a child failure does not cancel siblings.
     */
    @Provides
    @Singleton
    fun provideApplicationCoroutineScope(): CoroutineScope {
        return CoroutineScope(SupervisorJob() + Dispatchers.IO)
    }
}
