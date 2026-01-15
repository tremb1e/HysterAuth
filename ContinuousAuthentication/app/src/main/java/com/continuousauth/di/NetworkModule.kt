package com.continuousauth.di

import com.continuousauth.buffer.InMemoryBuffer
import com.continuousauth.buffer.InMemoryBufferImpl
import com.continuousauth.network.Uploader
import com.continuousauth.network.UploaderImpl
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Network module DI configuration.
 */
@Module
@InstallIn(SingletonComponent::class)
abstract class NetworkModule {
    
    @Binds
    @Singleton
    abstract fun bindUploader(
        uploaderImpl: UploaderImpl
    ): Uploader
    
    @Binds
    @Singleton
    abstract fun bindInMemoryBuffer(
        inMemoryBufferImpl: InMemoryBufferImpl
    ): InMemoryBuffer
}
