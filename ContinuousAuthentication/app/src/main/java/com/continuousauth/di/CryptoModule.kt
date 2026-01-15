package com.continuousauth.di

import com.continuousauth.crypto.CryptoBox
import com.continuousauth.crypto.EnvelopeCryptoBox
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Crypto module DI configuration.
 *
 * Uses {@link EnvelopeCryptoBox} for envelope encryption.
 */
@Module
@InstallIn(SingletonComponent::class)
abstract class CryptoModule {
    
    @Binds
    @Singleton
    abstract fun bindCryptoBox(
        envelopeCryptoBox: EnvelopeCryptoBox
    ): CryptoBox
}
