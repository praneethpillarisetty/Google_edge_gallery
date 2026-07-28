/* Copyright 2025 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import com.google.ai.edge.gallery.customtasks.common.CustomTask
import dagger.Binds
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dagger.multibindings.IntoSet
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
abstract class FluxEditorBindings {
  @Binds @Singleton abstract fun bindRepository(repository: DefaultFluxDownloadRepository): FluxDownloadRepository
}

@Module
@InstallIn(SingletonComponent::class)
object FluxEditorTaskModule {
  @Provides @IntoSet fun provideTask(): CustomTask = FluxEditorTask()
}
