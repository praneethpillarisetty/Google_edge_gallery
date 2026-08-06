/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import android.net.Uri
import androidx.compose.runtime.Composable
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxFigureEditRequest
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxFigurePreset

/** Release source-set deliberately renders no entry point or navigation action. */
@Composable internal fun FluxDeveloperVerificationSection(modelReady: Boolean, referenceUri: Uri?, editorPrompt: String, figureRequest: FluxFigureEditRequest?, figurePreset: FluxFigurePreset?) = Unit
