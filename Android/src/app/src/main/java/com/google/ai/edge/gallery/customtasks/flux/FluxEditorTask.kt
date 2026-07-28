/* Copyright 2025 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import android.content.Context
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AutoFixHigh
import androidx.compose.runtime.Composable
import com.google.ai.edge.gallery.customtasks.common.CustomTask
import com.google.ai.edge.gallery.data.Category
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.Task
import com.google.ai.edge.litertlm.Contents
import kotlinx.coroutines.CoroutineScope

class FluxEditorTask : CustomTask {
  override val task =
    Task(
      id = "flux_image_editor",
      label = "FLUX Image Editor",
      category = Category.EXPERIMENTAL,
      icon = Icons.Outlined.AutoFixHigh,
      description = "Prepare the on-device FLUX.2 Klein LiteRT one-reference image editor.",
      shortDescription = "Edit an image on-device",
      docUrl = "https://huggingface.co/litert-community/FLUX.2-klein-4B-LiteRT",
      models =
        mutableListOf(
          Model(
            name = "FLUX.2 Klein 4B LiteRT",
            info = "LiteRT GPU export fixed to one reference image at 256×256. Phase 1 manages model files only.",
            localFileRelativeDirPathOverride = FLUX_MODEL_DIRECTORY,
            bestForTaskIds = listOf("flux_image_editor"),
          )
        ),
      experimental = true,
      newFeature = true,
      useThemeColor = true,
    )

  override fun initializeModelFn(context: Context, coroutineScope: CoroutineScope, model: Model, systemInstruction: Contents?, onDone: (String) -> Unit) = onDone("")
  override fun cleanUpModelFn(context: Context, coroutineScope: CoroutineScope, model: Model, onDone: () -> Unit) = onDone()
  @Composable override fun MainScreen(data: Any) = FluxEditorScreen()
}
