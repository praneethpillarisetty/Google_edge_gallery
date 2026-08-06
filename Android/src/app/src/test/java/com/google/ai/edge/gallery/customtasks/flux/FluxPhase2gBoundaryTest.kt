/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import java.io.File
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import org.junit.Test

class FluxPhase2gBoundaryTest {
  private val project = File(".")
  private fun source(path: String) = project.resolve(path).readText()

  @Test fun `debug exposes controller while release remains a no op`() {
    val debug = source("src/debug/java/com/google/ai/edge/gallery/customtasks/flux/FluxDeveloperVerificationSection.kt")
    val release = source("src/release/java/com/google/ai/edge/gallery/customtasks/flux/FluxDeveloperVerificationSection.kt")
    assertTrue(debug.contains("Developer verification — VAE decoder and image"))
    assertTrue(debug.contains("FluxTransformerDenoisingVerificationViewModel"))
    assertFalse(release.contains("Transformer denoising"))
    assertTrue(release.contains("= Unit"))
  }

  @Test fun `phase 2g retains no tensor arrays in UI state or result`() {
    val controller = source("src/debug/java/com/google/ai/edge/gallery/customtasks/flux/FluxTransformerDenoisingVerification.kt")
    val state = controller.substring(controller.indexOf("data class FluxTransformerVerificationState"), controller.indexOf("data class FluxTransformerVerificationResult"))
    val result = controller.substring(controller.indexOf("data class FluxTransformerVerificationResult"), controller.indexOf("@HiltViewModel"))
    assertFalse(state.contains("FloatArray")); assertFalse(result.contains("FloatArray"))
  }

  @Test fun `decoder composes phase 2g without fallbacks and Generate is production-enabled`() {
    val denoiser = source("src/main/java/com/google/ai/edge/gallery/customtasks/flux/transformer/FluxTransformerDenoiser.kt")
    val controller = source("src/debug/java/com/google/ai/edge/gallery/customtasks/flux/FluxTransformerDenoisingVerification.kt")
    val editor = source("src/main/java/com/google/ai/edge/gallery/customtasks/flux/FluxEditorScreen.kt")
    val phase = denoiser + controller
    assertTrue(phase.contains("kv_vae.tflite"))
    assertFalse(phase.contains("Accelerator.CPU")); assertFalse(phase.contains("FP16"))
    assertFalse(phase.contains("cloud", ignoreCase = true)); assertFalse(phase.contains("Accelerator.NPU"))
    assertTrue(editor.contains("viewModel.compileSimpleAndGenerate(imageUri"))
  }
}
