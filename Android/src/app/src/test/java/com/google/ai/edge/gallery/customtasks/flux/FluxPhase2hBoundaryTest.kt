/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux
import java.io.File
import kotlin.test.*
import org.junit.Test
class FluxPhase2hBoundaryTest {
 @Test fun `debug exposes decoder verification release does not and Generate is production-enabled`() {val debug=File("src/debug/java/com/google/ai/edge/gallery/customtasks/flux/FluxDeveloperVerificationSection.kt").readText();val release=File("src/release/java/com/google/ai/edge/gallery/customtasks/flux/FluxDeveloperVerificationSection.kt").readText();assertContains(debug,"Verify VAE Decoder and Image");assertFalse(release.contains("Verify VAE Decoder and Image"));val screen=File("src/main/java/com/google/ai/edge/gallery/customtasks/flux/FluxEditorScreen.kt").readText();assertContains(screen,"viewModel.canGenerate")}
 @Test fun `UI state contains no final latent or decoder float arrays`() {val source=File("src/debug/java/com/google/ai/edge/gallery/customtasks/flux/FluxTransformerDenoisingVerification.kt").readText();val state=source.substringAfter("data class FluxTransformerVerificationState(").substringBefore("data class FluxTransformerVerificationResult");assertFalse(state.contains("FloatArray"));assertFalse(state.contains("Uri"));assertFalse(state.contains("prompt"))}
 @Test fun `decoder adds no forbidden fallback`() {val source=File("src/main/java/com/google/ai/edge/gallery/customtasks/flux/decoder/FluxVaeDecoder.kt").readText();for(word in listOf("Accelerator.CPU","FP16","cloud fallback","Accelerator.NPU","Accelerator.TPU"))assertFalse(source.contains(word))}
}
