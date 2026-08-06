/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class FluxPhase2fBoundaryTest {
  @Test fun debugUiInvokesViewModelRatherThanAssigningAPlaceholderString() {
    val ui = source("app/src/debug/java/com/google/ai/edge/gallery/customtasks/flux/FluxDeveloperVerificationSection.kt")
    assertContains(ui, "Developer verification — Editing transformer prep only")
    assertContains(ui, "viewModel.run(it, prompt, modelReady)")
    assertContains(ui, "onClick = viewModel::cancel")
    assertFalse(ui.contains("var diagnostic by remember"))
    assertFalse(ui.contains("onClick = { diagnostic ="))
  }

  @Test fun orchestrationNamesOnlyTheRequiredSequentialGraphBoundaries() {
    val verification = source("app/src/debug/java/com/google/ai/edge/gallery/customtasks/flux/FluxTransformerPrepVerification.kt")
    assertContains(verification, "FluxReferenceVaeEncoder(graphRunner)")
    assertContains(verification, "FluxPromptConditioner.create")
    assertContains(verification, "FluxTransformerPrepRunner(graphRunner)")
    assertContains(verification, "prep.run(prepGraph, editing, conditioning, timestep)")
    for (forbidden in listOf("kce_double0", "kce_double1", "kce_single0", "kce_single1", "kce_single2", "kce_single3", "kce_final", "kv_vae.tflite")) {
      assertFalse(verification.contains(forbidden))
    }
  }

  @Test fun everyRequiredProgressStageAndCleanupBoundaryIsPresent() {
    val verification = source("app/src/debug/java/com/google/ai/edge/gallery/customtasks/flux/FluxTransformerPrepVerification.kt")
    for (stage in listOf(
      "PREPARING_REFERENCE", "ENCODING_REFERENCE_VAE", "BUILDING_REFERENCE_TOKENS",
      "ENCODING_PROMPT", "BUILDING_SYNTHETIC_NOISE", "ASSEMBLING_EDITING_SEQUENCE",
      "BUILDING_SYNTHETIC_TIMESTEP", "HASHING_TRANSFORMER_PREP", "COMPILING_TRANSFORMER_PREP",
      "RUNNING_TRANSFORMER_PREP", "VALIDATING_TRANSFORMER_PREP_OUTPUTS", "COMPLETE", "CANCELLED", "ERROR",
    )) assertContains(verification, stage)
    assertContains(verification, "repository.withModelFilesLocked")
    assertContains(verification, ".withStagedSource")
    assertContains(verification, "environment.close()")
    assertContains(verification, "job?.cancel()")
  }

  @Test fun releaseHasNoPrepVerificationAndGenerateRemainsDisabled() {
    val release = source("app/src/release/java/com/google/ai/edge/gallery/customtasks/flux/FluxDeveloperVerificationSection.kt")
    assertFalse(release.contains("Editing transformer prep"))
    val editor = source("app/src/main/java/com/google/ai/edge/gallery/customtasks/flux/FluxEditorScreen.kt")
    assertContains(editor, "viewModel.generate(imageUri, compiled.positivePrompt")
  }

  private fun source(relative: String) = Files.readString(projectRoot().resolve(relative))
  private fun projectRoot(): Path {
    var current = Path.of("").toAbsolutePath()
    while (current.parent != null) {
      if (Files.isDirectory(current.resolve("app/src/main"))) return current
      current = current.parent
    }
    error("Android project root not found.")
  }
}
