package com.google.ai.edge.gallery.customtasks.flux

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class FluxPhase2eBoundaryTest {
  @Test fun debugSourceContainsEveryPhase2eProgressStageAndHeading() {
    val debug = source("app/src/debug/java/com/google/ai/edge/gallery/customtasks/flux/FluxReferenceVaeVerification.kt")
    val ui = source("app/src/debug/java/com/google/ai/edge/gallery/customtasks/flux/FluxDeveloperVerificationSection.kt")
    for (stage in listOf(
      "LOADING_REFERENCE_CONSTANTS", "PATCHIFYING_REFERENCE_LATENT",
      "NORMALIZING_REFERENCE_CHANNELS", "BUILDING_REFERENCE_TOKENS",
      "VALIDATING_REFERENCE_TOKENS",
    )) assertContains(debug, stage)
    assertContains(ui, "Developer verification — Reference VAE and tokens only")
  }

  @Test fun releaseSourceHasNoVerificationEntryPointAndGenerateIsProductionEnabled() {
    val mainRoot = projectRoot().resolve("app/src/main")
    val mainKotlin = Files.walk(mainRoot).use { paths ->
      paths.filter { Files.isRegularFile(it) && it.toString().endsWith(".kt") }
        .map(Files::readString).toList().joinToString("\n")
    }
    assertFalse(mainKotlin.contains("Developer verification — Reference VAE and tokens only"))
    val editor = source("app/src/main/java/com/google/ai/edge/gallery/customtasks/flux/FluxEditorScreen.kt")
    assertContains(editor, "viewModel.generate(imageUri, prompt)")
  }

  @Test fun phase2eDoesNotNameOrInvokeDeferredGraphs() {
    val encoder = source("app/src/main/java/com/google/ai/edge/gallery/customtasks/flux/image/FluxReferenceTokenEncoder.kt")
    for (deferred in listOf("kce_prep", "kce_double", "kce_single", "kce_final", "kv_vae.tflite")) {
      assertFalse(encoder.contains(deferred))
    }
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
