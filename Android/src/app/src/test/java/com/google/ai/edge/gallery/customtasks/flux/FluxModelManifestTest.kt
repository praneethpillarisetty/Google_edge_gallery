package com.google.ai.edge.gallery.customtasks.flux

import java.io.File
import kotlin.io.path.createTempDirectory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FluxModelManifestTest {
  private val authoritativeFiles =
    listOf(
      "tokenizer/qwen_vocab.txt",
      "tokenizer/qwen_merges.txt",
      "tokenizer/qwen_special.txt",
      "tokenizer/qwen_embed_fp16.bin",
      "ke_enc0.tflite",
      "ke_enc1.tflite",
      "ke_enc2.tflite",
      "kce_prep.tflite",
      "kce_double0.tflite",
      "kce_double1.tflite",
      "kce_single0.tflite",
      "kce_single1.tflite",
      "kce_single2.tflite",
      "kce_single3.tflite",
      "kce_final.tflite",
      "kv_vae_enc.tflite",
      "kv_vae.tflite",
    )

  @Test fun parsesManifest() {
    val value = """{"schemaVersion":1,"repository":"$FLUX_REPOSITORY","revision":"main","metadataPolicy":"remote","files":["model.tflite"]}"""
    assertEquals(listOf("model.tflite"), FluxModelManifest.parse(value).files)
  }

  @Test fun bundledManifestContainsCompleteAuthoritativeFileSet() {
    val manifestFile = File("src/main/assets/flux/manifest.json")
    val manifest = FluxModelManifest.parse(manifestFile.readText())

    assertEquals(FLUX_REPOSITORY, manifest.repository)
    assertEquals("main", manifest.revision)
    assertEquals(authoritativeFiles, manifest.files)
    assertEquals(authoritativeFiles.size, manifest.files.toSet().size)
    assertEquals(FluxArtifactTarget.GPU, manifest.asGpuArtifactSet().target)
    assertEquals(authoritativeFiles, manifest.asGpuArtifactSet().files)
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsTraversal() { requireSafeRelativePath("../secret") }

  @Test fun calculatesStorageWithoutDoubleCountingPartialFile() {
    assertEquals(1_400, requiredStorageBytes(missingBytes = 1_000, partialBytes = 100, safetyMarginBytes = 500))
  }

  @Test fun readinessRequiresEveryValidFile() {
    val validity = listOf(true, true, false)
    assertFalse(validity.all { it })
    assertTrue(validity.take(2).all { it })
  }

  @Test fun partialRecoveryUsesSafeSibling() {
    val root = createTempDirectory("flux-manifest-test").toFile()
    val target = DefaultFluxDownloadRepository.safeChild(root, "nested/model.tflite")
    val partial = File(target.path + ".partial")
    partial.parentFile!!.mkdirs()
    partial.writeBytes(ByteArray(19))
    assertEquals(19, partial.length())
    assertTrue(partial.canonicalPath.startsWith(root.canonicalPath + File.separator))
    root.deleteRecursively()
  }

  @Test fun viewModelStateTransitionsAreExplicit() {
    assertTrue(reduceFluxState(FluxDownloadEvent.Checking, 0) is FluxEditorUiState.Checking)
    assertEquals(
      FluxEditorUiState.NotInstalled(42),
      reduceFluxState(FluxDownloadEvent.NotInstalled(42), 10),
    )
    assertEquals(
      FluxEditorUiState.NotInstalled(10),
      reduceFluxState(FluxDownloadEvent.NotInstalled(null), 10),
    )
    assertTrue(
      reduceFluxState(FluxDownloadEvent.Downloading(emptyList()), 10) is
        FluxEditorUiState.Downloading
    )
    assertTrue(reduceFluxState(FluxDownloadEvent.Paused, 10) is FluxEditorUiState.Paused)
    assertEquals(FluxEditorUiState.Ready(10), reduceFluxState(FluxDownloadEvent.Ready(10), 10))
    assertTrue(
      reduceFluxState(FluxDownloadEvent.Error("failure"), 10) is FluxEditorUiState.Error
    )
  }
}
