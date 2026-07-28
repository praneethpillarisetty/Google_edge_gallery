package com.google.ai.edge.gallery.customtasks.flux

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FluxModelManifestTest {
  @Test fun parsesManifest() {
    val value = """{"schemaVersion":1,"repository":"$FLUX_REPOSITORY","revision":"main","metadataPolicy":"remote","files":["model.tflite"]}"""
    assertEquals(listOf("model.tflite"), FluxModelManifest.parse(value).files)
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
    val root = createTempDir()
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
