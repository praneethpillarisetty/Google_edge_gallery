/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.prompt

import com.google.ai.edge.gallery.customtasks.flux.FLUX_MODEL_REVISION
import com.google.ai.edge.gallery.customtasks.flux.FLUX_REPOSITORY
import com.google.ai.edge.gallery.customtasks.flux.FluxModelManifest
import java.nio.file.Files
import kotlin.io.path.createDirectories
import kotlin.io.path.createFile
import kotlin.io.path.createTempDirectory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class FluxPromptAssetResolverTest {
  private val names = listOf("tokenizer/qwen_vocab.txt", "tokenizer/qwen_merges.txt", "tokenizer/qwen_special.txt", "tokenizer/qwen_embed_fp16.bin") + FluxTextEncoderContracts.GRAPH_NAMES
  private fun manifest() = FluxModelManifest(1, FLUX_REPOSITORY, FLUX_MODEL_REVISION, "pinned", names)

  @Test fun resolvesOnlyCompleteCanonicalAssetSet() {
    val root = createTempDirectory("flux-assets")
    names.forEach { root.resolve(it).parent.createDirectories(); root.resolve(it).createFile() }
    assertEquals(3, FluxPromptAssetResolver(root.toFile(), manifest()).resolve().encoderGraphs.size)
  }

  @Test fun rejectsMissingTokenizerEmbeddingAndGraph() {
    for (missing in listOf(names[0], names[3], names.last())) {
      val root = createTempDirectory("flux-missing")
      names.filterNot { it == missing }.forEach { root.resolve(it).parent.createDirectories(); root.resolve(it).createFile() }
      assertThrows(FluxPromptPreparationException::class.java) { FluxPromptAssetResolver(root.toFile(), manifest()).resolve() }
    }
  }

  @Test fun rejectsSymlinkEscapingCanonicalRoot() {
    val root = createTempDirectory("flux-unsafe")
    names.forEach { root.resolve(it).parent.createDirectories(); root.resolve(it).createFile() }
    val outside = Files.createTempFile("outside", ".bin")
    Files.delete(root.resolve(names.last()))
    Files.createSymbolicLink(root.resolve(names.last()), outside)
    assertThrows(FluxPromptPreparationException::class.java) { FluxPromptAssetResolver(root.toFile(), manifest()).resolve() }
  }
}
