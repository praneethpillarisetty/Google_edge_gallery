/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.prompt

import com.google.ai.edge.gallery.customtasks.flux.FLUX_MODEL_REVISION
import com.google.ai.edge.gallery.customtasks.flux.FluxModelManifest
import java.io.File

data class FluxPromptAssets(
  val vocabulary: File,
  val merges: File,
  val specials: File,
  val embeddings: File,
  val encoderGraphs: List<File>,
)

/** Resolves assets only beneath the canonical app-owned model directory and pinned manifest. */
class FluxPromptAssetResolver(private val modelDirectory: File, private val manifest: FluxModelManifest) {
  fun resolve(): FluxPromptAssets {
    if (manifest.revision != FLUX_MODEL_REVISION) throw FluxPromptPreparationException("FLUX model revision is not pinned.")
    val root = try { modelDirectory.canonicalFile } catch (e: Exception) {
      throw FluxPromptPreparationException("Unable to canonicalize the FLUX model directory.", e)
    }
    if (!root.isDirectory) throw FluxPromptPreparationException("FLUX model directory is missing.")
    fun required(path: String): File {
      if (path !in manifest.files) throw FluxPromptPreparationException("Required prompt asset is absent from the manifest: $path")
      val file = try { File(root, path).canonicalFile } catch (e: Exception) {
        throw FluxPromptPreparationException("Unable to canonicalize prompt asset $path.", e)
      }
      if (!file.path.startsWith(root.path + File.separator) || !file.isFile) {
        throw FluxPromptPreparationException("Required prompt asset is missing or unsafe: $path")
      }
      return file
    }
    return FluxPromptAssets(
      required("tokenizer/qwen_vocab.txt"), required("tokenizer/qwen_merges.txt"),
      required("tokenizer/qwen_special.txt"), required("tokenizer/qwen_embed_fp16.bin"),
      FluxTextEncoderContracts.GRAPH_NAMES.map(::required),
    )
  }
}
