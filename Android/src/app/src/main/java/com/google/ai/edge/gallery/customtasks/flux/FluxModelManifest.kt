/* Copyright 2025 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import java.io.File
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class FluxModelManifest(
  val schemaVersion: Int,
  val repository: String,
  val revision: String,
  val metadataPolicy: String,
  val files: List<String>,
) {
  init {
    require(schemaVersion == 1) { "Unsupported FLUX manifest schema" }
    require(repository == FLUX_REPOSITORY) { "Unexpected model repository" }
    require(files.isNotEmpty() && files.distinct().size == files.size) {
      "The manifest must contain unique files"
    }
    files.forEach(::requireSafeRelativePath)
  }

  companion object {
    val json = Json { ignoreUnknownKeys = false }
    fun parse(value: String): FluxModelManifest = json.decodeFromString(value)
  }
}

/** Keeps backend-specific artifact inventories separate without inventing future TPU metadata. */
enum class FluxArtifactTarget { GPU, TENSOR_TPU }

data class FluxArtifactSet(
  val target: FluxArtifactTarget,
  val files: List<String>,
) {
  init {
    require(files.isNotEmpty() && files.distinct().size == files.size)
    files.forEach(::requireSafeRelativePath)
  }
}

fun FluxModelManifest.asGpuArtifactSet(): FluxArtifactSet =
  FluxArtifactSet(FluxArtifactTarget.GPU, files)

const val FLUX_REPOSITORY = "litert-community/FLUX.2-klein-4B-LiteRT"
const val FLUX_MODEL_DIRECTORY = "flux_2_klein_4b_litert"

fun requireSafeRelativePath(path: String): String {
  require(path.isNotBlank()) { "An empty filename is not allowed" }
  require(!File(path).isAbsolute && !path.startsWith('/') && !path.startsWith('\\')) {
    "Absolute paths are not allowed"
  }
  require(path.split('/', '\\').none { it.isEmpty() || it == "." || it == ".." }) {
    "Unsafe model filename: $path"
  }
  require(path.all { it.isLetterOrDigit() || it in "._-/" }) { "Unexpected filename: $path" }
  return path
}

fun requiredStorageBytes(missingBytes: Long, partialBytes: Long, safetyMarginBytes: Long): Long {
  require(missingBytes >= 0 && partialBytes >= 0 && safetyMarginBytes >= 0)
  // A partial file is the temporary file; only the remaining bytes plus the margin are needed.
  return (missingBytes - partialBytes).coerceAtLeast(0) + safetyMarginBytes
}
