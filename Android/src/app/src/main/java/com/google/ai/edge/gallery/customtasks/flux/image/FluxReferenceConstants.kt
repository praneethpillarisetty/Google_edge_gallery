/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.image

import android.content.res.AssetManager
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

class FluxReferenceTokenException(message: String, cause: Throwable? = null) :
  IllegalArgumentException(message, cause)

internal data class FluxReferenceConstants(
  val mean: FloatArray,
  val standardDeviation: FloatArray,
  val patchPermutation: IntArray,
  val revision: String,
)

/** Platform-neutral strict parser for the committed Phase 2E evidence bundle. */
internal object FluxReferenceConstantsParser {
  const val MEAN_SHA256 = "9027fac5727854f779ebbeae3032cfce0d47a11bc85f4329eb0a317c9ad90217"
  const val STD_SHA256 = "e89b48bf701b864cc6cad73070e0e49052ee673d2c34ec2084ecf6386284d199"
  const val PERM_SHA256 = "90c531082fddef4309f5ba43b8c898c823f049d2aa8951fb84e9cc4395942c3d"
  const val REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
  private const val COMPANION_REVISION = "f48a89e4f29a74ab51f29c311ac7a0e5e479d225"
  private const val FORMULA =
    "(packed[channel] - running_mean[channel]) / sqrt(running_variance[channel] + batch_norm_eps)"
  private val json = Json { ignoreUnknownKeys = false }

  fun parse(open: (String) -> InputStream): FluxReferenceConstants = try {
    val metadataBytes = readExactly(open, "flux_reference_constants.json", null)
    val metadata = json.decodeFromString<Metadata>(metadataBytes.decodeToString())
    validateMetadata(metadata)
    val meanBytes = readExactly(open, "bn_mean.bin", 512)
    val stdBytes = readExactly(open, "bn_std.bin", 512)
    val permutationBytes = readExactly(open, "patch_perm.bin", 131_072)
    verifyHash("bn_mean.bin", meanBytes, MEAN_SHA256)
    verifyHash("bn_std.bin", stdBytes, STD_SHA256)
    verifyHash("patch_perm.bin", permutationBytes, PERM_SHA256)
    val mean = floats(meanBytes)
    val std = floats(stdBytes)
    val permutation = ints(permutationBytes)
    validateMean(mean)
    validateStandardDeviation(std)
    validatePermutation(permutation)
    FluxReferenceConstants(mean, std, permutation, metadata.immutableBaseModelRevision)
  } catch (known: FluxReferenceTokenException) {
    throw known
  } catch (failure: Exception) {
    throw FluxReferenceTokenException("FLUX reference constants validation failed.", failure)
  }

  internal fun validatePermutation(permutation: IntArray) {
    if (permutation.size != FluxReferenceTokenContracts.ELEMENTS) fail("patch_perm.bin must contain exactly 32,768 signed little-endian int32 values.")
    val seen = BooleanArray(FluxReferenceTokenContracts.ELEMENTS)
    for (index in permutation) {
      if (index !in seen.indices) fail("patch_perm.bin contains an out-of-range index.")
      if (seen[index]) fail("patch_perm.bin contains a duplicate index.")
      seen[index] = true
    }
    if (!seen.all { it }) fail("patch_perm.bin is missing an index.")
  }

  internal fun validateMean(mean: FloatArray) {
    if (mean.size != 128 || mean.any { !it.isFinite() })
      fail("bn_mean.bin must contain 128 finite little-endian FP32 values.")
  }

  internal fun validateStandardDeviation(standardDeviation: FloatArray) {
    if (standardDeviation.size != 128 || standardDeviation.any { !it.isFinite() || it <= 0f })
      fail("bn_std.bin must contain 128 finite, positive little-endian FP32 values.")
  }

  private fun validateMetadata(value: Metadata) {
    if (
      value.schemaVersion != 1 ||
        value.baseModelRepository != "black-forest-labs/FLUX.2-klein-4B" ||
        value.immutableBaseModelRevision != REVISION ||
        value.companionRepository != "google-ai-edge/litert-samples" ||
        value.immutableCompanionRevision != COMPANION_REVISION ||
        value.generatorScriptPath != "compiled_model_api/text_to_image/flux2_klein_kotlin_gpu/conversion/gen_prep_klein.py" ||
        value.generatorScriptSha256 != "1f2b3d902d4f36281e61447d86331e08bd1c61f20f9034896817c086ae1ab61d" ||
        value.vaeClassName != "diffusers.models.autoencoders.autoencoder_kl_flux2.AutoencoderKLFlux2" ||
        value.batchNormEpsilon != 0.0001 || value.batchNormWeightApplied || value.batchNormBiasApplied ||
        value.sourceLatentShape != listOf(1, 32, 32, 32) ||
        value.packedLatentShape != listOf(1, 128, 16, 16) ||
        value.finalTokenShape != listOf(1, 256, 128) || value.normalizationFormula != FORMULA
    ) fail("flux_reference_constants.json provenance or tensor contract does not match Phase 2E.")
    validateArtifact(value.artifacts["bn_mean.bin"], 512, 128, "float32", "<f4", MEAN_SHA256)
    validateArtifact(value.artifacts["bn_std.bin"], 512, 128, "float32", "<f4", STD_SHA256)
    validateArtifact(value.artifacts["patch_perm.bin"], 131_072, 32_768, "int32", "<i4", PERM_SHA256)
    if (value.artifacts.keys != setOf("bn_mean.bin", "bn_std.bin", "patch_perm.bin")) fail("Metadata contains an unexpected artifact set.")
  }

  private fun validateArtifact(value: Artifact?, bytes: Int, count: Int, dtype: String, numpy: String, sha: String) {
    if (value == null || value.byteCount != bytes || value.elementCount != count || value.byteOrder != "little-endian" || value.dtype != dtype || value.numpyDtype != numpy || value.sha256 != sha) fail("Artifact metadata does not match the authoritative evidence.")
  }

  private fun readExactly(open: (String) -> InputStream, name: String, size: Int?): ByteArray =
    open(name).use { input -> input.readBytes().also { if (size != null && it.size != size) fail("$name has an invalid byte count.") } }

  private fun verifyHash(name: String, bytes: ByteArray, expected: String) {
    val actual = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    if (actual != expected) fail("$name failed SHA-256 validation.")
  }
  private fun floats(bytes: ByteArray) = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).let { b -> FloatArray(bytes.size / 4) { b.float } }
  private fun ints(bytes: ByteArray) = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).let { b -> IntArray(bytes.size / 4) { b.int } }
  private fun fail(message: String): Nothing = throw FluxReferenceTokenException(message)

  @Serializable private data class Artifact(val byteCount: Int, val byteOrder: String, val dtype: String, val elementCount: Int, val numpyDtype: String, val sha256: String)
  @Serializable private data class Libraries(val accelerate: String, val diffusers: String, val huggingface_hub: String, val numpy: String, val safetensors: String, val transformers: String)
  @Serializable private data class Metadata(
    val artifacts: Map<String, Artifact>, val baseModelRepository: String, val batchNormBiasApplied: Boolean,
    val batchNormEpsilon: Double, val batchNormWeightApplied: Boolean, val companionRepository: String,
    val extractionLibraries: Libraries, val finalTokenShape: List<Int>, val generatedAt: String,
    val generatorScriptPath: String, val generatorScriptSha256: String, val immutableBaseModelRevision: String,
    val immutableCompanionRevision: String, val normalizationFormula: String, val packedLatentShape: List<Int>,
    val provenanceStatement: String, val pythonVersion: String, val pytorchVersion: String, val schemaVersion: Int,
    val sourceLatentShape: List<Int>, val vaeClassName: String,
  )
}

/** Android asset adapter with synchronized successful-result-only caching. */
internal class FluxReferenceConstantsLoader(private val assets: AssetManager) {
  @Volatile private var cached: FluxReferenceConstants? = null
  internal fun load(): FluxReferenceConstants = cached ?: synchronized(this) {
    cached ?: FluxReferenceConstantsParser.parse { assets.open("$ASSET_DIRECTORY/$it") }.also { cached = it }
  }
  companion object { const val ASSET_DIRECTORY = "flux/reference" }
}
