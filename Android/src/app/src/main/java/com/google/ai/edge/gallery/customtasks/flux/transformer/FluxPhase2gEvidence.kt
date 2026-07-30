/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.transformer

import android.content.res.AssetManager
import java.io.InputStream
import java.security.MessageDigest
import java.time.OffsetDateTime
import java.time.ZoneOffset
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

class FluxPhase2gEvidenceException(message: String, cause: Throwable? = null) : IllegalArgumentException(message, cause)

/** Immutable validated Phase 2G tensors. Public access never exposes a backing array. */
class FluxPhase2gEvidence private constructor(
  private val latents: FloatArray,
  private val temb: FloatArray,
  private val deltas: FloatArray,
  private val rotaryCos: FloatArray,
  private val rotarySin: FloatArray,
) {
  fun initialLatents(): FloatArray = latents.copyOf()
  fun timestep(step: Int): FloatArray {
    require(step in 0 until STEPS)
    return temb.copyOfRange(step * TIMESTEP_SIZE, (step + 1) * TIMESTEP_SIZE)
  }
  fun dsigma(step: Int): Float { require(step in 0 until STEPS); return deltas[step] }
  fun cos(): FloatArray = rotaryCos.copyOf()
  fun sin(): FloatArray = rotarySin.copyOf()

  companion object {
    private const val STEPS = 4
    private const val TIMESTEP_SIZE = 3_072
    internal fun validated(latents: FloatArray, temb: FloatArray, deltas: FloatArray, cos: FloatArray, sin: FloatArray) =
      FluxPhase2gEvidence(latents, temb, deltas, cos, sin)
  }
}

data class FluxPhase2gEvidenceHashes(
  val latents0: String, val temb: String, val dsigma: String, val cos: String, val sin: String,
)

/** Strict parser. Hashing and FP32 conversion share one bounded streaming pass per artifact. */
object FluxPhase2gEvidenceParser {
  const val DIRECTORY = "flux/phase2g"
  const val BASE_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
  const val COMPANION_REVISION = "f48a89e4f29a74ab51f29c311ac7a0e5e479d225"
  const val GENERATOR_SHA = "1f2b3d902d4f36281e61447d86331e08bd1c61f20f9034896817c086ae1ab61d"
  private val json = Json { ignoreUnknownKeys = false }
  private val contracts = linkedMapOf(
    "latents0.bin" to Contract(listOf(1, 256, 128), 32_768, 131_072, "81c0e15a45448c9d8e8e449d02146c33fb7561aee039b0755cd5168e4ee2956b"),
    "temb.bin" to Contract(listOf(4, 3072), 12_288, 49_152, "c61e8934b1474620c4fe5b1cd387005f4673dae939b5e5432bb38fd8c3bc1cce"),
    "dsigma.bin" to Contract(listOf(4), 4, 16, "f69390537e24ea71a5b3c46954fc923aae52d40e049fa054e289134aa0ea7cfa"),
    "cos.bin" to Contract(listOf(1, 1024, 1, 64), 65_536, 262_144, "d45b2bb837a8e543ceae238e8dc72cbfcdbec7268fa691b722994a1f08ab68cc"),
    "sin.bin" to Contract(listOf(1, 1024, 1, 64), 65_536, 262_144, "657c868835d8622d791eef27dead19b2dad910014e6bd39343e6721efc0b13a3"),
  )

  fun parse(open: (String) -> InputStream, cancellationCheck: () -> Unit = {}): FluxPhase2gEvidence = try {
    cancellationCheck()
    val metadataBytes = readBounded(open("flux_phase2g_constants.json"), MAX_METADATA_BYTES, cancellationCheck)
    val metadata = json.decodeFromString<Metadata>(metadataBytes.decodeToString())
    validateMetadata(metadata)
    val tensors = contracts.mapValues { (name, contract) ->
      cancellationCheck()
      readTensor(open(name), name, contract, metadata.artifacts.getValue(name).sha256, cancellationCheck)
    }
    val dsigma = tensors.getValue("dsigma.bin")
    validateDsigma(dsigma)
    FluxPhase2gEvidence.validated(
      tensors.getValue("latents0.bin"), tensors.getValue("temb.bin"), dsigma,
      tensors.getValue("cos.bin"), tensors.getValue("sin.bin"),
    )
  } catch (known: FluxPhase2gEvidenceException) { throw known }
    catch (cancelled: kotlinx.coroutines.CancellationException) { throw cancelled }
    catch (failure: Exception) { throw FluxPhase2gEvidenceException("Phase 2G evidence validation failed.", failure) }

  fun expectedHashes() = FluxPhase2gEvidenceHashes(
    contracts.getValue("latents0.bin").sha, contracts.getValue("temb.bin").sha,
    contracts.getValue("dsigma.bin").sha, contracts.getValue("cos.bin").sha,
    contracts.getValue("sin.bin").sha,
  )

  internal fun validateDsigma(values: FloatArray) {
    if (values.size != 4 || values.any { !it.isFinite() || it >= 0f })
      fail("dsigma.bin must contain exactly four finite, strictly negative values.")
  }

  private fun readTensor(input: InputStream, name: String, contract: Contract, metadataSha: String, check: () -> Unit): FloatArray = input.use {
    val digest = MessageDigest.getInstance("SHA-256")
    val values = FloatArray(contract.count)
    val word = ByteArray(4)
    var count = 0
    while (count < contract.count) {
      check()
      var used = 0
      while (used < 4) {
        val read = it.read(word, used, 4 - used)
        if (read < 0) fail("$name is truncated or contains a partial float.")
        used += read
      }
      digest.update(word)
      val bits = (word[0].toInt() and 0xff) or ((word[1].toInt() and 0xff) shl 8) or
        ((word[2].toInt() and 0xff) shl 16) or ((word[3].toInt() and 0xff) shl 24)
      val value = Float.fromBits(bits)
      if (!value.isFinite()) fail("$name contains a non-finite value.")
      values[count++] = value
    }
    if (it.read() != -1) fail("$name contains trailing bytes.")
    val actual = digest.digest().toHex()
    if (actual != metadataSha || actual != contract.sha) fail("$name failed SHA-256 validation.")
    values
  }

  private fun validateMetadata(m: Metadata) {
    val generated = try { OffsetDateTime.parse(m.generatedAt) } catch (_: Exception) { fail("generatedAt is not a valid timestamp.") }
    if (generated.offset != ZoneOffset.UTC) fail("generatedAt must be a UTC timestamp.")
    if (m.schemaVersion != 1 || m.purpose != "FLUX Phase 2G editing host inputs" || m.mode != "image-editing" ||
      m.seed != 1234 || m.steps != 4 || m.imageSize != listOf(256, 256) ||
      m.baseModelRepository != "black-forest-labs/FLUX.2-klein-4B" || m.immutableBaseModelRevision != BASE_REVISION ||
      m.companionRepository != "google-ai-edge/litert-samples" || m.immutableCompanionRevision != COMPANION_REVISION ||
      m.generatorScriptPath != "compiled_model_api/text_to_image/flux2_klein_kotlin_gpu/conversion/gen_prep_klein.py" ||
      m.originalGeneratorScriptSha256 != GENERATOR_SHA ||
      m.authenticationPatchedScriptSha256 != "4f9dbc4c48998da80489978752e275a99b453f4cce5a0e29174431c490fb7a94" ||
      m.extractionLibraries != Libraries("1.14.0", "0.39.0", "1.25.1", "2.0.2", "0.8.0", "5.14.1") ||
      m.provenanceStatement != "The five files were generated by the recorded pinned companion algorithm from the recorded immutable base-model revision. The local script modification supplied only HF_TOKEN and the immutable revision to from_pretrained." ||
      m.pythonVersion != "3.12.13" || m.pytorchVersion != "2.9.0+cpu" || m.artifacts.keys != contracts.keys) {
      fail("Metadata provenance does not match the authoritative Phase 2G contract.")
    }
    contracts.forEach { (name, c) ->
      val a = m.artifacts[name]
      if (a == null || a.shape != c.shape || a.elementCount != c.count || a.byteCount != c.bytes ||
        a.sha256 != c.sha || a.dtype != "float32" || a.numpyDtype != "<f4" || a.byteOrder != "little-endian") {
        fail("$name metadata contract is invalid.")
      }
    }
  }

  private fun readBounded(input: InputStream, maximum: Int, check: () -> Unit): ByteArray = input.use {
    val output = java.io.ByteArrayOutputStream()
    val buffer = ByteArray(4096)
    var total = 0
    while (true) {
      check()
      val read = it.read(buffer)
      if (read < 0) break
      total += read
      if (total > maximum) fail("Phase 2G metadata is too large.")
      output.write(buffer, 0, read)
    }
    output.toByteArray()
  }
  private fun ByteArray.toHex() = joinToString("") { "%02x".format(it) }
  private fun fail(message: String): Nothing = throw FluxPhase2gEvidenceException(message)
  private data class Contract(val shape: List<Int>, val count: Int, val bytes: Int, val sha: String)
  @Serializable private data class Artifact(val byteCount: Int, val byteOrder: String, val dtype: String, val elementCount: Int, val numpyDtype: String, val sha256: String, val shape: List<Int>)
  @Serializable private data class Libraries(val accelerate: String, val diffusers: String, val huggingface_hub: String, val numpy: String, val safetensors: String, val transformers: String)
  @Serializable private data class Metadata(
    val artifacts: Map<String, Artifact>, val authenticationPatchedScriptSha256: String,
    val baseModelRepository: String, val companionRepository: String, val extractionLibraries: Libraries,
    val generatedAt: String, val generatorScriptPath: String, val imageSize: List<Int>,
    val immutableBaseModelRevision: String, val immutableCompanionRevision: String, val mode: String,
    val originalGeneratorScriptSha256: String, val provenanceStatement: String, val purpose: String,
    val pythonVersion: String, val pytorchVersion: String, val schemaVersion: Int, val seed: Int, val steps: Int,
  )
  private const val MAX_METADATA_BYTES = 64 * 1024
}

/** Successful-result-only, thread-safe cache around an app-owned evidence source. */
class FluxPhase2gEvidenceSourceLoader(private val open: (String) -> InputStream) {
  @Volatile private var cached: FluxPhase2gEvidence? = null
  fun load(cancellationCheck: () -> Unit = {}): FluxPhase2gEvidence = cached ?: synchronized(this) {
    cached ?: FluxPhase2gEvidenceParser.parse(open, cancellationCheck).also { cached = it }
  }
}

class FluxPhase2gEvidenceLoader(assets: AssetManager) {
  private val source = FluxPhase2gEvidenceSourceLoader { assets.open("${FluxPhase2gEvidenceParser.DIRECTORY}/$it") }
  fun load(cancellationCheck: () -> Unit = {}): FluxPhase2gEvidence = source.load(cancellationCheck)
}
