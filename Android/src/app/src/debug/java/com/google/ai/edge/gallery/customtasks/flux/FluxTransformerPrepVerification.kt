/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceConstantsLoader
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceImagePreprocessor
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceImageSourceStager
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceTokenEncoder
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceVaeEncoder
import com.google.ai.edge.gallery.customtasks.flux.prompt.FluxPromptAssetResolver
import com.google.ai.edge.gallery.customtasks.flux.prompt.FluxPromptConditioner
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxLiteRtEnvironment
import com.google.ai.edge.gallery.customtasks.flux.transformer.FluxEditingImageTokens
import com.google.ai.edge.gallery.customtasks.flux.transformer.FluxSyntheticDiagnosticInputs
import com.google.ai.edge.gallery.customtasks.flux.transformer.FluxTransformerPrepContracts
import com.google.ai.edge.gallery.customtasks.flux.transformer.FluxTransformerPrepRunner
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

enum class FluxTransformerPrepVerificationStage {
  IDLE, PREPARING_REFERENCE, ENCODING_REFERENCE_VAE, BUILDING_REFERENCE_TOKENS,
  ENCODING_PROMPT, BUILDING_SYNTHETIC_NOISE, ASSEMBLING_EDITING_SEQUENCE,
  BUILDING_SYNTHETIC_TIMESTEP, HASHING_TRANSFORMER_PREP, COMPILING_TRANSFORMER_PREP,
  RUNNING_TRANSFORMER_PREP, VALIDATING_TRANSFORMER_PREP_OUTPUTS, COMPLETE, CANCELLED, ERROR,
}

data class FluxTransformerPrepVerificationState(
  val running: Boolean = false,
  val stage: FluxTransformerPrepVerificationStage = FluxTransformerPrepVerificationStage.IDLE,
  val elapsedMillis: Long = 0,
  val summary: String = "",
  val error: String? = null,
)

@HiltViewModel
class FluxTransformerPrepVerificationViewModel @Inject constructor(
  @ApplicationContext private val context: Context,
  private val repository: FluxDownloadRepository,
) : ViewModel() {
  private val mutableState = MutableStateFlow(FluxTransformerPrepVerificationState())
  val state = mutableState.asStateFlow()
  private var job: Job? = null
  private val constantsLoader by lazy { FluxReferenceConstantsLoader(context.assets) }
  private val hashCache = mutableMapOf<String, String>()
  private val hasher = FluxStreamingHasher(object : FluxHashCache {
    override fun get(identity: FluxHashIdentity) = hashCache[identity.key]
    override fun put(identity: FluxHashIdentity, hash: String) { hashCache[identity.key] = hash }
  })

  fun run(uri: Uri, prompt: String, modelReadyHint: Boolean) {
    if (job?.isActive == true) return
    if (!modelReadyHint || prompt.isBlank()) {
      mutableState.value = error("Model repository must be Ready and prompt must be non-empty.")
      return
    }
    val thermalBefore = thermal()
    if (thermalBefore >= PowerManager.THERMAL_STATUS_EMERGENCY) {
      mutableState.value = error("Device thermal status is emergency/shutdown; wait for it to cool.")
      return
    }
    job = viewModelScope.launch {
      val started = System.nanoTime()
      mutableState.value = FluxTransformerPrepVerificationState(true, FluxTransformerPrepVerificationStage.PREPARING_REFERENCE)
      try {
        require(repository.events.first() is FluxDownloadEvent.Ready) { "Model repository must be Ready." }
        val summary = repository.withModelFilesLocked { root, manifest, metadata ->
          coroutineContext.ensureActive()
          FluxReferenceImageSourceStager(context.cacheDir, context.contentResolver, uri).withStagedSource { staged ->
            val timings = linkedMapOf<String, Long>()
            stage(FluxTransformerPrepVerificationStage.PREPARING_REFERENCE, started)
            val preprocessingStart = System.nanoTime()
            val image = FluxReferenceImagePreprocessor().preprocess(staged)
            timings["preprocessing"] = elapsed(preprocessingStart)
            coroutineContext.ensureActive()
            val environment = FluxLiteRtEnvironment()
            try {
              val graphRunner = environment.createGpuGraphRunner()
              val vae = FluxReferenceVaeEncoder(graphRunner)
              val vaeGraph = vae.resolve(root, manifest)
              verifySize(vaeGraph, metadata)
              stage(FluxTransformerPrepVerificationStage.ENCODING_REFERENCE_VAE, started)
              val vaeStart = System.nanoTime()
              val latent = vae.encode(vaeGraph, image)
              require(latent.shape == listOf(1, 32, 32, 32) && latent.dataType == "FP32")
              timings["VAE"] = elapsed(vaeStart)
              coroutineContext.ensureActive()

              stage(FluxTransformerPrepVerificationStage.BUILDING_REFERENCE_TOKENS, started)
              val tokenStart = System.nanoTime()
              val reference = FluxReferenceTokenEncoder(constantsLoader.load()).encode(latent)
              timings["reference tokens"] = elapsed(tokenStart)
              coroutineContext.ensureActive()

              stage(FluxTransformerPrepVerificationStage.ENCODING_PROMPT, started)
              val promptAssets = FluxPromptAssetResolver(root, manifest).resolve()
              (listOf(promptAssets.vocabulary, promptAssets.merges, promptAssets.specials, promptAssets.embeddings) + promptAssets.encoderGraphs)
                .forEach { verifySize(it, metadata) }
              val promptStart = System.nanoTime()
              val conditioning = FluxPromptConditioner.create(FluxPromptAssetResolver(root, manifest), graphRunner).condition(prompt)
              require(conditioning.shape == listOf(1, 512, 7680) && conditioning.values.size == 3_932_160)
              timings["text encoder"] = elapsed(promptStart)
              coroutineContext.ensureActive()

              stage(FluxTransformerPrepVerificationStage.BUILDING_SYNTHETIC_NOISE, started)
              val noise = FluxSyntheticDiagnosticInputs.syntheticZeroNoiseTokens()
              stage(FluxTransformerPrepVerificationStage.ASSEMBLING_EDITING_SEQUENCE, started)
              val assemblyStart = System.nanoTime()
              val editing = FluxEditingImageTokens.assemble(noise, reference)
              timings["sequence assembly"] = elapsed(assemblyStart)
              stage(FluxTransformerPrepVerificationStage.BUILDING_SYNTHETIC_TIMESTEP, started)
              val timestep = FluxSyntheticDiagnosticInputs.syntheticZeroTimestepEmbedding()
              coroutineContext.ensureActive()

              val prep = FluxTransformerPrepRunner(graphRunner)
              val prepGraph = prep.resolve(root, manifest)
              verifySize(prepGraph, metadata)
              stage(FluxTransformerPrepVerificationStage.HASHING_TRANSFORMER_PREP, started)
              val hash = hasher.hash(FluxHashIdentity(prepGraph.name, prepGraph.length(), prepGraph.lastModified()), prepGraph::inputStream)
              coroutineContext.ensureActive()
              stage(FluxTransformerPrepVerificationStage.COMPILING_TRANSFORMER_PREP, started)
              val prepStart = System.nanoTime()
              stage(FluxTransformerPrepVerificationStage.RUNNING_TRANSFORMER_PREP, started)
              val report = prep.run(prepGraph, editing, conditioning, timestep)
              timings["prep compile/run"] = elapsed(prepStart)
              stage(FluxTransformerPrepVerificationStage.VALIDATING_TRANSFORMER_PREP_OUTPUTS, started)
              coroutineContext.ensureActive()
              buildSummary(hash, conditioning.shape, reference.shape, editing.shape, report.outputContracts.map { it.shape to it.elements }, timings, started, thermalBefore)
            } finally {
              environment.close()
            }
          }
        }
        coroutineContext.ensureActive()
        mutableState.value = FluxTransformerPrepVerificationState(stage = FluxTransformerPrepVerificationStage.COMPLETE, elapsedMillis = elapsed(started), summary = summary)
      } catch (_: CancellationException) {
        mutableState.value = FluxTransformerPrepVerificationState(stage = FluxTransformerPrepVerificationStage.CANCELLED, elapsedMillis = elapsed(started), error = "Editing transformer prep verification cancelled.")
      } catch (_: OutOfMemoryError) {
        mutableState.value = error("Not enough memory for editing transformer prep verification.", started)
      } catch (_: LinkageError) {
        mutableState.value = error("LiteRT native linkage failed during editing transformer prep verification.", started)
      } catch (_: Exception) {
        mutableState.value = error("Editing transformer prep verification failed. Review local debug logs without sharing sensitive input data.", started)
      }
    }
  }

  fun cancel() { job?.cancel() }
  override fun onCleared() { job?.cancel(); super.onCleared() }

  private fun verifySize(file: java.io.File, metadata: List<FluxFileMetadata>) {
    val expected = metadata.singleOrNull { it.path == file.name || it.path.endsWith("/${file.name}") }
      ?: throw IllegalStateException("Authoritative metadata is missing for a required model asset.")
    require(file.length() == expected.size) { "A required model asset does not match authoritative metadata." }
  }
  private suspend fun stage(value: FluxTransformerPrepVerificationStage, started: Long) {
    coroutineContext.ensureActive()
    mutableState.value = mutableState.value.copy(running = true, stage = value, elapsedMillis = elapsed(started), summary = "", error = null)
  }
  private suspend fun buildSummary(hash: String, promptShape: List<Int>, referenceShape: List<Int>, editingShape: List<Int>, outputs: List<Pair<List<Int>, Int>>, timings: Map<String, Long>, started: Long, thermalBefore: Int): String {
    coroutineContext.ensureActive()
    val outputLines = outputs.mapIndexed { index, value -> "output ${index + 1}: ${value.first}; ${value.second} elements" }.joinToString("\n")
    return """backend: GPU FP32
      |graph: kce_prep.tflite
      |locally observed graph SHA-256 (not publisher-verified): $hash
      |synthetic zero noise: yes
      |synthetic zero timestep embedding: yes
      |prompt conditioning: $promptShape; 3932160 elements
      |reference tokens: $referenceShape; 32768 elements
      |combined editing tokens: $editingShape; 65536 elements
      |noise occupies tokens 0–255: yes
      |reference occupies tokens 256–511: yes
      |$outputLines
      |all outputs finite: yes
      |${timings.entries.joinToString("\n") { "${it.key}: ${it.value} ms" }}
      |total: ${elapsed(started)} ms
      |Java heap: ${Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()} bytes
      |process PSS: ${Debug.getPss()} kB
      |thermal: $thermalBefore -> ${thermal()}
      |device: ${Build.MANUFACTURER} ${Build.MODEL}; Android API ${Build.VERSION.SDK_INT}
      |cancellation available: yes""".trimMargin()
  }
  private fun error(message: String, started: Long? = null) = FluxTransformerPrepVerificationState(stage = FluxTransformerPrepVerificationStage.ERROR, elapsedMillis = started?.let(::elapsed) ?: 0, error = message)
  private fun elapsed(start: Long) = (System.nanoTime() - start) / 1_000_000
  private fun thermal() = context.getSystemService(PowerManager::class.java).currentThermalStatus
}
