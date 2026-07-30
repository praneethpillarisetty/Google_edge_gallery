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
import com.google.ai.edge.gallery.customtasks.flux.transformer.FluxPhase2gEvidenceHashes
import com.google.ai.edge.gallery.customtasks.flux.transformer.FluxPhase2gEvidenceLoader
import com.google.ai.edge.gallery.customtasks.flux.transformer.FluxPhase2gEvidenceParser
import com.google.ai.edge.gallery.customtasks.flux.transformer.FluxTransformerCoreResult
import com.google.ai.edge.gallery.customtasks.flux.transformer.FluxTransformerDenoiser
import com.google.ai.edge.gallery.customtasks.flux.transformer.FluxTransformerDenoisingContracts
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

enum class FluxTransformerVerificationStage {
  IDLE, VALIDATING_FILES, PREPROCESSING_REFERENCE, RUNNING_REFERENCE_VAE,
  PREPARING_REFERENCE_TOKENS, RUNNING_TEXT_ENCODER, ASSEMBLING_EDIT_SEQUENCE,
  RUNNING_PREP_GRAPH, DENOISING_STEP_1, DENOISING_STEP_2, DENOISING_STEP_3,
  DENOISING_STEP_4, VALIDATING_FINAL_LATENTS, COMPLETE, CANCELLED, ERROR,
}

data class FluxTransformerVerificationState(
  val running: Boolean = false,
  val stage: FluxTransformerVerificationStage = FluxTransformerVerificationStage.IDLE,
  val currentStep: Int = 0,
  val currentGraph: String = "none",
  val completedGraphs: Int = 0,
  val elapsedMillis: Long = 0,
  val cancellationAvailable: Boolean = false,
  val summary: String = "",
  val error: String? = null,
)

data class FluxTransformerVerificationResult(
  val backend: String,
  val graphSequence: List<String>,
  val completedSteps: Int,
  val initialShape: List<Int>,
  val initialElements: Int,
  val finalShape: List<Int>,
  val finalElements: Int,
  val allFinite: Boolean,
  val evidenceHashes: FluxPhase2gEvidenceHashes,
  val preprocessingMillis: Long,
  val vaeMillis: Long,
  val referenceTokensMillis: Long,
  val textConditioningMillis: Long,
  val stepDurationsMillis: List<Long>,
  val graphDurationsMillis: List<String>,
  val totalMillis: Long,
  val javaHeapBefore: Long,
  val javaHeapAfter: Long,
  val pssBeforeKb: Long,
  val pssAfterKb: Long,
  val thermalBefore: Int,
  val thermalAfter: Int,
  val device: String,
  val androidApi: Int,
  val cancellationAvailable: Boolean,
) {
  fun sanitizedSummary(): String = """backend: $backend
    |graph sequence: ${graphSequence.joinToString(" -> ")}
    |completed steps: $completedSteps/4
    |initial latents: $initialShape; $initialElements elements
    |final latents: $finalShape; $finalElements elements
    |all values finite: $allFinite
    |evidence SHA-256: latents0=${evidenceHashes.latents0}, temb=${evidenceHashes.temb}, dsigma=${evidenceHashes.dsigma}, cos=${evidenceHashes.cos}, sin=${evidenceHashes.sin}
    |preprocessing: $preprocessingMillis ms
    |reference VAE: $vaeMillis ms
    |reference tokens: $referenceTokensMillis ms
    |text conditioning: $textConditioningMillis ms
    |per-step: ${stepDurationsMillis.mapIndexed { i, value -> "${i + 1}=$value ms" }.joinToString(", ")}
    |per-graph:\n${graphDurationsMillis.joinToString("\n")}
    |total: $totalMillis ms
    |Java heap: $javaHeapBefore -> $javaHeapAfter bytes
    |process PSS: $pssBeforeKb -> $pssAfterKb kB
    |thermal: $thermalBefore -> $thermalAfter
    |device: $device; Android API $androidApi
    |cancellation available: $cancellationAvailable
    |VAE decoding: deferred
    |Generate: disabled""".trimMargin()
}

@HiltViewModel
class FluxTransformerDenoisingVerificationViewModel @Inject constructor(
  @ApplicationContext private val context: Context,
  private val repository: FluxDownloadRepository,
) : ViewModel() {
  private val mutableState = MutableStateFlow(FluxTransformerVerificationState())
  val state = mutableState.asStateFlow()
  private val evidenceLoader by lazy { FluxPhase2gEvidenceLoader(context.assets) }
  private val referenceConstants by lazy { FluxReferenceConstantsLoader(context.assets) }
  private var job: Job? = null

  fun run(uri: Uri, prompt: String, modelReadyHint: Boolean) {
    if (job?.isActive == true) return
    if (!modelReadyHint || prompt.isBlank()) { failState("Model repository must be Ready, with a selected image and non-empty prompt."); return }
    val thermalBefore = thermal()
    if (thermalBefore >= PowerManager.THERMAL_STATUS_EMERGENCY) { failState("Device thermal state is emergency or shutdown; wait for it to cool."); return }
    job = viewModelScope.launch {
      val started = System.nanoTime()
      val heapBefore = heap()
      val pssBefore = Debug.getPss()
      update(FluxTransformerVerificationStage.VALIDATING_FILES, started)
      try {
        require(repository.events.first() is FluxDownloadEvent.Ready) { "Model repository is not Ready." }
        val result = repository.withModelFilesLocked { root, manifest, metadata ->
          coroutineContext.ensureActive()
          val evidence = evidenceLoader.load { coroutineContext.ensureActive() }
          FluxTransformerDenoisingContracts.GRAPH_ORDER.forEach { name -> verifyModel(root, name, manifest, metadata) }
          FluxReferenceImageSourceStager(context.cacheDir, context.contentResolver, uri).withStagedSource { staged ->
            update(FluxTransformerVerificationStage.PREPROCESSING_REFERENCE, started)
            coroutineContext.ensureActive()
            val preprocessingStart = System.nanoTime()
            val image = FluxReferenceImagePreprocessor().preprocess(staged)
            val preprocessing = elapsed(preprocessingStart)
            val environment = FluxLiteRtEnvironment()
            try {
              val runner = environment.createGpuGraphRunner()
              update(FluxTransformerVerificationStage.RUNNING_REFERENCE_VAE, started)
              coroutineContext.ensureActive()
              val vae = FluxReferenceVaeEncoder(runner)
              val vaeGraph = vae.resolve(root, manifest)
              verifySize(vaeGraph, metadata)
              val vaeStart = System.nanoTime()
              val latent = vae.encode(vaeGraph, image)
              val vaeMillis = elapsed(vaeStart)
              coroutineContext.ensureActive()
              update(FluxTransformerVerificationStage.PREPARING_REFERENCE_TOKENS, started)
              val referenceStart = System.nanoTime()
              val reference = FluxReferenceTokenEncoder(referenceConstants.load()).encode(latent)
              val referenceMillis = elapsed(referenceStart)
              coroutineContext.ensureActive()
              update(FluxTransformerVerificationStage.RUNNING_TEXT_ENCODER, started)
              val promptAssets = FluxPromptAssetResolver(root, manifest).resolve()
              (listOf(promptAssets.vocabulary, promptAssets.merges, promptAssets.specials, promptAssets.embeddings) + promptAssets.encoderGraphs).forEach { verifySize(it, metadata) }
              val textStart = System.nanoTime()
              val conditioning = FluxPromptConditioner.create(FluxPromptAssetResolver(root, manifest), runner).condition(prompt)
              val textMillis = elapsed(textStart)
              coroutineContext.ensureActive()
              update(FluxTransformerVerificationStage.ASSEMBLING_EDIT_SEQUENCE, started)
              val core = FluxTransformerDenoiser(runner).run(root, manifest, evidence, conditioning, reference) { progress ->
                update(stageFor(progress.step, progress.graph), started, progress.step, progress.graph, progress.completedGraphs)
              }
              update(FluxTransformerVerificationStage.VALIDATING_FINAL_LATENTS, started, 4, "none", 32)
              coroutineContext.ensureActive()
              buildResult(core, preprocessing, vaeMillis, referenceMillis, textMillis, started, heapBefore, pssBefore, thermalBefore)
            } finally {
              environment.close()
            }
          }
        }
        coroutineContext.ensureActive()
        mutableState.value = FluxTransformerVerificationState(
          stage = FluxTransformerVerificationStage.COMPLETE,
          currentStep = 4,
          completedGraphs = 32,
          elapsedMillis = elapsed(started),
          summary = result.sanitizedSummary(),
        )
      } catch (_: CancellationException) {
        mutableState.value = FluxTransformerVerificationState(stage = FluxTransformerVerificationStage.CANCELLED, elapsedMillis = elapsed(started), error = "Transformer verification cancelled.")
      } catch (_: OutOfMemoryError) {
        failState("Not enough memory for transformer verification.", started)
      } catch (_: LinkageError) {
        failState("LiteRT native linkage failed.", started)
      } catch (failure: Exception) {
        failState(sanitize(failure), started)
      }
    }
  }

  fun cancel() { job?.cancel() }
  override fun onCleared() { job?.cancel(); super.onCleared() }

  private fun verifyModel(root: java.io.File, name: String, manifest: FluxModelManifest, metadata: List<FluxFileMetadata>) {
    require(name in manifest.files)
    val canonicalRoot = root.canonicalFile
    val file = java.io.File(canonicalRoot, name).canonicalFile
    require(file.parentFile == canonicalRoot && file.isFile)
    verifySize(file, metadata)
  }
  private fun verifySize(file: java.io.File, metadata: List<FluxFileMetadata>) {
    val expected = metadata.singleOrNull { it.path == file.name || it.path.endsWith("/${file.name}") } ?: error("Required model metadata is missing.")
    require(file.length() == expected.size) { "Required model file changed." }
  }
  private suspend fun update(stage: FluxTransformerVerificationStage, started: Long, step: Int = 0, graph: String = "none", completed: Int = 0) {
    coroutineContext.ensureActive()
    mutableState.value = FluxTransformerVerificationState(true, stage, step, graph, completed, elapsed(started), true)
  }
  private fun stageFor(step: Int, graph: String) = if (graph == "kce_prep.tflite") FluxTransformerVerificationStage.RUNNING_PREP_GRAPH else when (step) {
    1 -> FluxTransformerVerificationStage.DENOISING_STEP_1
    2 -> FluxTransformerVerificationStage.DENOISING_STEP_2
    3 -> FluxTransformerVerificationStage.DENOISING_STEP_3
    else -> FluxTransformerVerificationStage.DENOISING_STEP_4
  }
  private fun buildResult(core: FluxTransformerCoreResult, preprocessing: Long, vae: Long, reference: Long, text: Long, started: Long, heapBefore: Long, pssBefore: Long, thermalBefore: Int) =
    FluxTransformerVerificationResult(
      core.backend, core.graphSequence, core.completedSteps, core.initialShape, core.initialElements,
      core.finalShape, core.finalElements, core.allFinite, FluxPhase2gEvidenceParser.expectedHashes(),
      preprocessing, vae, reference, text, core.stepDurationsMillis,
      core.graphTimings.map { "step ${it.step} ${it.graph}: ${it.durationMillis} ms" }, elapsed(started),
      heapBefore, heap(), pssBefore, Debug.getPss(), thermalBefore, thermal(),
      "${Build.MANUFACTURER} ${Build.MODEL}", Build.VERSION.SDK_INT, true,
    )
  private fun sanitize(failure: Exception): String = when {
    failure.message?.contains("evidence", true) == true -> "Phase 2G evidence validation failed."
    failure.message?.contains("model", true) == true -> "A required model file is missing or changed."
    failure.message?.contains("image", true) == true -> "The selected source image could not be read or is unsupported."
    failure.message?.contains("non-finite", true) == true -> "A transformer graph produced a non-finite tensor."
    failure.message?.contains("graph", true) == true -> "GPU graph compilation or execution failed."
    else -> "Transformer verification failed unexpectedly."
  }
  private fun failState(message: String, started: Long? = null) { mutableState.value = FluxTransformerVerificationState(stage = FluxTransformerVerificationStage.ERROR, elapsedMillis = started?.let(::elapsed) ?: 0, error = message) }
  private fun elapsed(start: Long) = (System.nanoTime() - start) / 1_000_000
  private fun heap() = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
  private fun thermal() = context.getSystemService(PowerManager::class.java).currentThermalStatus
}
