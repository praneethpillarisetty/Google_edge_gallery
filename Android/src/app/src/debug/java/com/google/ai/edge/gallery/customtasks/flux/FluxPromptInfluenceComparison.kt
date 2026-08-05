/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Debug
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.gallery.customtasks.flux.decoder.FluxDecodedBitmapConverter
import com.google.ai.edge.gallery.customtasks.flux.decoder.FluxDecoderTail
import com.google.ai.edge.gallery.customtasks.flux.decoder.FluxPhase2hEvidenceLoader
import com.google.ai.edge.gallery.customtasks.flux.decoder.FluxVaeDecoder
import com.google.ai.edge.gallery.customtasks.flux.diagnostics.FluxFp32Summary
import com.google.ai.edge.gallery.customtasks.flux.diagnostics.FluxPromptInfluenceInterpretation
import com.google.ai.edge.gallery.customtasks.flux.diagnostics.FluxPromptInfluenceMetrics
import com.google.ai.edge.gallery.customtasks.flux.diagnostics.FluxTensorDiff
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxProductionNoiseFactory
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxSeedParser
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceConstantsLoader
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceImagePreprocessor
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceImageSourceStager
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceTokenEncoder
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceVaeEncoder
import com.google.ai.edge.gallery.customtasks.flux.prompt.FluxPromptAssetResolver
import com.google.ai.edge.gallery.customtasks.flux.prompt.FluxPromptConditioner
import com.google.ai.edge.gallery.customtasks.flux.prompt.FluxQwenTokenizer
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxLiteRtEnvironment
import com.google.ai.edge.gallery.customtasks.flux.transformer.FluxPhase2gEvidenceLoader
import com.google.ai.edge.gallery.customtasks.flux.transformer.FluxTransformerCoreResult
import com.google.ai.edge.gallery.customtasks.flux.transformer.FluxTransformerDenoiser
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

enum class FluxPromptComparisonStage { IDLE, VALIDATING, STAGING_REFERENCE, RUNNING_REFERENCE, TOKENIZING, CONDITIONING_PROMPT, DENOISING, DECODING, COMPARING, COMPLETE, CANCELLED, ERROR }

data class FluxPromptComparisonState(
  val running: Boolean = false,
  val runLabel: String = "none",
  val stage: FluxPromptComparisonStage = FluxPromptComparisonStage.IDLE,
  val currentStep: Int = 0,
  val completedGraphs: Int = 0,
  val elapsedMillis: Long = 0,
  val summary: String = "",
  val error: String? = null,
  val bitmapA: Bitmap? = null,
  val bitmapB: Bitmap? = null,
)

@HiltViewModel
class FluxPromptInfluenceComparisonViewModel @Inject constructor(
  @ApplicationContext private val context: Context,
  private val repository: FluxDownloadRepository,
) : ViewModel() {
  private val mutableState = MutableStateFlow(FluxPromptComparisonState())
  val state = mutableState.asStateFlow()
  private var job: Job? = null

  fun run(uri: Uri?, promptA: String, promptB: String, fixedSeedText: String, modelReadyHint: Boolean) {
    if (job?.isActive == true) return
    val selected = uri ?: return fail("Select one reference image before running comparison.")
    if (!modelReadyHint) return fail("Model repository must be Ready before running comparison.")
    if (promptA.isBlank() || promptB.isBlank()) return fail("Prompt A and Prompt B are required.")
    val seed = try { FluxSeedParser.parseFixed(fixedSeedText).value } catch (_: IllegalArgumentException) { return fail("Fixed seed must be a signed 64-bit integer.") }
    val previousState = mutableState.value
    mutableState.value = FluxPromptComparisonState(running = true, stage = FluxPromptComparisonStage.VALIDATING)
    recycleResultBitmaps(previousState)
    job = viewModelScope.launch {
      val started = System.nanoTime()
      val heapBefore = heap()
      val pssBefore = Debug.getPss()
      try {
        update("none", FluxPromptComparisonStage.VALIDATING, started)
        require(repository.events.first() is FluxDownloadEvent.Ready) { "Model repository is not Ready." }
        val initial = FluxProductionNoiseFactory().create(com.google.ai.edge.gallery.customtasks.flux.generation.FluxSeedSelection.Fixed(seed))
        val noiseA = initial.copyValues()
        val noiseB = initial.copyValues()
        require(noiseA.contentEquals(noiseB)) { "Initial-noise defensive copies differ." }
        val result = repository.withModelFilesLocked { root, manifest, metadata ->
          update("none", FluxPromptComparisonStage.STAGING_REFERENCE, started)
          FluxReferenceImageSourceStager(context.cacheDir, context.contentResolver, selected).withStagedSource { staged ->
            val evidence = FluxPhase2gEvidenceLoader(context.assets).load { coroutineContext.ensureActive() }
            val decoderEvidence = FluxPhase2hEvidenceLoader(context.assets).load()
            val constants = FluxReferenceConstantsLoader(context.assets).load()
            val image = FluxReferenceImagePreprocessor().preprocess(staged)
            var a: RunCapture? = null
            var b: RunCapture? = null
            var resultOwnsBitmaps = false
            try {
              a = runOne("A", promptA, noiseA, root, manifest, metadata, evidence, decoderEvidence, constants, image, started)
              coroutineContext.ensureActive()
              b = runOne("B", promptB, noiseB, root, manifest, metadata, evidence, decoderEvidence, constants, image, started)
              update("none", FluxPromptComparisonStage.COMPARING, started, 4, 32)
              buildComparison(seed, requireNotNull(a), requireNotNull(b), started, heapBefore, pssBefore).also { resultOwnsBitmaps = true }
            } finally {
              if (!resultOwnsBitmaps) {
                a?.bitmap?.recycle()
                b?.bitmap?.recycle()
              }
            }
          }
        }
        mutableState.value = FluxPromptComparisonState(stage = FluxPromptComparisonStage.COMPLETE, currentStep = 4, completedGraphs = 32, elapsedMillis = elapsed(started), summary = result.summary, bitmapA = result.bitmapA, bitmapB = result.bitmapB)
      } catch (_: CancellationException) {
        mutableState.value = FluxPromptComparisonState(stage = FluxPromptComparisonStage.CANCELLED, elapsedMillis = elapsed(started), summary = "interpretation: CANCELLED\ncancellation available: true")
      } catch (_: OutOfMemoryError) {
        fail("Prompt comparison ran out of memory.", started)
      } catch (_: LinkageError) {
        fail("LiteRT native linkage failed during prompt comparison.", started)
      } catch (_: Exception) {
        mutableState.value = FluxPromptComparisonState(stage = FluxPromptComparisonStage.ERROR, elapsedMillis = elapsed(started), summary = "interpretation: ERROR", error = "Prompt comparison failed at a sanitized pipeline boundary.")
      }
    }
  }

  fun cancel() { job?.cancel() }
  override fun onCleared() { job?.cancel(); recycleResultBitmaps(mutableState.value); super.onCleared() }

  private suspend fun runOne(label: String, prompt: String, initialLatents: FloatArray, root: java.io.File, manifest: FluxModelManifest, metadata: List<FluxFileMetadata>, evidence: com.google.ai.edge.gallery.customtasks.flux.transformer.FluxPhase2gEvidence, decoderEvidence: com.google.ai.edge.gallery.customtasks.flux.decoder.FluxPhase2hEvidence, constants: com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceConstants, image: com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceImageTensor, started: Long): RunCapture {
    coroutineContext.ensureActive()
    val runContext = coroutineContext
    update(label, FluxPromptComparisonStage.RUNNING_REFERENCE, started)
    FluxLiteRtEnvironment().use { environment ->
      val runner = environment.createGpuGraphRunner()
      val vae = FluxReferenceVaeEncoder(runner)
      verifySize(vae.resolve(root, manifest), metadata)
      val latent = vae.encode(vae.resolve(root, manifest), image)
      val reference = FluxReferenceTokenEncoder(constants).encode(latent)
      update(label, FluxPromptComparisonStage.TOKENIZING, started)
      val assets = FluxPromptAssetResolver(root, manifest).resolve()
      val tokens = FluxQwenTokenizer.load(assets.vocabulary.toPath(), assets.merges.toPath(), assets.specials.toPath()).prepareForTextEncoder(prompt).tokenIds.map(Int::toLong).toLongArray()
      update(label, FluxPromptComparisonStage.CONDITIONING_PROMPT, started)
      val conditioning = FluxPromptConditioner.create(FluxPromptAssetResolver(root, manifest), runner).condition(prompt)
      update(label, FluxPromptComparisonStage.DENOISING, started)
      val core = FluxTransformerDenoiser(runner).run(root, manifest, evidence, initialLatents.copyOf(), conditioning, reference) { progress ->
        update(label, FluxPromptComparisonStage.DENOISING, started, progress.step, progress.completedGraphs)
      }
      val finalLatents = core.copyFinalLatents()
      update(label, FluxPromptComparisonStage.DECODING, started, 4, 32)
      verifyModel(root, FluxVaeDecoder.GRAPH, manifest, metadata)
      val decoderInput = FluxDecoderTail(decoderEvidence, constants).prepare(finalLatents.copyOf()) { runContext.ensureActive() }.first
      val decoded = FluxVaeDecoder(runner).decode(root, manifest, decoderInput)
      val pixels = FluxDecodedBitmapConverter.pixels(decoded) { runContext.ensureActive() }
      val bitmap = Bitmap.createBitmap(pixels, 256, 256, Bitmap.Config.ARGB_8888)
      return RunCapture(label, tokens, conditioning.values.copyOf(), finalLatents, decoded.copyOf(), pixels, bitmap, core)
    }
  }

  private fun buildComparison(seed: Long, a: RunCapture, b: RunCapture, started: Long, heapBefore: Long, pssBefore: Long): ComparisonResult {
    val tokenDiff = FluxPromptInfluenceMetrics.compareTokens(a.tokens, b.tokens)
    val textDiff = FluxPromptInfluenceMetrics.compareFp32(a.conditioning, b.conditioning)
    val latentDiff = FluxPromptInfluenceMetrics.compareFp32(a.finalLatents, b.finalLatents)
    val decoderDiff = FluxPromptInfluenceMetrics.compareFp32(a.decoder, b.decoder)
    val bitmapDiff = FluxPromptInfluenceMetrics.compareArgb(256, 256, a.pixels, b.pixels)
    val interpretation = FluxPromptInfluenceMetrics.interpret(tokenDiff, textDiff, latentDiff, bitmapDiff)
    val summary = """
      |backend: GPU FP32
      |fixed seed: $seed
      |graph sequence: kv_vae_enc.tflite -> ke_enc0.tflite -> ke_enc1.tflite -> ke_enc2.tflite -> ${a.core.graphSequence.joinToString(" -> ")} -> ${FluxVaeDecoder.GRAPH}
      |runs completed: A and B
      |token diff: count=${tokenDiff.tokenCount}, differing=${tokenDiff.differingPositions}, percent=${tokenDiff.differingPercent}, aSha=${tokenDiff.aSha256LittleEndianInt64}, bSha=${tokenDiff.bSha256LittleEndianInt64}
      |text conditioning A: ${formatSummary(FluxPromptInfluenceMetrics.summarizeFp32(a.conditioning))}
      |text conditioning diff: ${formatDiff(textDiff)}
      |final latents A: ${formatSummary(FluxPromptInfluenceMetrics.summarizeFp32(a.finalLatents))}
      |final latents diff: ${formatDiff(latentDiff)}
      |decoder output A: ${formatSummary(FluxPromptInfluenceMetrics.summarizeFp32(a.decoder))}
      |decoder output diff: ${formatDiff(decoderDiff)}
      |bitmap diff: pixels=${bitmapDiff.width * bitmapDiff.height}, differing=${bitmapDiff.differingPixels}, percent=${bitmapDiff.differingPercent}, aSha=${bitmapDiff.aArgbSha256}, bSha=${bitmapDiff.bArgbSha256}, meanAbsARGB=${bitmapDiff.meanAbsA}/${bitmapDiff.meanAbsR}/${bitmapDiff.meanAbsG}/${bitmapDiff.meanAbsB}, rgbRmse=${bitmapDiff.rgbRmse}
      |Java heap: $heapBefore -> ${heap()} bytes
      |process PSS: $pssBefore -> ${Debug.getPss()} kB
      |cancellation available: true
      |interpretation: $interpretation
      |locally observed diagnostic hashes only; no prompt text, URI, path, image bytes, or full tensor values included
    """.trimMargin()
    return ComparisonResult(summary, a.bitmap, b.bitmap)
  }

  private fun formatSummary(s: FluxFp32Summary) = "count=${s.elementCount}, finite=${s.allFinite}, sha=${s.sha256LittleEndian}, min=${s.min}, max=${s.max}, mean=${s.mean}, stddev=${s.stddev}"
  private fun formatDiff(d: FluxTensorDiff) = "equal=${d.equalCount}, equalPercent=${d.equalPercent}, mae=${d.mae}, rmse=${d.rmse}, maxAbs=${d.maxAbsError}, cosine=${d.cosineSimilarity}, different=${d.different}"
  private suspend fun update(label: String, stage: FluxPromptComparisonStage, started: Long, step: Int = 0, graphs: Int = 0) { coroutineContext.ensureActive(); mutableState.value = FluxPromptComparisonState(true, label, stage, step, graphs, elapsed(started)) }
  private fun verifyModel(root: java.io.File, name: String, manifest: FluxModelManifest, metadata: List<FluxFileMetadata>) { require(name in manifest.files); verifySize(java.io.File(root.canonicalFile, name).canonicalFile, metadata) }
  private fun verifySize(file: java.io.File, metadata: List<FluxFileMetadata>) { val expected = metadata.singleOrNull { it.path == file.name || it.path.endsWith("/${file.name}") } ?: error("Required model metadata is missing."); require(file.length() == expected.size) }
  private fun fail(message: String, started: Long? = null) { mutableState.value = FluxPromptComparisonState(stage = FluxPromptComparisonStage.ERROR, elapsedMillis = started?.let(::elapsed) ?: 0, error = message, summary = "interpretation: ERROR") }
  private fun recycleResultBitmaps(state: FluxPromptComparisonState) {
    state.bitmapA?.takeUnless { it.isRecycled }?.recycle()
    state.bitmapB?.takeUnless { it.isRecycled }?.recycle()
  }
  private fun elapsed(start: Long) = (System.nanoTime() - start) / 1_000_000
  private fun heap() = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
  private data class RunCapture(val label: String, val tokens: LongArray, val conditioning: FloatArray, val finalLatents: FloatArray, val decoder: FloatArray, val pixels: IntArray, val bitmap: Bitmap, val core: FluxTransformerCoreResult)
  private data class ComparisonResult(val summary: String, val bitmapA: Bitmap, val bitmapB: Bitmap)
}
