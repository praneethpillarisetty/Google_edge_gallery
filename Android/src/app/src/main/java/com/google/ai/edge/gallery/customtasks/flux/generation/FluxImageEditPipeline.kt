/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.generation

import android.content.Context
import android.net.Uri
import com.google.ai.edge.gallery.customtasks.flux.FluxDownloadEvent
import com.google.ai.edge.gallery.customtasks.flux.FluxDownloadRepository
import com.google.ai.edge.gallery.customtasks.flux.FluxFileMetadata
import com.google.ai.edge.gallery.customtasks.flux.FluxModelManifest
import com.google.ai.edge.gallery.customtasks.flux.decoder.FluxDecodedBitmapConverter
import com.google.ai.edge.gallery.customtasks.flux.decoder.FluxDecoderTail
import com.google.ai.edge.gallery.customtasks.flux.decoder.FluxPhase2hEvidenceLoader
import com.google.ai.edge.gallery.customtasks.flux.decoder.FluxVaeDecoder
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceConstantsLoader
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceImagePreprocessor
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceImageSourceStager
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceTokenEncoder
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceVaeEncoder
import com.google.ai.edge.gallery.customtasks.flux.prompt.FluxPromptAssetResolver
import com.google.ai.edge.gallery.customtasks.flux.prompt.FluxPromptConditioner
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxLiteRtEnvironment
import com.google.ai.edge.gallery.customtasks.flux.transformer.FluxPhase2gEvidenceLoader
import com.google.ai.edge.gallery.customtasks.flux.transformer.FluxTransformerDenoiser
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import javax.inject.Inject
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.first

interface FluxImageEditGenerator {
  suspend fun generate(reference: Uri, prompt: String, progress: (FluxGenerationProgress) -> Unit): FluxGenerationResult
}

/** Production owner for the single, verified GPU-FP32 image-edit graph sequence. */
class FluxImageEditPipeline @Inject constructor(
  @ApplicationContext private val context: Context,
  private val repository: FluxDownloadRepository,
) : FluxImageEditGenerator {
  override suspend fun generate(reference: Uri, prompt: String, progress: (FluxGenerationProgress) -> Unit): FluxGenerationResult {
    if (prompt.isBlank()) throw FluxGenerationException(FluxGenerationError.PROMPT_MISSING)
    if (repository.events.first() !is FluxDownloadEvent.Ready) throw FluxGenerationException(FluxGenerationError.MODEL_NOT_READY)
    val generationContext = coroutineContext
    val started = System.nanoTime()
    try {
      progress(FluxGenerationProgress(FluxGenerationStage.VALIDATING))
      return repository.withModelFilesLocked { root, manifest, metadata ->
        val evidence = FluxPhase2gEvidenceLoader(context.assets).load { generationContext.ensureActive() }
        val decoderEvidence = FluxPhase2hEvidenceLoader(context.assets).load()
        progress(FluxGenerationProgress(FluxGenerationStage.STAGING_REFERENCE))
        FluxReferenceImageSourceStager(context.cacheDir, context.contentResolver, reference).withStagedSource { staged ->
          coroutineContext.ensureActive()
          progress(FluxGenerationProgress(FluxGenerationStage.PREPROCESSING_REFERENCE))
          val image = FluxReferenceImagePreprocessor().preprocess(staged)
          FluxLiteRtEnvironment().use { environment ->
            val runner = environment.createGpuGraphRunner()
            progress(FluxGenerationProgress(FluxGenerationStage.ENCODING_REFERENCE, currentGraph = "kv_vae_enc.tflite"))
            val encoder = FluxReferenceVaeEncoder(runner)
            val encoderGraph = encoder.resolve(root, manifest).also { verify(it, metadata) }
            val latent = encoder.encode(encoderGraph, image)
            coroutineContext.ensureActive()
            progress(FluxGenerationProgress(FluxGenerationStage.PREPARING_REFERENCE_TOKENS))
            val constants = FluxReferenceConstantsLoader(context.assets).load()
            val referenceTokens = FluxReferenceTokenEncoder(constants).encode(latent)
            progress(FluxGenerationProgress(FluxGenerationStage.CONDITIONING_PROMPT))
            // Deliberately pass the literal editor body. FluxPromptConditioner alone owns Qwen wrapping.
            val conditioning = FluxPromptConditioner.create(FluxPromptAssetResolver(root, manifest), runner).condition(prompt)
            val core = FluxTransformerDenoiser(runner).run(root, manifest, evidence, conditioning, referenceTokens) { p ->
              val stage = when (p.step) { 1 -> FluxGenerationStage.TRANSFORMER_STEP_1; 2 -> FluxGenerationStage.TRANSFORMER_STEP_2; 3 -> FluxGenerationStage.TRANSFORMER_STEP_3; else -> FluxGenerationStage.TRANSFORMER_STEP_4 }
              progress(FluxGenerationProgress(stage, p.step, p.graph, p.completedGraphs))
            }
            coroutineContext.ensureActive()
            progress(FluxGenerationProgress(FluxGenerationStage.PREPARING_DECODER, 4, completedGraphCount = 32))
            val decoderInput = FluxDecoderTail(decoderEvidence, constants).prepare(core.copyFinalLatents()) { generationContext.ensureActive() }.first
            progress(FluxGenerationProgress(FluxGenerationStage.DECODING_VAE, 4, FluxVaeDecoder.GRAPH, 32))
            verify(File(root, FluxVaeDecoder.GRAPH), metadata)
            val decoded = FluxVaeDecoder(runner).decode(root, manifest, decoderInput)
            coroutineContext.ensureActive()
            progress(FluxGenerationProgress(FluxGenerationStage.CREATING_BITMAP, 4, completedGraphCount = 32))
            val bitmap = FluxDecodedBitmapConverter.bitmap(decoded) { generationContext.ensureActive() }
            coroutineContext.ensureActive()
            FluxGenerationResult(bitmap, (System.nanoTime() - started) / 1_000_000)
          }
        }
      }
    } catch (e: FluxGenerationException) { throw e }
      catch (e: OutOfMemoryError) { throw FluxGenerationException(FluxGenerationError.OUT_OF_MEMORY, e) }
      catch (e: LinkageError) { throw FluxGenerationException(FluxGenerationError.NATIVE_LINKAGE, e) }
      catch (e: Exception) { throw FluxGenerationException(classify(e), e) }
  }

  private fun verify(file: File, metadata: List<FluxFileMetadata>) {
    val expected = metadata.singleOrNull { it.path == file.name || it.path.endsWith("/${file.name}") }
      ?: throw FluxGenerationException(FluxGenerationError.MODEL_CHANGED)
    if (!file.canonicalFile.isFile || file.length() != expected.size) throw FluxGenerationException(FluxGenerationError.MODEL_CHANGED)
  }

  private fun classify(error: Exception) = when {
    error is SecurityException -> FluxGenerationError.REFERENCE_UNAVAILABLE
    error.message?.contains("finite", true) == true -> FluxGenerationError.NON_FINITE_TENSOR
    error.message?.contains("evidence", true) == true -> FluxGenerationError.INVALID_EVIDENCE
    error.message?.contains("model", true) == true || error.message?.contains("graph", true) == true -> FluxGenerationError.GPU_FAILURE
    error.message?.contains("image", true) == true -> FluxGenerationError.UNSUPPORTED_IMAGE
    else -> FluxGenerationError.UNKNOWN
  }
}
