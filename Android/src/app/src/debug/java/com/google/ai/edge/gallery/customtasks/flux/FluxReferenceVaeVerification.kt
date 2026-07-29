/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceImagePreprocessor
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceVaeEncoder
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxLiteRtEnvironment
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlin.coroutines.coroutineContext

enum class FluxReferenceVaeStage {
  IDLE, VALIDATING_FILES, READING_ORIENTATION, DECODING_IMAGE, APPLYING_ORIENTATION,
  CENTER_CROPPING, RESIZING, BUILDING_INPUT, HASHING_VAE_GRAPH, COMPILING_VAE,
  RUNNING_VAE, VALIDATING_OUTPUT, COMPLETE, CANCELLED, ERROR,
}

data class FluxReferenceVaeVerificationState(
  val running: Boolean = false,
  val stage: FluxReferenceVaeStage = FluxReferenceVaeStage.IDLE,
  val elapsedMillis: Long = 0,
  val summary: String = "",
  val error: String? = null,
)

@HiltViewModel
class FluxReferenceVaeVerificationViewModel @Inject constructor(
  @ApplicationContext private val context: Context,
  private val repository: FluxDownloadRepository,
) : ViewModel() {
  private val mutableState = MutableStateFlow(FluxReferenceVaeVerificationState())
  val state = mutableState.asStateFlow()
  private var job: Job? = null
  private val hashes = mutableMapOf<String, String>()
  private val hasher = FluxStreamingHasher(
    object : FluxHashCache {
      override fun get(identity: FluxHashIdentity) = hashes[identity.key]
      override fun put(identity: FluxHashIdentity, hash: String) { hashes[identity.key] = hash }
    },
  )

  @Suppress("UNUSED_PARAMETER")
  fun run(uri: Uri, modelReadyHint: Boolean) {
    if (job?.isActive == true) return
    val thermalBefore = thermal()
    if (thermalBefore >= PowerManager.THERMAL_STATUS_EMERGENCY) {
      mutableState.value = failure("Device thermal status is emergency/shutdown; wait for it to cool.")
      return
    }
    job = viewModelScope.launch {
      val started = System.nanoTime()
      mutableState.value = FluxReferenceVaeVerificationState(true, FluxReferenceVaeStage.VALIDATING_FILES)
      try {
        if (repository.events.first() !is FluxDownloadEvent.Ready) {
          throw IllegalStateException("Model repository must be Ready.")
        }
        val preprocessStarted = System.nanoTime()
        val tensor = FluxReferenceImagePreprocessor(context.contentResolver).preprocess(uri) {
          update(FluxReferenceVaeStage.valueOf(it), started)
        }
        val preprocessingMillis = elapsed(preprocessStarted)
        coroutineContext.ensureActive()
        val summary = repository.withModelFilesLocked { root, manifest, metadata ->
          update(FluxReferenceVaeStage.VALIDATING_FILES, started)
          val environment = FluxLiteRtEnvironment()
          try {
            val encoder = FluxReferenceVaeEncoder(environment.createGpuGraphRunner())
            val graph = encoder.resolve(root, manifest)
            val expected = metadata.singleOrNull { it.path == graph.name }
              ?: error("Authoritative VAE graph metadata is missing.")
            require(expected.size == graph.length()) {
              "VAE graph size does not match authoritative metadata."
            }
            update(FluxReferenceVaeStage.HASHING_VAE_GRAPH, started)
            val hash = hasher.hash(
              FluxHashIdentity(graph.name, graph.length(), graph.lastModified()),
              graph::inputStream,
            )
            coroutineContext.ensureActive()
            update(FluxReferenceVaeStage.COMPILING_VAE, started)
            val graphStarted = System.nanoTime()
            update(FluxReferenceVaeStage.RUNNING_VAE, started)
            val latent = encoder.encode(graph, tensor)
            coroutineContext.ensureActive()
            val graphMillis = elapsed(graphStarted)
            update(FluxReferenceVaeStage.VALIDATING_OUTPUT, started)
            """backend: GPU FP32
                |graph: ${graph.name}
                |locally observed SHA-256 (not publisher-verified): $hash
                |pinned input contract: ${tensor.shape} FP32
                |pinned output contract: ${latent.shape} FP32
                |runtime output elements: ${latent.values.size}
                |runtime all values finite: true
                |sampled decode estimate: ${tensor.decodePlan.estimatedWidth}x${tensor.decodePlan.estimatedHeight}, ${tensor.decodePlan.estimatedArgbBytes} bytes
                |preprocessing: $preprocessingMillis ms
                |graph: $graphMillis ms
                |total: ${elapsed(started)} ms
                |Java heap: ${usedHeap()} bytes
                |process PSS: ${Debug.getPss()} kB
                |thermal: $thermalBefore -> ${thermal()}
                |device: ${Build.MANUFACTURER} ${Build.MODEL}; Android API ${Build.VERSION.SDK_INT}
                |cancellation available: yes""".trimMargin()
          } finally {
            environment.close()
          }
        }
        mutableState.value = FluxReferenceVaeVerificationState(
          stage = FluxReferenceVaeStage.COMPLETE,
          elapsedMillis = elapsed(started),
          summary = summary,
        )
      } catch (cancelled: CancellationException) {
        mutableState.value = FluxReferenceVaeVerificationState(
          stage = FluxReferenceVaeStage.CANCELLED,
          elapsedMillis = elapsed(started),
          error = "Reference VAE verification cancelled.",
        )
      } catch (oom: OutOfMemoryError) {
        mutableState.value = failure("Not enough memory for reference VAE verification.", started)
      } catch (linkage: LinkageError) {
        mutableState.value = failure("LiteRT native linkage failed.", started)
      } catch (failure: Exception) {
        mutableState.value = failure(
          failure.message ?: "Reference VAE verification failed.",
          started,
        )
      }
    }
  }

  fun cancel() { job?.cancel() }
  override fun onCleared() { job?.cancel(); super.onCleared() }

  private fun update(stage: FluxReferenceVaeStage, started: Long) {
    mutableState.value = mutableState.value.copy(stage = stage, elapsedMillis = elapsed(started))
  }
  private fun failure(message: String, started: Long? = null) =
    FluxReferenceVaeVerificationState(
      stage = FluxReferenceVaeStage.ERROR,
      elapsedMillis = started?.let(::elapsed) ?: 0,
      error = message,
    )
  private fun elapsed(start: Long) = (System.nanoTime() - start) / 1_000_000
  private fun usedHeap() = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
  private fun thermal() = context.getSystemService(PowerManager::class.java).currentThermalStatus
}
