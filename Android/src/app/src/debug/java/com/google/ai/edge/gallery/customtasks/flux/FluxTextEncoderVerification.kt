/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.gallery.customtasks.flux.prompt.FluxPromptAssetResolver
import com.google.ai.edge.gallery.customtasks.flux.prompt.FluxPromptConditioner
import com.google.ai.edge.gallery.customtasks.flux.prompt.FluxTextEncoderContracts
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxLiteRtEnvironment
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import javax.inject.Inject
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.coroutines.coroutineContext

enum class FluxVerificationStage {
  IDLE, VALIDATING_FILES, HASHING_ENC0, HASHING_ENC1, HASHING_ENC2,
  TOKENIZING, EMBEDDING, BUILDING_MASK, BUILDING_ROTARY,
  RUNNING_ENC0, RUNNING_ENC1, RUNNING_ENC2, INTERLEAVING,
  COMPLETE, CANCELLED, ERROR,
}

data class FluxVerificationUiState(
  val running: Boolean = false,
  val stage: FluxVerificationStage = FluxVerificationStage.IDLE,
  val elapsedMillis: Long = 0,
  val hashBytes: Long = 0,
  val hashTotalBytes: Long = 0,
  val summary: String = "",
  val error: String? = null,
)

@HiltViewModel
class FluxVerificationViewModel @Inject constructor(
  @ApplicationContext private val context: Context,
  private val repository: FluxDownloadRepository,
) : ViewModel() {
  private val mutableState = MutableStateFlow(FluxVerificationUiState())
  val state = mutableState.asStateFlow()
  private var job: Job? = null

  fun run(prompt: String, modelReady: Boolean) {
    if (job?.isActive == true) return
    if (!modelReady) {
      mutableState.value = FluxVerificationUiState(stage = FluxVerificationStage.ERROR, error = "Model repository must be Ready.")
      return
    }
    val thermalBefore = thermalStatus(context)
    if (thermalBefore >= PowerManager.THERMAL_STATUS_EMERGENCY) {
      mutableState.value = FluxVerificationUiState(stage = FluxVerificationStage.ERROR, error = "Device thermal status is emergency/shutdown; wait for it to cool.")
      return
    }
    job = viewModelScope.launch {
      val started = System.nanoTime()
      mutableState.value = FluxVerificationUiState(running = true, stage = FluxVerificationStage.VALIDATING_FILES)
      try {
        val summary = withContext(Dispatchers.Default) {
          repository.withModelFilesLocked { root, manifest, metadata ->
            runLocked(root, manifest, metadata, prompt, thermalBefore, started)
          }
        }
        mutableState.value = FluxVerificationUiState(stage = FluxVerificationStage.COMPLETE, elapsedMillis = elapsed(started), summary = summary)
      } catch (cancelled: CancellationException) {
        mutableState.value = FluxVerificationUiState(stage = FluxVerificationStage.CANCELLED, elapsedMillis = elapsed(started), error = "Text encoder verification cancelled.")
      } catch (oom: OutOfMemoryError) {
        mutableState.value = FluxVerificationUiState(stage = FluxVerificationStage.ERROR, elapsedMillis = elapsed(started), error = "Not enough memory for text encoder verification.")
      } catch (link: UnsatisfiedLinkError) {
        mutableState.value = FluxVerificationUiState(stage = FluxVerificationStage.ERROR, elapsedMillis = elapsed(started), error = "LiteRT native linkage failed: ${link.message.orEmpty()}")
      } catch (failure: Exception) {
        mutableState.value = FluxVerificationUiState(stage = FluxVerificationStage.ERROR, elapsedMillis = elapsed(started), error = failure.message ?: "Text encoder verification failed.")
      }
    }
  }

  fun cancel() { job?.cancel() }

  private suspend fun runLocked(
    root: File,
    manifest: FluxModelManifest,
    metadata: List<FluxFileMetadata>,
    prompt: String,
    thermalBefore: Int,
    started: Long,
  ): String {
    update(FluxVerificationStage.VALIDATING_FILES, started)
    val assets = FluxPromptAssetResolver(root, manifest).resolve()
    val required = listOf(assets.vocabulary, assets.merges, assets.specials, assets.embeddings) + assets.encoderGraphs
    required.forEach { file ->
      coroutineContext.ensureActive()
      val relative = file.relativeTo(root.canonicalFile).invariantSeparatorsPath
      val expected = metadata.singleOrNull { it.path == relative }
        ?: error("Authoritative size metadata is missing for ${file.name}.")
      require(file.length() == expected.size) { "Size mismatch for ${file.name}: expected ${expected.size} bytes." }
    }
    val hashes = assets.encoderGraphs.mapIndexed { index, file -> observedHash(file, index, started) }
    coroutineContext.ensureActive()
    val heapBefore = usedHeap()
    val processBefore = processPssKb()
    val stageStarts = mutableMapOf<FluxVerificationStage, Long>()
    val stageDurations = linkedMapOf<FluxVerificationStage, Long>()
    val environment = FluxLiteRtEnvironment()
    var shape: List<Int>? = null
    var count = 0
    var finite = false
    var encoderDurations = emptyList<Long>()
    try {
      val conditioner = FluxPromptConditioner.create(FluxPromptAssetResolver(root, manifest), environment.createGpuGraphRunner())
      val conditioning = conditioner.condition(prompt) { conditioningStage ->
        val stage = FluxVerificationStage.valueOf(conditioningStage.name)
        val now = System.nanoTime()
        stageStarts.keys.lastOrNull()?.let { previous ->
          if (previous !in stageDurations) stageDurations[previous] = (now - stageStarts.getValue(previous)) / 1_000_000
        }
        stageStarts[stage] = now
        update(stage, started)
      }
      shape = conditioning.shape.toList()
      count = conditioning.values.size
      finite = conditioning.values.all(Float::isFinite)
      encoderDurations = conditioning.stageDurationsMillis.toList()
      if (shape != listOf(1, 512, 7680) || count != FluxTextEncoderContracts().conditioningElements) error("Output-size mismatch: $shape ($count elements).")
      if (!finite) error("Text encoder output contains non-finite values.")
      // No tensor or FluxTextConditioning reference escapes this scope.
    } finally {
      environment.close()
    }
    val heapAfter = usedHeap()
    val processAfter = processPssKb()
    val preprocessing = stageDurations.filterKeys { it in setOf(FluxVerificationStage.TOKENIZING, FluxVerificationStage.EMBEDDING, FluxVerificationStage.BUILDING_MASK, FluxVerificationStage.BUILDING_ROTARY) }.values.sum()
    return buildString {
      appendLine("FLUX text encoder device verification")
      appendLine("backend: GPU FP32")
      appendLine("graphs: ${assets.encoderGraphs.joinToString { it.name }}")
      hashes.forEachIndexed { i, hash -> appendLine("${assets.encoderGraphs[i].name} SHA-256 (observed): $hash") }
      appendLine("final shape: ${shape!!.joinToString(prefix = "[", postfix = "]")}")
      appendLine("output elements: $count")
      appendLine("all values finite: $finite")
      appendLine("tokenizer/preprocessing: ${preprocessing} ms")
      encoderDurations.forEachIndexed { i, duration -> appendLine("encoder ${i + 1}: $duration ms") }
      appendLine("total: ${elapsed(started)} ms")
      appendLine("approx Java heap before/after: $heapBefore / $heapAfter bytes")
      appendLine("approx process PSS before/after: ${processBefore ?: "unavailable"} / ${processAfter ?: "unavailable"} kB")
      appendLine("device: ${Build.MANUFACTURER} ${Build.MODEL}; Android API ${Build.VERSION.SDK_INT}")
      appendLine("thermal before/after: $thermalBefore / ${thermalStatus(context)}")
      append("cancellation available: yes")
    }
  }

  private suspend fun observedHash(file: File, index: Int, started: Long): String {
    val prefs = context.getSharedPreferences("flux_encoder_observed_hashes", Context.MODE_PRIVATE)
    val identity = FluxHashIdentity(file.name, file.length(), file.lastModified())
    val stage = listOf(FluxVerificationStage.HASHING_ENC0, FluxVerificationStage.HASHING_ENC1, FluxVerificationStage.HASHING_ENC2)[index]
    val hasher = FluxStreamingHasher(object : FluxHashCache {
      override fun get(identity: FluxHashIdentity) = prefs.getString(identity.key, null)
      override fun put(identity: FluxHashIdentity, hash: String) {
        prefs.edit().putString(identity.key, hash).apply()
      }
    })
    return hasher.hash(identity, file::inputStream) { read ->
      mutableState.value = FluxVerificationUiState(true, stage, elapsed(started), read, file.length())
    }
  }

  private fun update(stage: FluxVerificationStage, started: Long) {
    mutableState.value = mutableState.value.copy(running = true, stage = stage, elapsedMillis = elapsed(started), hashBytes = 0, hashTotalBytes = 0)
  }

  companion object {
    private fun elapsed(started: Long) = (System.nanoTime() - started) / 1_000_000
    private fun usedHeap() = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
    private fun processPssKb(): Int? = runCatching { Debug.MemoryInfo().also(Debug::getMemoryInfo).totalPss }.getOrNull()
    private fun thermalStatus(context: Context): Int = if (Build.VERSION.SDK_INT >= 29) context.getSystemService(PowerManager::class.java).currentThermalStatus else -1
  }
}
