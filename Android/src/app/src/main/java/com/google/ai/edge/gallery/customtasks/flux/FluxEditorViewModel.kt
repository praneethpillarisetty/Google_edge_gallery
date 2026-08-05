/* Copyright 2025 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import android.graphics.Bitmap
import android.net.Uri
import android.os.PowerManager
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import dagger.hilt.android.qualifiers.ApplicationContext
import android.content.Context
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxGenerationException
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxGenerationProgress
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxGenerationStage
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxImageEditGenerator
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxSeedParser
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxSeedSelection
import com.google.ai.edge.gallery.customtasks.flux.output.FluxGeneratedImageStore
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job

data class FluxProductionGenerationState(
  val running: Boolean = false,
  val stage: FluxGenerationStage = FluxGenerationStage.IDLE,
  val currentStep: Int = 0,
  val currentGraph: String? = null,
  val completedGraphCount: Int = 0,
  val elapsedMillis: Long = 0,
  val cancellationAvailable: Boolean = false,
  val sanitizedError: String? = null,
  val finalBitmap: Bitmap? = null,
  val savedOutputUri: Uri? = null,
  val seedSelection: FluxSeedSelection = FluxSeedSelection.Random,
  val fixedSeedText: String = "",
  val actualSeed: Long? = null,
)

fun fluxGenerateReady(repositoryReady: Boolean, reference: Uri?, prompt: String, running: Boolean, thermalStatus: Int) =
  repositoryReady && reference != null && prompt.isNotBlank() && !running &&
    thermalStatus < PowerManager.THERMAL_STATUS_EMERGENCY

sealed interface FluxEditorUiState {
  data class NotInstalled(val totalBytes: Long) : FluxEditorUiState
  data object Checking : FluxEditorUiState
  data class Downloading(val files: List<FluxFileProgress>) : FluxEditorUiState
  data object Paused : FluxEditorUiState
  data class Ready(val totalBytes: Long) : FluxEditorUiState
  data class Error(val message: String) : FluxEditorUiState
}

fun reduceFluxState(event: FluxDownloadEvent, knownTotalBytes: Long): FluxEditorUiState =
  when (event) {
    FluxDownloadEvent.Checking -> FluxEditorUiState.Checking
    is FluxDownloadEvent.NotInstalled ->
      FluxEditorUiState.NotInstalled(event.totalBytes ?: knownTotalBytes)
    is FluxDownloadEvent.Downloading -> FluxEditorUiState.Downloading(event.files)
    FluxDownloadEvent.Paused -> FluxEditorUiState.Paused
    is FluxDownloadEvent.Ready -> FluxEditorUiState.Ready(event.totalBytes)
    is FluxDownloadEvent.Error -> FluxEditorUiState.Error(event.message)
  }

@HiltViewModel
class FluxEditorViewModel @Inject constructor(
  private val repository: FluxDownloadRepository,
  private val generator: FluxImageEditGenerator,
  val imageStore: FluxGeneratedImageStore,
  @ApplicationContext context: Context,
) :
  ViewModel() {
  private val mutableUiState = MutableStateFlow<FluxEditorUiState>(FluxEditorUiState.Checking)
  val uiState: StateFlow<FluxEditorUiState> = mutableUiState.asStateFlow()
  private var knownTotalBytes: Long = 0
  private val powerManager = context.getSystemService(PowerManager::class.java)
  private val mutableGenerationState = MutableStateFlow(FluxProductionGenerationState())
  val generationState = mutableGenerationState.asStateFlow()
  private var generationJob: Job? = null

  init {
    viewModelScope.launch {
      repository.events.collect { event ->
        if (event is FluxDownloadEvent.NotInstalled && event.totalBytes != null) {
          knownTotalBytes = event.totalBytes
        }
        if (event is FluxDownloadEvent.Downloading) knownTotalBytes = event.files.sumOf { it.total }
        if (event is FluxDownloadEvent.Ready) knownTotalBytes = event.totalBytes
        mutableUiState.value = reduceFluxState(event, knownTotalBytes)
      }
    }
    refresh()
  }

  fun refresh() { viewModelScope.launch { repository.check() } }
  fun download() { viewModelScope.launch { repository.download() } }
  fun pause() = repository.pause()
  fun retry() = download()
  fun cancel() { viewModelScope.launch { repository.cancel() } }

  fun canGenerate(reference: Uri?, prompt: String): Boolean = fluxGenerateReady(
    mutableUiState.value is FluxEditorUiState.Ready, reference, prompt,
    generationJob?.isActive == true, powerManager.currentThermalStatus,
  )

  fun setSeedMode(selection: FluxSeedSelection) { mutableGenerationState.value = mutableGenerationState.value.copy(seedSelection = selection, sanitizedError = null) }
  fun setFixedSeedText(text: String) { mutableGenerationState.value = mutableGenerationState.value.copy(fixedSeedText = text, sanitizedError = null) }
  fun randomizeSeed() { val value = java.security.SecureRandom().nextLong(); mutableGenerationState.value = mutableGenerationState.value.copy(seedSelection = FluxSeedSelection.Fixed(value), fixedSeedText = value.toString(), sanitizedError = null) }

  fun generate(reference: Uri?, prompt: String) {
    if (!canGenerate(reference, prompt) || generationJob?.isActive == true) return
    val selected = reference ?: return
    generationJob = viewModelScope.launch {
      val started = System.nanoTime()
      val previous = mutableGenerationState.value.finalBitmap
      try {
        val seedSelection = when (val current = mutableGenerationState.value.seedSelection) {
          FluxSeedSelection.Random -> FluxSeedSelection.Random
          is FluxSeedSelection.Fixed -> FluxSeedParser.parseFixed(mutableGenerationState.value.fixedSeedText)
        }
        val result = generator.generate(selected, prompt, seedSelection) { progress: FluxGenerationProgress ->
          mutableGenerationState.value = mutableGenerationState.value.copy(
            running = true, stage = progress.stage, currentStep = progress.currentStep,
            currentGraph = progress.currentGraph, completedGraphCount = progress.completedGraphCount,
            elapsedMillis = (System.nanoTime() - started) / 1_000_000, cancellationAvailable = true,
            sanitizedError = null, savedOutputUri = null,
          )
        }
        mutableGenerationState.value = FluxProductionGenerationState(
          stage = FluxGenerationStage.COMPLETE, currentStep = 4, completedGraphCount = 32,
          elapsedMillis = result.elapsedMillis, finalBitmap = result.bitmap, actualSeed = result.seed,
          seedSelection = mutableGenerationState.value.seedSelection, fixedSeedText = mutableGenerationState.value.fixedSeedText,
        )
      } catch (_: CancellationException) {
        mutableGenerationState.value = FluxProductionGenerationState(
          stage = FluxGenerationStage.CANCELLED, elapsedMillis = (System.nanoTime() - started) / 1_000_000,
          sanitizedError = "Generation cancelled.", finalBitmap = previous,
        )
      } catch (invalid: IllegalArgumentException) {
        mutableGenerationState.value = mutableGenerationState.value.copy(running = false, stage = FluxGenerationStage.ERROR,
          elapsedMillis = (System.nanoTime() - started) / 1_000_000, sanitizedError = invalid.message ?: "Invalid fixed seed.", finalBitmap = previous)
      } catch (failure: FluxGenerationException) {
        mutableGenerationState.value = FluxProductionGenerationState(
          stage = FluxGenerationStage.ERROR, elapsedMillis = (System.nanoTime() - started) / 1_000_000,
          sanitizedError = failure.category.userMessage, finalBitmap = previous,
        )
      }
    }
  }

  fun cancelGeneration() { generationJob?.cancel() }
  fun saved(uri: Uri) { mutableGenerationState.value = mutableGenerationState.value.copy(savedOutputUri = uri, sanitizedError = null) }
  fun outputError(message: String) { mutableGenerationState.value = mutableGenerationState.value.copy(sanitizedError = message) }
  override fun onCleared() { generationJob?.cancel(); super.onCleared() }
}
