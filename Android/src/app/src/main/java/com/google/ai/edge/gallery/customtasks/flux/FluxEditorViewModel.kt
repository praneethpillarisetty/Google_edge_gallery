/* Copyright 2025 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

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
class FluxEditorViewModel @Inject constructor(private val repository: FluxDownloadRepository) :
  ViewModel() {
  private val mutableUiState = MutableStateFlow<FluxEditorUiState>(FluxEditorUiState.Checking)
  val uiState: StateFlow<FluxEditorUiState> = mutableUiState.asStateFlow()
  private var knownTotalBytes: Long = 0

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
}
