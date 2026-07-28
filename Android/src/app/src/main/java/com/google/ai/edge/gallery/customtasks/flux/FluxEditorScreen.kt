/* Copyright 2025 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.google.ai.edge.gallery.R

@Composable
fun FluxEditorScreen(viewModel: FluxEditorViewModel = hiltViewModel()) {
  val state by viewModel.uiState.collectAsState()
  val context = LocalContext.current
  var imageUri by remember { mutableStateOf<Uri?>(null) }
  var prompt by remember { mutableStateOf("") }
  val picker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { imageUri = it }
  val bitmap = remember(imageUri) { imageUri?.let { context.contentResolver.openInputStream(it)?.use(BitmapFactory::decodeStream) } }
  val totalBytes = when (state) {
    is FluxEditorUiState.NotInstalled -> (state as FluxEditorUiState.NotInstalled).totalBytes
    is FluxEditorUiState.Downloading -> (state as FluxEditorUiState.Downloading).files.sumOf { it.total }
    is FluxEditorUiState.Ready -> (state as FluxEditorUiState.Ready).totalBytes
    else -> 0L
  }

  Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
    Text(stringResource(R.string.flux_phase_one_notice))
    OutlinedButton(onClick = { picker.launch("image/*") }, modifier = Modifier.fillMaxWidth()) { Text(stringResource(R.string.flux_pick_reference)) }
    bitmap?.let { Image(it.asImageBitmap(), stringResource(R.string.flux_reference_preview_description), Modifier.fillMaxWidth().height(220.dp), contentScale = ContentScale.Fit) }
    OutlinedTextField(prompt, { prompt = it }, Modifier.fillMaxWidth(), label = { Text(stringResource(R.string.flux_edit_prompt)) }, minLines = 3)
    Text(stringResource(R.string.flux_model_status, statusText(state)))
    Text(if (totalBytes > 0) stringResource(R.string.flux_storage_requirement, DefaultFluxDownloadRepository.formatBytes(totalBytes), DefaultFluxDownloadRepository.formatBytes(totalBytes + DefaultFluxDownloadRepository.SAFETY_MARGIN)) else stringResource(R.string.flux_storage_checking))
    when (val current = state) {
      FluxEditorUiState.Checking -> CircularProgressIndicator()
      is FluxEditorUiState.NotInstalled -> Button(viewModel::download) { Text(stringResource(R.string.flux_download_models)) }
      is FluxEditorUiState.Downloading -> {
        val received = current.files.sumOf { it.received }
        LinearProgressIndicator({ if (totalBytes == 0L) 0f else received.toFloat() / totalBytes }, Modifier.fillMaxWidth())
        current.files.forEach { file ->
          Text("${file.path}: ${DefaultFluxDownloadRepository.formatBytes(file.received)} / ${DefaultFluxDownloadRepository.formatBytes(file.total)}")
          LinearProgressIndicator({ file.received.toFloat() / file.total }, Modifier.fillMaxWidth())
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
          OutlinedButton(viewModel::pause) { Text(stringResource(R.string.flux_pause)) }
          OutlinedButton(viewModel::cancel) { Text(stringResource(R.string.cancel)) }
        }
      }
      FluxEditorUiState.Paused -> Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(viewModel::retry) { Text(stringResource(R.string.flux_retry)) }
        OutlinedButton(viewModel::cancel) { Text(stringResource(R.string.cancel)) }
      }
      is FluxEditorUiState.Error -> {
        Text(current.message)
        Button(viewModel::refresh) { Text(stringResource(R.string.flux_retry)) }
      }
      is FluxEditorUiState.Ready -> Text(stringResource(R.string.flux_model_ready))
    }
    Button(onClick = {}, enabled = false, modifier = Modifier.fillMaxWidth()) {
      Text(stringResource(R.string.flux_generate))
    }
    Text(stringResource(R.string.flux_generation_phase_two_explanation))
  }
}

@Composable private fun statusText(state: FluxEditorUiState) = when (state) {
  is FluxEditorUiState.NotInstalled -> stringResource(R.string.flux_not_installed)
  FluxEditorUiState.Checking -> stringResource(R.string.flux_checking)
  is FluxEditorUiState.Downloading -> stringResource(R.string.flux_downloading)
  FluxEditorUiState.Paused -> stringResource(R.string.flux_paused)
  is FluxEditorUiState.Ready -> stringResource(R.string.flux_ready)
  is FluxEditorUiState.Error -> stringResource(R.string.error)
}
