/* Copyright 2025 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import android.content.Intent
import android.net.Uri
import android.widget.ImageView
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.weight
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.ui.window.Dialog
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.google.ai.edge.gallery.R
import kotlinx.coroutines.launch

@Composable
fun FluxEditorScreen(viewModel: FluxEditorViewModel = hiltViewModel()) {
  val state by viewModel.uiState.collectAsState()
  val generation by viewModel.generationState.collectAsState()
  var imageUri by remember { mutableStateOf<Uri?>(null) }
  var prompt by remember { mutableStateOf("") }
  val referencePreviewDescription =
    stringResource(R.string.flux_reference_preview_description)
  val resolver = LocalContext.current.contentResolver
  val context = LocalContext.current
  val scope = rememberCoroutineScope()
  var viewerOpen by remember { mutableStateOf(false) }
  val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { selected ->
    selected?.let {
      try {
        resolver.takePersistableUriPermission(it, Intent.FLAG_GRANT_READ_URI_PERMISSION)
      } catch (_: SecurityException) {
        // Some document providers grant read access only for the current process.
      }
    }
    imageUri = selected
  }
  val totalBytes = when (state) {
    is FluxEditorUiState.NotInstalled -> (state as FluxEditorUiState.NotInstalled).totalBytes
    is FluxEditorUiState.Downloading -> (state as FluxEditorUiState.Downloading).files.sumOf { it.total }
    is FluxEditorUiState.Ready -> (state as FluxEditorUiState.Ready).totalBytes
    else -> 0L
  }

  Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
    Text(stringResource(R.string.flux_phase_one_notice))
    OutlinedButton(onClick = { picker.launch(arrayOf("image/*")) }, enabled = !generation.running, modifier = Modifier.fillMaxWidth()) { Text(stringResource(R.string.flux_pick_reference)) }
    imageUri?.let { selected ->
      AndroidView(
        factory = { viewContext ->
          ImageView(viewContext).apply { scaleType = ImageView.ScaleType.FIT_CENTER }
        },
        update = { imageView ->
          imageView.contentDescription = referencePreviewDescription
          imageView.setImageURI(selected)
        },
        modifier = Modifier.fillMaxWidth().height(220.dp),
      )
    }
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
    Text(if (imageUri == null) "Select a reference image to edit. Text-to-image requires an additional model package." else "Mode: Image edit")
    Button(onClick = { viewModel.generate(imageUri, prompt) }, enabled = viewModel.canGenerate(imageUri, prompt), modifier = Modifier.fillMaxWidth()) {
      Text(stringResource(R.string.flux_generate))
    }
    if (generation.running) {
      LinearProgressIndicator(Modifier.fillMaxWidth())
      Text("Current stage: ${generation.stage}")
      Text("Denoising step: ${generation.currentStep}/4")
      generation.currentGraph?.let { Text("Current graph: $it") }
      Text("Completed graphs: ${generation.completedGraphCount}/32")
      Text("Elapsed: ${generation.elapsedMillis} ms")
      OutlinedButton(viewModel::cancelGeneration) { Text("Cancel generation") }
    }
    generation.sanitizedError?.let { Text(it) }
    generation.finalBitmap?.let { bitmap ->
      Text("Generated image")
      Image(bitmap.asImageBitmap(), "Generated FLUX image; tap to open fullscreen", Modifier.fillMaxWidth().clickable { viewerOpen = true })
      Text("Original dimensions: ${bitmap.width}×${bitmap.height}")
      Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(onClick = { scope.launch { runCatching { viewModel.imageStore.savePng(bitmap) }.onSuccess(viewModel::saved).onFailure { viewModel.outputError("The PNG could not be saved.") } } }) { Text("Save PNG") }
        Button(onClick = { scope.launch {
          runCatching { generation.savedOutputUri ?: viewModel.imageStore.savePng(bitmap) }.onSuccess { uri ->
            viewModel.saved(uri); val intent = viewModel.imageStore.shareIntent(uri)
            if (viewModel.imageStore.canShare(intent)) context.startActivity(Intent.createChooser(intent, "Share FLUX image")) else viewModel.outputError("No compatible sharing app is available.")
          }.onFailure { viewModel.outputError("The image could not be shared.") }
        } }) { Text("Share") }
      }
      Button(onClick = { scope.launch { runCatching { viewModel.imageStore.saveResized1024(bitmap) }.onSuccess(viewModel::saved).onFailure { viewModel.outputError("The 1024×1024 resized export failed.") } } }) { Text("Export 1024×1024") }
      Text("1024×1024 resized export — filtered resize, not AI super-resolution")
      if (viewerOpen) FluxFullscreenViewer(bitmap = bitmap, onClose = { viewerOpen = false })
    }
    generation.savedOutputUri?.let { Text("Saved content URI: $it") }
    FluxDeveloperVerificationSection(modelReady = state is FluxEditorUiState.Ready, referenceUri = imageUri, editorPrompt = prompt)
  }
}

@Composable private fun FluxFullscreenViewer(bitmap: android.graphics.Bitmap, onClose: () -> Unit) {
  var scale by remember { mutableStateOf(1f) }
  var offsetX by remember { mutableStateOf(0f) }
  var offsetY by remember { mutableStateOf(0f) }
  Dialog(onDismissRequest = onClose) {
    Column(Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
      Text("Generated image — 256×256")
      Image(bitmap.asImageBitmap(), "Fullscreen generated FLUX image", Modifier.weight(1f).fillMaxWidth()
        .pointerInput(Unit) { detectTransformGestures { _, pan, zoom, _ -> scale = (scale * zoom).coerceIn(1f, 8f); offsetX += pan.x; offsetY += pan.y } }
        .graphicsLayer(scaleX = scale, scaleY = scale, translationX = offsetX, translationY = offsetY))
      Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedButton(onClick = { scale = 1f; offsetX = 0f; offsetY = 0f }) { Text("Reset") }
        Button(onClick = onClose) { Text("Close") }
      }
    }
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
