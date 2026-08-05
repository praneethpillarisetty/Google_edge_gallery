/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.net.Uri
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.Button
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel

@Composable
internal fun FluxDeveloperVerificationSection(
  modelReady: Boolean,
  referenceUri: Uri?,
  editorPrompt: String,
  viewModel: FluxVerificationViewModel = hiltViewModel(),
) {
  val state by viewModel.state.collectAsState()
  var isolatedPrompt by remember { mutableStateOf("") }
  val context = LocalContext.current
  Column(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
    Text("Developer verification — Text encoder only")
    Text("Warning: GPU FP32 verification may take several minutes and heat the device. Stop if the device becomes uncomfortable to hold.")
    OutlinedTextField(value = isolatedPrompt, onValueChange = { isolatedPrompt = it }, label = { Text("Isolated text-encoder test prompt") }, enabled = !state.running, modifier = Modifier.fillMaxWidth())
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      Button(onClick = { viewModel.run(isolatedPrompt, modelReady) }, enabled = modelReady && isolatedPrompt.isNotBlank() && !state.running) { Text("Run Text Encoder") }
      OutlinedButton(onClick = viewModel::cancel, enabled = state.running) { Text("Cancel") }
    }
    Text("Current stage: ${state.stage}")
    Text("Elapsed: ${state.elapsedMillis} ms")
    if (state.hashTotalBytes > 0) {
      LinearProgressIndicator(progress = { state.hashBytes.toFloat() / state.hashTotalBytes }, modifier = Modifier.fillMaxWidth())
      Text("Hash progress: ${state.hashBytes} / ${state.hashTotalBytes} bytes")
    } else if (state.running) {
      LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
    }
    state.error?.let { Text(it) }
    if (state.summary.isNotEmpty()) {
      OutlinedTextField(value = state.summary, onValueChange = {}, readOnly = true, label = { Text("Sanitized diagnostic summary") }, modifier = Modifier.fillMaxWidth())
      OutlinedButton(onClick = {
        context.getSystemService(ClipboardManager::class.java).setPrimaryClip(ClipData.newPlainText("FLUX diagnostic", state.summary))
      }) { Text("Copy diagnostic summary") }
    }

    FluxPromptInfluenceComparisonSection(modelReady, referenceUri)
    FluxReferenceVaeVerificationSection(modelReady, referenceUri)
    FluxTransformerPrepVerificationSection(modelReady, referenceUri, editorPrompt)
    FluxTransformerDenoisingVerificationSection(modelReady, referenceUri, editorPrompt)
  }
}

@Composable
private fun FluxTransformerDenoisingVerificationSection(
  modelReady: Boolean,
  referenceUri: Uri?,
  prompt: String,
  viewModel: FluxTransformerDenoisingVerificationViewModel = hiltViewModel(),
) {
  val state by viewModel.state.collectAsState()
  val context = LocalContext.current
  Column(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
    Text("Developer verification — VAE decoder and image")
    Text("Runs the complete four-step GPU FP32 verification and displays the decoded image.")
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      Button(onClick = { referenceUri?.let { viewModel.run(it, prompt, modelReady) } }, enabled = modelReady && referenceUri != null && prompt.isNotBlank() && !state.running) { Text("Verify VAE Decoder and Image") }
      OutlinedButton(onClick = viewModel::cancel, enabled = state.running) { Text("Cancel") }
    }
    Text("Current stage: ${state.stage}")
    Text("Current step: ${state.currentStep}/4")
    Text("Current graph: ${state.currentGraph}")
    Text("Completed graphs: ${state.completedGraphs}/32")
    Text("Overall elapsed: ${state.elapsedMillis} ms")
    Text("Cancellation available: ${state.cancellationAvailable}")
    if (state.running) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
    state.error?.let { Text(it) }
    state.bitmap?.let { Image(it.asImageBitmap(), "Decoded FLUX verification image", modifier = Modifier.fillMaxWidth()) }
    if (state.summary.isNotEmpty()) {
      OutlinedTextField(state.summary, {}, readOnly = true, label = { Text("Sanitized denoising diagnostic") }, modifier = Modifier.fillMaxWidth())
      OutlinedButton(onClick = { context.getSystemService(ClipboardManager::class.java).setPrimaryClip(ClipData.newPlainText("FLUX denoising diagnostic", state.summary)) }) { Text("Copy diagnostic summary") }
    }
  }
}


@Composable
private fun FluxPromptInfluenceComparisonSection(
  modelReady: Boolean,
  referenceUri: Uri?,
  viewModel: FluxPromptInfluenceComparisonViewModel = hiltViewModel(),
) {
  val state by viewModel.state.collectAsState()
  val context = LocalContext.current
  var promptA by remember { mutableStateOf("Preserve the same adult person and facial identity. Replace the outfit with a bright red formal evening gown. Place the person on a sunny tropical beach.") }
  var promptB by remember { mutableStateOf("Preserve the same adult person and facial identity. Replace the outfit with a black winter coat. Place the person in a snowy mountain landscape at night.") }
  var seed by remember { mutableStateOf("1234") }
  Column(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
    Text("Developer verification — Prompt influence")
    Text("Runs Prompt A then Prompt B sequentially with the same staged reference and identical fixed-seed initial noise. Summaries omit prompt text and image identifiers.")
    OutlinedTextField(promptA, { promptA = it }, label = { Text("Prompt A") }, enabled = !state.running, modifier = Modifier.fillMaxWidth(), minLines = 2)
    OutlinedTextField(promptB, { promptB = it }, label = { Text("Prompt B") }, enabled = !state.running, modifier = Modifier.fillMaxWidth(), minLines = 2)
    OutlinedTextField(seed, { seed = it }, label = { Text("Fixed seed (signed 64-bit)") }, enabled = !state.running, modifier = Modifier.fillMaxWidth(), singleLine = true)
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      Button(onClick = { viewModel.run(referenceUri, promptA, promptB, seed, modelReady) }, enabled = modelReady && referenceUri != null && promptA.isNotBlank() && promptB.isNotBlank() && seed.isNotBlank() && !state.running) { Text("Run comparison") }
      OutlinedButton(onClick = viewModel::cancel, enabled = state.running) { Text("Cancel") }
    }
    Text("Current run: ${state.runLabel}")
    Text("Current stage: ${state.stage}")
    Text("Denoising step: ${state.currentStep}/4")
    Text("Completed graphs: ${state.completedGraphs}/32")
    Text("Elapsed: ${state.elapsedMillis} ms")
    if (state.running) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
    state.error?.let { Text(it) }
    if (state.summary.isNotEmpty()) {
      OutlinedTextField(state.summary, {}, readOnly = true, label = { Text("Sanitized prompt-influence summary") }, modifier = Modifier.fillMaxWidth())
      OutlinedButton(onClick = { context.getSystemService(ClipboardManager::class.java).setPrimaryClip(ClipData.newPlainText("FLUX prompt influence diagnostic", state.summary)) }) { Text("Copy sanitized diagnostic summary") }
    }
    state.bitmapA?.let { Image(it.asImageBitmap(), "Prompt A comparison output thumbnail", modifier = Modifier.fillMaxWidth()) }
    state.bitmapB?.let { Image(it.asImageBitmap(), "Prompt B comparison output thumbnail", modifier = Modifier.fillMaxWidth()) }
  }
}

/** Phase 2F remains a deliberately separate debug-only terminal boundary. */
@Composable
private fun FluxTransformerPrepVerificationSection(
  modelReady: Boolean,
  referenceUri: Uri?,
  prompt: String,
  viewModel: FluxTransformerPrepVerificationViewModel = hiltViewModel(),
) {
  val state by viewModel.state.collectAsState()
  val context = LocalContext.current
  Column(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
    Text("Developer verification — Editing transformer prep only")
    Text("Uses synthetic zero noise and a synthetic zero timestep embedding; it does not generate an image.")
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      Button(
        onClick = { referenceUri?.let { viewModel.run(it, prompt, modelReady) } },
        enabled = modelReady && referenceUri != null && prompt.isNotBlank() && !state.running,
      ) { Text("Run Transformer Prep") }
      OutlinedButton(onClick = viewModel::cancel, enabled = state.running) { Text("Cancel") }
    }
    Text("Current stage: ${state.stage}")
    Text("Elapsed: ${state.elapsedMillis} ms")
    if (state.running) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
    state.error?.let { Text(it) }
    if (state.summary.isNotEmpty()) {
      OutlinedTextField(state.summary, {}, readOnly = true, label = { Text("Sanitized transformer-prep diagnostic") }, modifier = Modifier.fillMaxWidth())
      OutlinedButton(onClick = {
        context.getSystemService(ClipboardManager::class.java).setPrimaryClip(ClipData.newPlainText("FLUX transformer-prep diagnostic", state.summary))
      }) { Text("Copy diagnostic summary") }
    }
  }
}

@Composable
private fun FluxReferenceVaeVerificationSection(
  modelReady: Boolean,
  referenceUri: Uri?,
  viewModel: FluxReferenceVaeVerificationViewModel = hiltViewModel(),
) {
  val state by viewModel.state.collectAsState()
  val context = LocalContext.current
  Column(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
    Text("Developer verification — Reference VAE and tokens only")
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      Button(onClick = { referenceUri?.let { viewModel.run(it, modelReady) } }, enabled = modelReady && referenceUri != null && !state.running) { Text("Verify Reference VAE + Tokens") }
      OutlinedButton(onClick = viewModel::cancel, enabled = state.running) { Text("Cancel") }
    }
    Text("Current stage: ${state.stage}")
    Text("Elapsed: ${state.elapsedMillis} ms")
    if (state.running) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
    state.error?.let { Text(it) }
    if (state.summary.isNotEmpty()) {
      OutlinedTextField(state.summary, {}, readOnly = true, label = { Text("Sanitized VAE/token diagnostic") }, modifier = Modifier.fillMaxWidth())
      OutlinedButton(onClick = {
        context.getSystemService(ClipboardManager::class.java)
          .setPrimaryClip(ClipData.newPlainText("FLUX Reference VAE/token diagnostic", state.summary))
      }) { Text("Copy diagnostic summary") }
    }
  }
}
