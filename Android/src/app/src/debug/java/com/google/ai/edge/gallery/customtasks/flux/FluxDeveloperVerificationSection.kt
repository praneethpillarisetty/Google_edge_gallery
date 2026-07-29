/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.net.Uri
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel

@Composable
internal fun FluxDeveloperVerificationSection(
  modelReady: Boolean,
  referenceUri: Uri?,
  viewModel: FluxVerificationViewModel = hiltViewModel(),
) {
  val state by viewModel.state.collectAsState()
  var prompt by remember { mutableStateOf("A red ceramic cup on a table") }
  val context = LocalContext.current
  Column(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
    Text("Developer verification — Text encoder only")
    Text("Warning: GPU FP32 verification may take several minutes and heat the device. Stop if the device becomes uncomfortable to hold.")
    OutlinedTextField(value = prompt, onValueChange = { prompt = it }, label = { Text("Non-sensitive test prompt") }, enabled = !state.running, modifier = Modifier.fillMaxWidth())
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      Button(onClick = { viewModel.run(prompt, modelReady) }, enabled = modelReady && !state.running) { Text("Run Text Encoder") }
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
    FluxReferenceVaeVerificationSection(modelReady, referenceUri)
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
    Text("Developer verification — Reference VAE only")
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      Button(onClick = { referenceUri?.let { viewModel.run(it, modelReady) } }, enabled = modelReady && referenceUri != null && !state.running) { Text("Verify Reference VAE") }
      OutlinedButton(onClick = viewModel::cancel, enabled = state.running) { Text("Cancel") }
    }
    Text("Current stage: ${state.stage}")
    Text("Elapsed: ${state.elapsedMillis} ms")
    if (state.running) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
    state.error?.let { Text(it) }
    if (state.summary.isNotEmpty()) {
      OutlinedTextField(state.summary, {}, readOnly = true, label = { Text("Sanitized VAE diagnostic") }, modifier = Modifier.fillMaxWidth())
      OutlinedButton(onClick = {
        context.getSystemService(ClipboardManager::class.java)
          .setPrimaryClip(ClipData.newPlainText("FLUX Reference VAE diagnostic", state.summary))
      }) { Text("Copy diagnostic summary") }
    }
  }
}
