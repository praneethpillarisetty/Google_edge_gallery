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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Checkbox
import androidx.compose.material3.AlertDialog
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
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxSeedSelection
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxEditMode
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxFigureAction
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxFigureEditRequest
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxSimpleEditRequest
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxBuiltInFigurePresets
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxFigureAttributes
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxFigurePreset
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxRealismProfile
import kotlinx.coroutines.launch

@Composable
fun FluxEditorScreen(viewModel: FluxEditorViewModel = hiltViewModel()) {
  val state by viewModel.uiState.collectAsState()
  val generation by viewModel.generationState.collectAsState()
  val userPresets by viewModel.userPresets.collectAsState()
  var imageUri by remember { mutableStateOf<Uri?>(generation.iterativeReference) }
  var originalUri by remember { mutableStateOf<Uri?>(null) }
  var instruction by remember { mutableStateOf("") }
  var mode by remember { mutableStateOf(FluxEditMode.SIMPLE) }
  var advanced by remember { mutableStateOf(false) }
  var realismProfile by remember { mutableStateOf(FluxRealismProfile.NATURAL_PHOTO) }
  var preserveIdentity by remember { mutableStateOf(true) }; var preservePoseSimple by remember { mutableStateOf(true) }; var preserveBackgroundSimple by remember { mutableStateOf(false) }
  var figureAction by remember { mutableStateOf<FluxFigureAction>(FluxFigureAction.PreserveCurrent) }
  var presetIndex by remember { mutableStateOf(1) }; var figureDescription by remember { mutableStateOf(FluxBuiltInFigurePresets.all[presetIndex].attributes.description()) }
  var customPresetName by remember { mutableStateOf("My figure preset") }; var selectedUserPresetId by remember { mutableStateOf<String?>(null) }
  var preserveOutfit by remember { mutableStateOf(true) }; var preservePose by remember { mutableStateOf(true) }; var preserveBackground by remember { mutableStateOf(true) }; var preserveCamera by remember { mutableStateOf(true) }
  var preserveLighting by remember { mutableStateOf(true) }; var preserveHair by remember { mutableStateOf(true) }; var preserveMakeup by remember { mutableStateOf(true) }; var preserveAccessories by remember { mutableStateOf(true) }
  var conflicts by remember { mutableStateOf<List<String>>(emptyList()) }; var viewerOpen by remember { mutableStateOf(false) }; var showAfter by remember { mutableStateOf(true) }
  var confirmDifferentSeed by remember { mutableStateOf(false) }
  val context = LocalContext.current; val resolver = context.contentResolver; val scope = rememberCoroutineScope()
  val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { selected ->
    selected?.let {
      try {
        resolver.takePersistableUriPermission(it, Intent.FLAG_GRANT_READ_URI_PERMISSION)
      } catch (_: SecurityException) {
        // Some providers grant a process-lifetime read permission instead.
      }
      imageUri = selected
      originalUri = selected
    }
  }
  fun figureRequest() = FluxFigureEditRequest(figureAction, figureDescription, preserveOutfit, preservePose, preserveBackground, preserveCamera, preserveLighting, preserveHair, preserveMakeup, preserveAccessories, instruction, realismProfile)
  fun runGeneration(unlock: Boolean = false) {
    runCatching {
      if (mode == FluxEditMode.SIMPLE) {
        val compiled = viewModel.compile(FluxSimpleEditRequest(instruction, preserveIdentity, preservePoseSimple, preserveBackgroundSimple, realismProfile))
        viewModel.generate(imageUri, compiled.positivePrompt, mode, instruction)
      } else {
        var request = figureRequest(); val compiled = viewModel.compile(request, figureDescription)
        if (compiled.conflicts.isNotEmpty() && !unlock) { conflicts = compiled.conflicts.map { it.lockName }; return }
        if (unlock) request = request.copy(
          preserveOutfit = preserveOutfit && "outfit" !in conflicts && "accessories" !in conflicts,
          preservePose = preservePose && "pose and hand placement" !in conflicts,
          preserveBackground = preserveBackground && "location and background" !in conflicts,
          preserveCamera = preserveCamera && "framing and camera" !in conflicts,
          preserveLighting = preserveLighting && "lighting and shadows" !in conflicts,
          preserveHair = preserveHair && "hairstyle" !in conflicts,
          preserveMakeupAndExpression = preserveMakeup && "makeup and expression" !in conflicts,
          preserveAccessories = preserveAccessories && "accessories" !in conflicts)
        val resolved = viewModel.compile(request, figureDescription)
        viewModel.generate(imageUri, resolved.positivePrompt, mode, instruction, FluxBuiltInFigurePresets.all.getOrNull(presetIndex)?.name)
        conflicts = emptyList()
      }
    }.onFailure { viewModel.outputError(it.message ?: "The edit instruction is invalid.") }
  }
  Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
    Text("Local FLUX image editor")
    Text("Model status: ${statusText(state)}")
    when (state) {
      FluxEditorUiState.Checking -> CircularProgressIndicator()
      is FluxEditorUiState.NotInstalled -> Button(viewModel::download) { Text("Download models") }
      is FluxEditorUiState.Downloading -> {
        LinearProgressIndicator(Modifier.fillMaxWidth())
        Row { OutlinedButton(viewModel::pause) { Text("Pause") }; OutlinedButton(viewModel::cancel) { Text("Cancel download") } }
      }
      FluxEditorUiState.Paused -> Row { Button(viewModel::retry) { Text("Resume") }; OutlinedButton(viewModel::cancel) { Text("Cancel download") } }
      is FluxEditorUiState.Error -> { Text((state as FluxEditorUiState.Error).message); Button(viewModel::refresh) { Text("Retry") } }
      is FluxEditorUiState.Ready -> Text("Models ready")
    }
    OutlinedButton({ picker.launch(arrayOf("image/*")) }, enabled = !generation.running, modifier = Modifier.fillMaxWidth()) { Text("Select reference image") }
    imageUri?.let { uri -> AndroidView(factory = { ImageView(it).apply { scaleType = ImageView.ScaleType.FIT_CENTER } }, update = { it.setImageURI(uri); it.contentDescription = "Selected reference image" }, modifier = Modifier.fillMaxWidth().height(220.dp)) }
    if (imageUri == null) Text("This local FLUX model requires a reference image. Text-to-image is not available with the installed model package.")
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      OutlinedButton({ mode = FluxEditMode.SIMPLE }, enabled = !generation.running) { Text(if (mode == FluxEditMode.SIMPLE) "Simple Edit ✓" else "Simple Edit") }
      OutlinedButton({ mode = FluxEditMode.FIGURE }, enabled = !generation.running) { Text(if (mode == FluxEditMode.FIGURE) "Figure Edit ✓" else "Figure Edit") }
    }
    if (mode == FluxEditMode.SIMPLE) OutlinedTextField(instruction, { instruction = it }, Modifier.fillMaxWidth(), label = { Text("Describe your edit") }, minLines = 3)
    else {
      Text("Figure action")
      Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        OutlinedButton({ figureAction = FluxFigureAction.PreserveCurrent }) { Text("Preserve current") }
        OutlinedButton({ figureAction = FluxFigureAction.ApplyPreset(FluxBuiltInFigurePresets.all[presetIndex].id) }) { Text("Apply preset") }
        OutlinedButton({ figureAction = FluxFigureAction.Custom(figureDescription) }) { Text("Custom") }
      }
      Text("Figure preset: ${FluxBuiltInFigurePresets.all[presetIndex].name}")
      Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        OutlinedButton({ presetIndex = (presetIndex - 1).coerceAtLeast(0); figureDescription = FluxBuiltInFigurePresets.all[presetIndex].attributes.description(); figureAction = FluxFigureAction.ApplyPreset(FluxBuiltInFigurePresets.all[presetIndex].id) }) { Text("Previous") }
        OutlinedButton({ presetIndex = (presetIndex + 1).coerceAtMost(FluxBuiltInFigurePresets.all.lastIndex); figureDescription = FluxBuiltInFigurePresets.all[presetIndex].attributes.description(); figureAction = FluxFigureAction.ApplyPreset(FluxBuiltInFigurePresets.all[presetIndex].id) }) { Text("Next") }
      }
      OutlinedTextField(figureDescription, { figureDescription = it; if (figureAction is FluxFigureAction.Custom) figureAction = FluxFigureAction.Custom(it) }, Modifier.fillMaxWidth(), label = { Text("Editable figure attributes: build, shoulders, torso, waist, hips, legs, height") }, minLines = 3)
      OutlinedTextField(customPresetName, { customPresetName = it }, Modifier.fillMaxWidth(), label = { Text("Custom preset name") })
      Row {
        OutlinedButton({
          val id = selectedUserPresetId ?: "user.${System.currentTimeMillis()}"
          viewModel.savePreset(FluxFigurePreset(id, customPresetName.trim(), FluxFigureAttributes(overallBuild = figureDescription), false)); selectedUserPresetId = id
        }, enabled = customPresetName.isNotBlank() && figureDescription.isNotBlank()) { Text(if (selectedUserPresetId == null) "Duplicate/save preset" else "Update preset") }
        selectedUserPresetId?.let { id -> OutlinedButton({ viewModel.deletePreset(id); selectedUserPresetId = null }) { Text("Delete custom preset") } }
      }
      userPresets.forEach { saved -> OutlinedButton({ selectedUserPresetId = saved.id; customPresetName = saved.name; figureDescription = saved.attributes.description(); figureAction = FluxFigureAction.Custom(figureDescription) }) { Text("Use ${saved.name}") } }
      Text("Preservation locks")
      Lock("Preserve outfit", preserveOutfit) { preserveOutfit = it }; Lock("Preserve pose", preservePose) { preservePose = it }; Lock("Preserve location/background", preserveBackground) { preserveBackground = it }; Lock("Preserve framing/camera", preserveCamera) { preserveCamera = it }
      Lock("Preserve lighting", preserveLighting) { preserveLighting = it }; Lock("Preserve hairstyle", preserveHair) { preserveHair = it }; Lock("Preserve makeup/expression", preserveMakeup) { preserveMakeup = it }; Lock("Preserve accessories", preserveAccessories) { preserveAccessories = it }
      OutlinedTextField(instruction, { instruction = it }, Modifier.fillMaxWidth(), label = { Text("Additional instruction") })
      Text("Figure controls are natural-language instructions, not exact geometric constraints. Major body changes may require small clothing-fit adjustments.")
      Text("Realism and anatomy controls guide the model through natural-language instructions. They reduce common structural errors but cannot guarantee perfect hands, faces, limbs or identity preservation.")
    }
    OutlinedButton({ advanced = !advanced }, Modifier.fillMaxWidth()) { Text(if (advanced) "Hide Advanced generation" else "Advanced generation") }
    if (advanced) {
      Text("Realism")
      Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        FluxRealismProfile.entries.forEach { profile ->
          val label = when (profile) { FluxRealismProfile.NATURAL_PHOTO -> "Natural photo"; FluxRealismProfile.EDITORIAL_PHOTO -> "Editorial photo"; FluxRealismProfile.CINEMATIC_PHOTO -> "Cinematic photo" }
          OutlinedButton({ realismProfile = profile }, enabled = !generation.running) { Text(if (realismProfile == profile) "$label ✓" else label) }
        }
      }
      Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { OutlinedButton({ viewModel.setSeedMode(FluxSeedSelection.Random) }) { Text("Random seed") }; OutlinedButton({ viewModel.setSeedMode(FluxSeedSelection.Fixed(generation.fixedSeedText.toLongOrNull() ?: 0)) }) { Text("Fixed seed") }; OutlinedButton(viewModel::randomizeSeed) { Text("Randomize seed") } }
      if (generation.seedSelection is FluxSeedSelection.Fixed) OutlinedTextField(generation.fixedSeedText, viewModel::setFixedSeedText, label = { Text("Fixed seed (signed 64-bit)") })
      if (mode == FluxEditMode.SIMPLE) { Lock("Preserve facial identity", preserveIdentity) { preserveIdentity = it }; Lock("Preserve pose/composition", preservePoseSimple) { preservePoseSimple = it }; Lock("Preserve background", preserveBackgroundSimple) { preserveBackgroundSimple = it } }
      Text("These options are prompt instructions, not guaranteed numerical controls.")
    }
    generation.actualSeed?.let { seed -> Row { Text("Actual seed used: $seed"); OutlinedButton({ context.getSystemService(android.content.ClipboardManager::class.java).setPrimaryClip(android.content.ClipData.newPlainText("FLUX seed", seed.toString())) }) { Text("Copy seed") }; OutlinedButton({ viewModel.setSeedMode(FluxSeedSelection.Fixed(seed)); viewModel.setFixedSeedText(seed.toString()) }) { Text("Use same seed again") } } }
    Button({ runGeneration() }, enabled = viewModel.canGenerate(imageUri, if (mode == FluxEditMode.SIMPLE) instruction else "figure edit"), modifier = Modifier.fillMaxWidth()) { Text("Generate") }
    if (generation.running) { LinearProgressIndicator(Modifier.fillMaxWidth()); Text("${generation.stage}: step ${generation.currentStep}/4 — ${generation.elapsedMillis} ms"); OutlinedButton(viewModel::cancelGeneration) { Text("Cancel") } }
    generation.sanitizedError?.let { Text(it) }
    generation.finalBitmap?.let { bitmap ->
      Text("Before / After")
      Row { OutlinedButton({ showAfter = false }) { Text("Before") }; OutlinedButton({ showAfter = true }) { Text("After") } }
      if (showAfter) Image(bitmap.asImageBitmap(), "Generated FLUX image", Modifier.fillMaxWidth().clickable { viewerOpen = true }) else imageUri?.let { uri -> AndroidView(factory = { ImageView(it) }, update = { it.setImageURI(uri) }, modifier = Modifier.fillMaxWidth().height(256.dp)) }
      Text("Mode: ${generation.editingMode?.name ?: mode.name}"); generation.visibleInstruction?.takeIf { it.isNotBlank() }?.let { Text("Instruction: $it") }; generation.figurePresetName?.let { Text("Figure preset: $it") }
      Row { Button({ scope.launch { runCatching { viewModel.imageStore.savePng(bitmap) }.onSuccess(viewModel::saved) } }) { Text("Save PNG") }; Button({ scope.launch { runCatching { generation.savedOutputUri ?: viewModel.imageStore.savePng(bitmap) }.onSuccess { uri -> viewModel.saved(uri); context.startActivity(Intent.createChooser(viewModel.imageStore.shareIntent(uri), "Share FLUX image")) } } }) { Text("Share") } }
      Button({ scope.launch { runCatching { viewModel.imageStore.saveResized1024(bitmap) }.onSuccess(viewModel::saved) } }) { Text("Export 1024×1024") }
      Text("1024×1024 export is filtered resizing, not AI super-resolution.")
      Button({ viewModel.stageResultForEditing { imageUri = it; instruction = "" } }, enabled = !generation.stagingReference) { Text("Edit this result") }
      if (generation.editingMode == FluxEditMode.FIGURE) Button({ viewModel.stageResultForEditing { imageUri = it; instruction = ""; figureAction = FluxFigureAction.PreserveCurrent } }) { Text("Use this figure for subsequent edits") }
      Button(viewModel::regenerateSameSettings, enabled = !generation.running) { Text("Regenerate with same settings") }
      OutlinedButton({ confirmDifferentSeed = true }, enabled = !generation.running) { Text("Try a different seed") }
      if (viewerOpen) FluxFullscreenViewer(bitmap) { viewerOpen = false }
    }
    FluxDeveloperVerificationSection(state is FluxEditorUiState.Ready, imageUri, instruction)
  }
  if (conflicts.isNotEmpty()) AlertDialog(onDismissRequest = { conflicts = emptyList() }, title = { Text("Instruction conflicts with locks") }, text = { Text(conflicts.joinToString("\n")) }, confirmButton = { Button({ runGeneration(true) }) { Text("Unlock conflicting properties") } }, dismissButton = { Row { OutlinedButton({ conflicts = emptyList() }) { Text("Edit instruction") }; OutlinedButton({ conflicts = emptyList() }) { Text("Cancel") } } })
  if (confirmDifferentSeed) AlertDialog(onDismissRequest = { confirmDifferentSeed = false }, title = { Text("Try a different seed?") }, text = { Text("This starts one manual generation with a fresh random seed. The current result remains visible until the new result succeeds.") }, confirmButton = { Button({ confirmDifferentSeed = false; viewModel.regenerateWithDifferentSeed() }) { Text("Generate") } }, dismissButton = { OutlinedButton({ confirmDifferentSeed = false }) { Text("Cancel") } })
}

@Composable private fun Lock(label: String, checked: Boolean, change: (Boolean) -> Unit) { Row(verticalAlignment = Alignment.CenterVertically) { Checkbox(checked, change); Text(label) } }

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
