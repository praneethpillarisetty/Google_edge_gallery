/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.generation

enum class FluxEditMode { SIMPLE, FIGURE }

enum class FluxPromptIntent { OUTFIT, BACKGROUND, POSE, CAMERA, LIGHTING, HAIR, MAKEUP_EXPRESSION, BODY }

sealed interface FluxFigureAction {
  data object PreserveCurrent : FluxFigureAction
  data class ApplyPreset(val presetId: String) : FluxFigureAction
  data class Custom(val description: String) : FluxFigureAction
}

data class FluxSimpleEditRequest(
  val userInstruction: String,
  val preserveIdentity: Boolean = true,
  val preservePoseAndComposition: Boolean = true,
  val preserveBackground: Boolean = false,
)

data class FluxFigureEditRequest(
  val figureAction: FluxFigureAction,
  val figureDescription: String? = null,
  val preserveOutfit: Boolean = true,
  val preservePose: Boolean = true,
  val preserveBackground: Boolean = true,
  val preserveCamera: Boolean = true,
  val preserveLighting: Boolean = true,
  val preserveHair: Boolean = true,
  val preserveMakeupAndExpression: Boolean = true,
  val preserveAccessories: Boolean = true,
  val additionalInstruction: String? = null,
)

data class FluxPromptConflict(val intent: FluxPromptIntent, val lockName: String)

data class FluxCompiledPrompt(
  val positivePrompt: String,
  val intents: Set<FluxPromptIntent>,
  val conflicts: List<FluxPromptConflict> = emptyList(),
)
