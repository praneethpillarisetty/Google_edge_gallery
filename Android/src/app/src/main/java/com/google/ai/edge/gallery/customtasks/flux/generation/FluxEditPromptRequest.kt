/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.generation

enum class FluxEditMode { SIMPLE, FIGURE }

enum class FluxPromptIntent { OUTFIT, BACKGROUND, POSE, CAMERA, LIGHTING, HAIR, MAKEUP_EXPRESSION, BODY }

enum class FluxRealismProfile { NATURAL_PHOTO, EDITORIAL_PHOTO, CINEMATIC_PHOTO }

enum class FluxFramingCategory { UNSPECIFIED, FULL_BODY, THREE_QUARTER, WAIST_UP, CLOSE_UP }

enum class FluxPoseCategory { NONE, GENERIC, SITTING, STANDING, GAIT, RAISED_ARMS, LEANING }

data class FluxEditVisualContext(
  val framing: FluxFramingCategory = FluxFramingCategory.UNSPECIFIED,
  val handsVisible: Boolean = false,
  val limbsOverlap: Boolean = false,
  val faceTurned: Boolean = false,
  val faceOccluded: Boolean = false,
)

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
  val realismProfile: FluxRealismProfile = FluxRealismProfile.NATURAL_PHOTO,
  val visualContext: FluxEditVisualContext = FluxEditVisualContext(),
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
  val realismProfile: FluxRealismProfile = FluxRealismProfile.NATURAL_PHOTO,
  val visualContext: FluxEditVisualContext = FluxEditVisualContext(),
)

data class FluxPromptConflict(val intent: FluxPromptIntent, val lockName: String)

data class FluxCompiledPrompt(
  val positivePrompt: String,
  val intents: Set<FluxPromptIntent>,
  val conflicts: List<FluxPromptConflict> = emptyList(),
  val sectionNames: Set<String> = emptySet(),
  val framingCategory: FluxFramingCategory = FluxFramingCategory.UNSPECIFIED,
  val poseCategory: FluxPoseCategory = FluxPoseCategory.NONE,
)
