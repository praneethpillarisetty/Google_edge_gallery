/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.generation

/** Debug-only structural report. It deliberately has no field capable of carrying prompt text. */
data class FluxPromptSectionReport(
  val sectionNames: Set<String>,
  val bodyTokenCount: Int,
  val maximumBodyTokens: Int,
  val truncationOccurred: Boolean,
  val omittedGeneratedSectionNames: Set<String>,
  val editingMode: FluxEditMode,
  val framingCategory: FluxFramingCategory,
  val poseCategory: FluxPoseCategory,
  val realismProfile: FluxRealismProfile,
  val enabledPreservationLockNames: Set<String>,
)
