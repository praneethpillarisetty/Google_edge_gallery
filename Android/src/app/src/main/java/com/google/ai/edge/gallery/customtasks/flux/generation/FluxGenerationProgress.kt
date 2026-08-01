/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.generation

enum class FluxGenerationStage {
  IDLE, VALIDATING, STAGING_REFERENCE, PREPROCESSING_REFERENCE, ENCODING_REFERENCE,
  PREPARING_REFERENCE_TOKENS, CONDITIONING_PROMPT, TRANSFORMER_STEP_1,
  TRANSFORMER_STEP_2, TRANSFORMER_STEP_3, TRANSFORMER_STEP_4, PREPARING_DECODER,
  DECODING_VAE, CREATING_BITMAP, COMPLETE, CANCELLED, ERROR,
}

data class FluxGenerationProgress(
  val stage: FluxGenerationStage,
  val currentStep: Int = 0,
  val currentGraph: String? = null,
  val completedGraphCount: Int = 0,
)
