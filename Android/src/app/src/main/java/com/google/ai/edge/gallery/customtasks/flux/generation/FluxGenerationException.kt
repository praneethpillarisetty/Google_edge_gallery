/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.generation

enum class FluxGenerationError(val userMessage: String) {
  MODEL_NOT_READY("The FLUX model package is not ready."), REFERENCE_MISSING("Select a reference image."),
  PROMPT_MISSING("Enter an edit prompt."), REFERENCE_UNAVAILABLE("The reference image is unavailable."),
  STAGING_FAILED("The reference image could not be staged."), UNSUPPORTED_IMAGE("The selected image is unsupported."),
  INVALID_EVIDENCE("Generation evidence is invalid."), MODEL_CHANGED("A required model file is missing or changed."),
  GPU_FAILURE("FLUX GPU execution failed."), NON_FINITE_TENSOR("Generation produced an invalid tensor."),
  DECODER_FAILURE("The generated image could not be decoded."), BITMAP_FAILURE("The generated bitmap could not be created."),
  OUT_OF_MEMORY("There is not enough memory to generate this image."), THERMAL("Generation is unavailable while the device is too hot."),
  NATIVE_LINKAGE("The LiteRT native runtime is unavailable."), UNKNOWN("Image generation failed."),
}

class FluxGenerationException(val category: FluxGenerationError, cause: Throwable? = null) :
  Exception(category.userMessage, cause)
