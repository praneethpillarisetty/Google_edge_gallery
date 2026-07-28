/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.prompt

/** Layer-9/18/27 taps interleaved per token, in row-major FP32 storage. */
data class FluxTextConditioning(
  val values: FloatArray,
  val shape: List<Int>,
  val stageDurationsMillis: List<Long>,
) {
  init {
    require(shape.all { it > 0 })
    require(shape.fold(1, Math::multiplyExact) == values.size)
    require(values.all(Float::isFinite))
  }
}

class FluxTextConditioningException(message: String, cause: Throwable? = null) :
  Exception(message, cause)
