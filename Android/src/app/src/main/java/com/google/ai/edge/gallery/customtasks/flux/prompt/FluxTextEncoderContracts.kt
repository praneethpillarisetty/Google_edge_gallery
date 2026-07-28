/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.prompt

/** Fixed FP32 contracts exported by the pinned FLUX.2 Klein companion implementation. */
data class FluxTextEncoderContracts(
  val sequenceLength: Int = 512,
  val hiddenSize: Int = 2560,
  val attentionHeads: Int = 32,
  val rotaryDimension: Int = 128,
) {
  val hiddenElements = checkedElements(1, sequenceLength, hiddenSize)
  val maskElements = checkedElements(1, attentionHeads, sequenceLength, sequenceLength)
  val rotaryElements = checkedElements(1, sequenceLength, rotaryDimension)
  val conditioningElements = checkedElements(1, sequenceLength, 3, hiddenSize)

  init {
    require(sequenceLength > 0 && hiddenSize > 0 && attentionHeads > 0)
    require(rotaryDimension > 0 && rotaryDimension % 2 == 0)
  }

  /** Peak host arrays: input embedding, mask, cos/sin, three taps, and destination conditioning. */
  fun estimatedPeakBytes(): Long = checkedBytes(
    hiddenElements.toLong() + maskElements + 2L * rotaryElements +
      3L * hiddenElements + conditioningElements
  )

  companion object {
    const val NEGATIVE_MASK_VALUE = -1e9f
    const val ROTARY_BASE = 1_000_000.0
    val GRAPH_NAMES = listOf("ke_enc0.tflite", "ke_enc1.tflite", "ke_enc2.tflite")
    const val EXPECTED_PEAK_BYTES = 70_778_880L

    fun checkedElements(vararg dimensions: Int): Int {
      require(dimensions.all { it > 0 }) { "Tensor dimensions must be positive." }
      return dimensions.fold(1, Math::multiplyExact)
    }

    fun checkedBytes(elements: Long): Long {
      require(elements >= 0) { "Tensor element count must not be negative." }
      return Math.multiplyExact(elements, Float.SIZE_BYTES.toLong())
    }
  }
}
