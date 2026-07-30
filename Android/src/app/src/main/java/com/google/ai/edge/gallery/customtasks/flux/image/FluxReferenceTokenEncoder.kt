/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.image

import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext

object FluxReferenceTokenContracts {
  const val SPATIAL_TOKENS = 256
  const val TOKEN_WIDTH = 128
  const val ELEMENTS = SPATIAL_TOKENS * TOKEN_WIDTH
  val SHAPE = listOf(1, SPATIAL_TOKENS, TOKEN_WIDTH)
}

enum class FluxReferenceTokenStage {
  PATCHIFYING_REFERENCE_LATENT,
  NORMALIZING_REFERENCE_CHANNELS,
  BUILDING_REFERENCE_TOKENS,
  VALIDATING_REFERENCE_TOKENS,
}

/** Authoritative gather, normalization, and token construction for a typed Phase 2D latent. */
class FluxReferenceTokenEncoder internal constructor(private val constants: FluxReferenceConstants) {
  suspend fun encode(
    latent: FluxReferenceLatent,
    onStage: (FluxReferenceTokenStage) -> Unit = {},
  ): FluxReferenceTokens {
    coroutineContext.ensureActive()
    require(latent.shape == FluxReferenceImageContracts.OUTPUT_SHAPE && latent.dataType == "FP32") {
      "Reference latent contract is not [1,32,32,32] FP32 NCHW."
    }
    val source = latent.copyValues()
    coroutineContext.ensureActive()
    onStage(FluxReferenceTokenStage.PATCHIFYING_REFERENCE_LATENT)
    val packed = FloatArray(FluxReferenceTokenContracts.ELEMENTS)
    for (outputIndex in packed.indices) {
      if (outputIndex and 1023 == 0) coroutineContext.ensureActive()
      packed[outputIndex] = source[constants.patchPermutation[outputIndex]]
    }
    coroutineContext.ensureActive()
    onStage(FluxReferenceTokenStage.NORMALIZING_REFERENCE_CHANNELS)
    for (packedIndex in packed.indices) {
      if (packedIndex and 1023 == 0) coroutineContext.ensureActive()
      val channel = packedIndex / FluxReferenceTokenContracts.SPATIAL_TOKENS
      packed[packedIndex] =
        (packed[packedIndex] - constants.mean[channel]) / constants.standardDeviation[channel]
      if (!packed[packedIndex].isFinite()) throw FluxReferenceTokenException("Reference normalization produced a non-finite value.")
    }
    coroutineContext.ensureActive()
    onStage(FluxReferenceTokenStage.BUILDING_REFERENCE_TOKENS)
    val tokens = FloatArray(FluxReferenceTokenContracts.ELEMENTS)
    for (channel in 0 until FluxReferenceTokenContracts.TOKEN_WIDTH) {
      val channelOffset = Math.multiplyExact(channel, FluxReferenceTokenContracts.SPATIAL_TOKENS)
      for (spatialIndex in 0 until FluxReferenceTokenContracts.SPATIAL_TOKENS) {
        val tokenIndex = Math.addExact(Math.multiplyExact(spatialIndex, FluxReferenceTokenContracts.TOKEN_WIDTH), channel)
        tokens[tokenIndex] = packed[channelOffset + spatialIndex]
      }
    }
    coroutineContext.ensureActive()
    onStage(FluxReferenceTokenStage.VALIDATING_REFERENCE_TOKENS)
    val result = FluxReferenceTokens.checked(tokens)
    coroutineContext.ensureActive()
    return result
  }
}

class FluxReferenceTokens private constructor(private val storage: FloatArray) {
  val shape: List<Int> = FluxReferenceTokenContracts.SHAPE
  val dataType = "FP32"
  val spatialTokens = FluxReferenceTokenContracts.SPATIAL_TOKENS
  val tokenWidth = FluxReferenceTokenContracts.TOKEN_WIDTH
  val size get() = storage.size
  fun allFinite(): Boolean = storage.all(Float::isFinite)
  fun copyValues(): FloatArray = storage.copyOf()
  companion object {
    fun checked(values: FloatArray): FluxReferenceTokens {
      if (values.size != FluxReferenceTokenContracts.ELEMENTS) throw FluxReferenceTokenException("Reference tokens must contain exactly 32,768 values.")
      if (values.any { !it.isFinite() }) throw FluxReferenceTokenException("Reference tokens contain a non-finite value.")
      return FluxReferenceTokens(values.copyOf())
    }
  }
}
