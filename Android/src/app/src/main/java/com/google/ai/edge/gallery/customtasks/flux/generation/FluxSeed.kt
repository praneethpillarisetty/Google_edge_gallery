/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.generation

import java.security.SecureRandom
import java.util.Random

sealed interface FluxSeedSelection {
  data object Random : FluxSeedSelection
  data class Fixed(val value: Long) : FluxSeedSelection
}

data class FluxInitialLatents(val seed: Long, private val values: FloatArray) {
  val shape: List<Int> = listOf(1, 256, 128)
  val elementCount: Int = ELEMENTS
  fun copyValues(): FloatArray = values.copyOf()
  init { require(values.size == ELEMENTS && values.all(Float::isFinite)) }
  companion object { const val ELEMENTS = 32_768 }
}

object FluxSeedParser {
  fun parseFixed(text: String): FluxSeedSelection.Fixed {
    val trimmed = text.trim()
    require(trimmed.isNotEmpty() && trimmed.toLongOrNull() != null) { "Fixed seed must be a signed 64-bit integer." }
    return FluxSeedSelection.Fixed(trimmed.toLong())
  }
}

class FluxProductionNoiseFactory(
  private val randomSeedSource: () -> Long = { SecureRandom().nextLong() },
) {
  fun create(selection: FluxSeedSelection = FluxSeedSelection.Random): FluxInitialLatents {
    val actualSeed = when (selection) {
      FluxSeedSelection.Random -> randomSeedSource()
      is FluxSeedSelection.Fixed -> selection.value
    }
    return FluxInitialLatents(actualSeed, gaussianLatents(actualSeed))
  }

  private fun gaussianLatents(seed: Long): FloatArray {
    val rng = Random(seed)
    return FloatArray(FluxInitialLatents.ELEMENTS) { rng.nextGaussian().toFloat() }
  }
}

object FluxProductionBoundaryPolicy {
  const val DIAGNOSTIC_LATENTS0 = "latents0.bin"
  const val SYNTHETIC_ZERO_NOISE = "synthetic-zero-noise"
  const val DIAGNOSTIC_TIMESTEP = "diagnostic-timestep"
  val productionInitialNoiseSource = FluxProductionNoiseFactory::class.qualifiedName.orEmpty()
}
