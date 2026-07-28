/*
 * Copyright 2026 Google LLC
 * Licensed under the Apache License, Version 2.0.
 * Derived from google-ai-edge/litert-samples PR 227 at
 * f48a89e4f29a74ab51f29c311ac7a0e5e479d225.
 */
package com.google.ai.edge.gallery.customtasks.flux.prompt

import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxGraphRunner
import java.io.Closeable
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.ensureActive

internal data class FluxPreparedText(
  val embeddings: FloatArray,
  val mask: FloatArray,
  val cosine: FloatArray,
  val sine: FloatArray,
)

internal fun interface FluxPromptPreprocessor : Closeable {
  fun prepare(prompt: String): FluxPreparedText
  override fun close() = Unit
}

internal class FluxAssetPromptPreprocessor(
  private val assets: FluxPromptAssets,
  private val contracts: FluxTextEncoderContracts,
) : FluxPromptPreprocessor {
  private var table: FluxEmbeddingTable? = null

  override fun prepare(prompt: String): FluxPreparedText {
    val tokenizer = FluxQwenTokenizer.load(assets.vocabulary.toPath(), assets.merges.toPath(), assets.specials.toPath())
    val tokens = tokenizer.prepareForTextEncoder(prompt)
    val activeTable = FluxEmbeddingTable.open(assets.embeddings.toPath()).also { table = it }
    val embeddings = activeTable.lookup(tokens.tokenIds)
    validateFinite("embedding", embeddings)
    return FluxPreparedText(
      embeddings,
      FluxPromptConditioner.buildMask(tokens.validPositions, contracts),
      FluxPromptConditioner.buildRotary(contracts, false),
      FluxPromptConditioner.buildRotary(contracts, true),
    )
  }

  override fun close() { table?.close(); table = null }
}

/** Runs exactly the three Qwen encoder chunks and returns their per-token interleaved taps. */
class FluxPromptConditioner internal constructor(
  private val assets: FluxPromptAssets,
  private val runner: FluxGraphRunner,
  private val contracts: FluxTextEncoderContracts = FluxTextEncoderContracts(),
  private val preprocessorFactory: (FluxPromptAssets, FluxTextEncoderContracts) -> FluxPromptPreprocessor = ::FluxAssetPromptPreprocessor,
) {
  suspend fun condition(prompt: String): FluxTextConditioning {
    try { contracts.estimatedPeakBytes() } catch (e: ArithmeticException) {
      throw FluxTextConditioningException("Prompt-encoder memory calculation overflowed.", e)
    }
    coroutineContext.ensureActive()
    val prepared = preprocessorFactory(assets, contracts).use { it.prepare(prompt) }
    validateSize("embeddings", prepared.embeddings, contracts.hiddenElements)
    validateSize("mask", prepared.mask, contracts.maskElements)
    validateSize("cosine", prepared.cosine, contracts.rotaryElements)
    validateSize("sine", prepared.sine, contracts.rotaryElements)
    listOf(prepared.embeddings, prepared.mask, prepared.cosine, prepared.sine).forEach { validateFinite("encoder input", it) }

    var hidden = prepared.embeddings
    val taps = ArrayList<FloatArray>(3)
    val durations = ArrayList<Long>(3)
    for ((index, graph) in assets.encoderGraphs.withIndex()) {
      coroutineContext.ensureActive()
      val start = System.nanoTime()
      val result = runner.run(graph, listOf(hidden, prepared.mask, prepared.cosine, prepared.sine))
      durations += (System.nanoTime() - start) / 1_000_000
      if (result.graph != FluxTextEncoderContracts.GRAPH_NAMES[index]) throw FluxTextConditioningException("Unexpected graph result at encoder stage $index.")
      if (result.outputs.size != 1) throw FluxTextConditioningException("Encoder stage $index returned ${result.outputs.size} outputs; expected 1.")
      hidden = result.outputs.single()
      validateSize("encoder stage $index output", hidden, contracts.hiddenElements)
      validateFinite("encoder stage $index output", hidden)
      taps += hidden
    }
    coroutineContext.ensureActive()
    val output = interleaveTaps(taps, contracts)
    taps.clear()
    return FluxTextConditioning(output, listOf(1, contracts.sequenceLength, 3 * contracts.hiddenSize), durations)
  }

  companion object {
    fun create(resolver: FluxPromptAssetResolver, runner: FluxGraphRunner): FluxPromptConditioner =
      FluxPromptConditioner(resolver.resolve(), runner)

    internal fun buildMask(valid: BooleanArray, c: FluxTextEncoderContracts): FloatArray {
      require(valid.size == c.sequenceLength)
      val plane = FloatArray(FluxTextEncoderContracts.checkedElements(c.sequenceLength, c.sequenceLength))
      for (query in 0 until c.sequenceLength) for (key in 0 until c.sequenceLength) {
        var value = 0f
        if (key > query) value += FluxTextEncoderContracts.NEGATIVE_MASK_VALUE
        if (!valid[key]) value += FluxTextEncoderContracts.NEGATIVE_MASK_VALUE
        plane[query * c.sequenceLength + key] = value
      }
      return FloatArray(c.maskElements).also { out ->
        repeat(c.attentionHeads) { plane.copyInto(out, it * plane.size) }
      }
    }

    /** Qwen3 default RoPE: concat each 64-frequency table with itself to head width 128. */
    internal fun buildRotary(c: FluxTextEncoderContracts, sine: Boolean): FloatArray {
      val half = c.rotaryDimension / 2
      return FloatArray(c.rotaryElements) { flat ->
        val position = flat / c.rotaryDimension
        val frequency = flat % c.rotaryDimension % half
        val inverse = 1.0 / StrictMath.pow(FluxTextEncoderContracts.ROTARY_BASE, (2.0 * frequency) / c.rotaryDimension)
        (if (sine) StrictMath.sin(position * inverse) else StrictMath.cos(position * inverse)).toFloat()
      }
    }

    internal fun interleaveTaps(taps: List<FloatArray>, c: FluxTextEncoderContracts): FloatArray {
      require(taps.size == 3 && taps.all { it.size == c.hiddenElements })
      val destination = FloatArray(c.conditioningElements)
      for (token in 0 until c.sequenceLength) for (tap in 0 until 3) {
        taps[tap].copyInto(destination, (token * 3 + tap) * c.hiddenSize, token * c.hiddenSize, (token + 1) * c.hiddenSize)
      }
      return destination
    }
  }
}

private fun validateSize(name: String, values: FloatArray, expected: Int) {
  if (values.size != expected) throw FluxTextConditioningException("$name has ${values.size} values; expected $expected.")
}

private fun validateFinite(name: String, values: FloatArray) {
  if (values.any { !it.isFinite() }) throw FluxTextConditioningException("$name contains a non-finite value.")
}
