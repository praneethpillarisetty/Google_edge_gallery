/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.transformer

import com.google.ai.edge.gallery.customtasks.flux.FLUX_MODEL_REVISION
import com.google.ai.edge.gallery.customtasks.flux.FluxModelManifest
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceTokens
import com.google.ai.edge.gallery.customtasks.flux.prompt.FluxTextConditioning
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxGraphRunner
import java.io.File
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.ensureActive

object FluxTransformerPrepContracts {
  const val GRAPH = "kce_prep.tflite"
  const val TOKEN_WIDTH = 128
  const val HALF_TOKENS = 256
  const val HALF_ELEMENTS = 32_768
  const val EDITING_TOKENS = 512
  const val EDITING_ELEMENTS = 65_536
  const val TIMESTEP_ELEMENTS = 3_072
  val NOISE_SHAPE = listOf(1, HALF_TOKENS, TOKEN_WIDTH)
  val EDITING_SHAPE = listOf(1, EDITING_TOKENS, TOKEN_WIDTH)
  val TIMESTEP_SHAPE = listOf(1, TIMESTEP_ELEMENTS)
  val OUTPUTS = listOf(
    FluxPrepOutputContract("image hidden", listOf(1, 512, 3072), 1_572_864),
    FluxPrepOutputContract("text hidden", listOf(1, 512, 3072), 1_572_864),
    FluxPrepOutputContract("image modulation", listOf(1, 1, 18432), 18_432),
    FluxPrepOutputContract("text modulation", listOf(1, 1, 18432), 18_432),
    FluxPrepOutputContract("single-stream modulation", listOf(1, 1, 9216), 9_216),
  )
}

data class FluxPrepOutputContract(val identity: String, val shape: List<Int>, val elements: Int)
class FluxTransformerPrepException(message: String) : Exception(message)

class FluxDiagnosticNoiseTokens private constructor(private val storage: FloatArray) {
  val shape = FluxTransformerPrepContracts.NOISE_SHAPE
  val dataType = "FP32"
  val size get() = storage.size
  fun copyValues() = storage.copyOf()
  companion object {
    fun checked(values: FloatArray, shape: List<Int> = FluxTransformerPrepContracts.NOISE_SHAPE): FluxDiagnosticNoiseTokens {
      validate("Diagnostic noise", values, shape, FluxTransformerPrepContracts.NOISE_SHAPE, FluxTransformerPrepContracts.HALF_ELEMENTS)
      return FluxDiagnosticNoiseTokens(values.copyOf())
    }
  }
}

class FluxTimestepEmbedding private constructor(private val storage: FloatArray) {
  val shape = FluxTransformerPrepContracts.TIMESTEP_SHAPE
  val dataType = "FP32"
  val size get() = storage.size
  fun copyValues() = storage.copyOf()
  companion object {
    fun checked(values: FloatArray, shape: List<Int> = FluxTransformerPrepContracts.TIMESTEP_SHAPE): FluxTimestepEmbedding {
      validate("Timestep embedding", values, shape, FluxTransformerPrepContracts.TIMESTEP_SHAPE, FluxTransformerPrepContracts.TIMESTEP_ELEMENTS)
      return FluxTimestepEmbedding(values.copyOf())
    }
  }
}

/** Debug/test structural inputs. This is deliberately not a production scheduler boundary. */
object FluxSyntheticDiagnosticInputs {
  fun syntheticZeroNoiseTokens() = FluxDiagnosticNoiseTokens.checked(FloatArray(FluxTransformerPrepContracts.HALF_ELEMENTS))
  fun syntheticZeroTimestepEmbedding() = FluxTimestepEmbedding.checked(FloatArray(FluxTransformerPrepContracts.TIMESTEP_ELEMENTS))
}

class FluxEditingImageTokens private constructor(private val storage: FloatArray) {
  val shape = FluxTransformerPrepContracts.EDITING_SHAPE
  val dataType = "FP32"
  val size get() = storage.size
  fun copyValues() = storage.copyOf()

  companion object {
    suspend fun assemble(noise: FluxDiagnosticNoiseTokens, reference: FluxReferenceTokens): FluxEditingImageTokens {
      coroutineContext.ensureActive() // before validation
      require(noise.shape == FluxTransformerPrepContracts.NOISE_SHAPE && noise.size == FluxTransformerPrepContracts.HALF_ELEMENTS)
      require(reference.shape == FluxTransformerPrepContracts.NOISE_SHAPE && reference.size == FluxTransformerPrepContracts.HALF_ELEMENTS)
      val noiseValues = noise.copyValues()
      val referenceValues = reference.copyValues()
      if (noiseValues.any { !it.isFinite() } || referenceValues.any { !it.isFinite() }) throw FluxTransformerPrepException("Editing-token input contains a non-finite value.")
      coroutineContext.ensureActive() // before allocation
      val count = Math.addExact(noiseValues.size, referenceValues.size)
      val combined = FloatArray(count)
      for (index in noiseValues.indices) {
        if (index and 1023 == 0) coroutineContext.ensureActive()
        combined[index] = noiseValues[index]
      }
      for (index in referenceValues.indices) {
        if (index and 1023 == 0) coroutineContext.ensureActive()
        combined[Math.addExact(noiseValues.size, index)] = referenceValues[index]
      }
      coroutineContext.ensureActive() // before final validation
      validate("Editing image tokens", combined, FluxTransformerPrepContracts.EDITING_SHAPE, FluxTransformerPrepContracts.EDITING_SHAPE, FluxTransformerPrepContracts.EDITING_ELEMENTS)
      val result = FluxEditingImageTokens(combined.copyOf())
      coroutineContext.ensureActive() // immediately before return
      return result
    }
  }
}

data class FluxTransformerPrepReport(val outputContracts: List<FluxPrepOutputContract>, val allFinite: Boolean)

/** Runs exactly the prep graph. Returned graph arrays are validated and then made unreachable. */
class FluxTransformerPrepRunner(private val runner: FluxGraphRunner) {
  fun resolve(root: File, manifest: FluxModelManifest): File {
    require(manifest.revision == FLUX_MODEL_REVISION && FluxTransformerPrepContracts.GRAPH in manifest.files)
    val canonicalRoot = root.canonicalFile
    val graph = File(canonicalRoot, FluxTransformerPrepContracts.GRAPH).canonicalFile
    require(graph.parentFile == canonicalRoot && graph.isFile) { "Required transformer prep graph is missing or unsafe." }
    return graph
  }

  suspend fun run(graph: File, editing: FluxEditingImageTokens, prompt: FluxTextConditioning, timestep: FluxTimestepEmbedding): FluxTransformerPrepReport {
    coroutineContext.ensureActive()
    require(graph.name == FluxTransformerPrepContracts.GRAPH)
    require(prompt.shape == listOf(1, 512, 7680) && prompt.values.size == 3_932_160 && prompt.values.all(Float::isFinite))
    val result = runner.run(graph, listOf(editing.copyValues(), prompt.values.copyOf(), timestep.copyValues()))
    coroutineContext.ensureActive()
    if (result.graph != FluxTransformerPrepContracts.GRAPH) throw FluxTransformerPrepException("Unexpected transformer prep graph result.")
    if (result.outputs.size != 5) throw FluxTransformerPrepException("kce_prep.tflite returned ${result.outputs.size} outputs; expected exactly five.")
    result.outputs.forEachIndexed { index, values ->
      val contract = FluxTransformerPrepContracts.OUTPUTS[index]
      if (values.size != contract.elements) throw FluxTransformerPrepException("${contract.identity} has ${values.size} values; expected ${contract.elements}.")
      if (values.any { !it.isFinite() }) throw FluxTransformerPrepException("${contract.identity} contains a non-finite value.")
    }
    coroutineContext.ensureActive()
    return FluxTransformerPrepReport(FluxTransformerPrepContracts.OUTPUTS, true)
  }
}

private fun validate(name: String, values: FloatArray, shape: List<Int>, expectedShape: List<Int>, expectedElements: Int) {
  val shapeElements = try { shape.fold(1, Math::multiplyExact) } catch (_: ArithmeticException) { throw FluxTransformerPrepException("$name shape overflows.") }
  if (shape != expectedShape || shapeElements != expectedElements || values.size != expectedElements) throw FluxTransformerPrepException("$name has an incorrect shape or element count.")
  if (values.any { !it.isFinite() }) throw FluxTransformerPrepException("$name contains a non-finite value.")
}
