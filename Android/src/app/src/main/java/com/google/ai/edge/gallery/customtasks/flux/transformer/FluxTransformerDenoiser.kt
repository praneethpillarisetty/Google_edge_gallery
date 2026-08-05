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
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

object FluxTransformerDenoisingContracts {
  const val STEPS = 4
  const val GRAPHS_PER_STEP = 8
  val GRAPH_ORDER = listOf("kce_prep.tflite", "kce_double0.tflite", "kce_double1.tflite", "kce_single0.tflite", "kce_single1.tflite", "kce_single2.tflite", "kce_single3.tflite", "kce_final.tflite")
  val inputElements = mapOf(
    "kce_prep.tflite" to listOf(65_536, 3_932_160, 3_072),
    "kce_double0.tflite" to listOf(1_572_864, 1_572_864, 65_536, 65_536, 18_432, 18_432),
    "kce_double1.tflite" to listOf(1_572_864, 1_572_864, 65_536, 65_536, 18_432, 18_432),
    "kce_single0.tflite" to listOf(3_145_728, 65_536, 65_536, 9_216),
    "kce_single1.tflite" to listOf(3_145_728, 65_536, 65_536, 9_216),
    "kce_single2.tflite" to listOf(3_145_728, 65_536, 65_536, 9_216),
    "kce_single3.tflite" to listOf(3_145_728, 65_536, 65_536, 9_216),
    "kce_final.tflite" to listOf(3_145_728, 3_072),
  )
  val outputElements = mapOf(
    "kce_prep.tflite" to listOf(1_572_864, 1_572_864, 18_432, 18_432, 9_216),
    "kce_double0.tflite" to listOf(1_572_864, 1_572_864),
    "kce_double1.tflite" to listOf(1_572_864, 1_572_864),
    "kce_single0.tflite" to listOf(3_145_728), "kce_single1.tflite" to listOf(3_145_728),
    "kce_single2.tflite" to listOf(3_145_728), "kce_single3.tflite" to listOf(3_145_728),
    "kce_final.tflite" to listOf(65_536),
  )
}

data class FluxGraphTiming(val step: Int, val graph: String, val durationMillis: Long)
class FluxTransformerCoreResult private constructor(
  finalLatents: FloatArray,
  val backend: String = "GPU FP32",
  val graphSequence: List<String>,
  val completedSteps: Int,
  val initialShape: List<Int> = listOf(1, 256, 128),
  val initialElements: Int = 32_768,
  val finalShape: List<Int> = listOf(1, 256, 128),
  val finalElements: Int = 32_768,
  val allFinite: Boolean,
  val stepDurationsMillis: List<Long>,
  val graphTimings: List<FluxGraphTiming>,
) {
  private val ownedFinalLatents = finalLatents.copyOf()
  init {
    require(finalShape == listOf(1, 256, 128) && finalElements == 32_768 &&
      ownedFinalLatents.size == finalElements && ownedFinalLatents.all(Float::isFinite)) {
      "Final transformer latents must be finite FP32 [1,256,128]."
    }
  }
  fun copyFinalLatents(): FloatArray = ownedFinalLatents.copyOf()

  companion object {
    internal fun checked(
      finalLatents: FloatArray,
      graphSequence: List<String>, completedSteps: Int, allFinite: Boolean,
      stepDurationsMillis: List<Long>, graphTimings: List<FluxGraphTiming>,
    ) = FluxTransformerCoreResult(finalLatents, graphSequence = graphSequence,
      completedSteps = completedSteps, allFinite = allFinite,
      stepDurationsMillis = stepDurationsMillis, graphTimings = graphTimings)
  }
}
class FluxTransformerDenoisingException(message: String) : IllegalStateException(message)

data class FluxDenoisingProgress(val step: Int, val graph: String, val completedGraphs: Int)

internal object FluxTransformerExecutionGate {
  private val mutex = Mutex()
  suspend fun <T> exclusive(block: suspend () -> T): T = mutex.withLock { block() }
}

/** Exact editing loop transcribed from immutable companion revision f48a89e. */
class FluxTransformerDenoiser(
  private val runner: FluxGraphRunner,
  private val nanoTime: () -> Long = System::nanoTime,
) {
  suspend fun run(
    root: File,
    manifest: FluxModelManifest,
    evidence: FluxPhase2gEvidence,
    initialLatents: FloatArray,
    prompt: FluxTextConditioning,
    reference: FluxReferenceTokens,
    onProgress: suspend (FluxDenoisingProgress) -> Unit = {},
  ): FluxTransformerCoreResult = FluxTransformerExecutionGate.exclusive {
    require(manifest.revision == FLUX_MODEL_REVISION && FluxTransformerDenoisingContracts.GRAPH_ORDER.all { it in manifest.files }) { "Required transformer model inventory is missing or changed." }
    validate("preflight", 0, "prompt conditioning", prompt.values, 3_932_160)
    require(prompt.shape == listOf(1, 512, 7680)) { "Prompt conditioning has an invalid typed shape." }
    require(reference.shape == listOf(1, 256, 128) && reference.size == 32_768) { "Reference tokens have an invalid typed shape." }
    val canonical = root.canonicalFile
    val graphs = FluxTransformerDenoisingContracts.GRAPH_ORDER.associateWith { name ->
      File(canonical, name).canonicalFile.also { require(it.parentFile == canonical && it.isFile) { "Required transformer model inventory is missing or changed." } }
    }
    var latents = initialLatents.copyOf().also { validate("preflight", 0, "initial latents", it, 32_768) }
    val referenceValues = reference.copyValues().also { validate("preflight", 0, "reference tokens", it, 32_768) }
    val cos = evidence.cos().also { validate("preflight", 0, "rotary cosine", it, 65_536) }
    val sin = evidence.sin().also { validate("preflight", 0, "rotary sine", it, 65_536) }
    val graphTimings = mutableListOf<FluxGraphTiming>()
    val stepTimings = mutableListOf<Long>()
    var completedGraphs = 0
    for (stepIndex in 0 until FluxTransformerDenoisingContracts.STEPS) {
      coroutineContext.ensureActive()
      val step = stepIndex + 1
      val stepStart = nanoTime()
      val timestep = evidence.timestep(stepIndex)
      val delta = evidence.dsigma(stepIndex)
      val tokens = FloatArray(65_536)
      latents.copyInto(tokens)
      referenceValues.copyInto(tokens, 32_768)
      // Conditioning is immutable for the lifetime of this verification job and the runner only
      // reads inputs. Reuse it across steps rather than allocating a ~15 MiB duplicate per step.
      var outputs = execute(step, "kce_prep.tflite", graphs, listOf(tokens, prompt.values, timestep), completedGraphs, onProgress, graphTimings)
      completedGraphs++
      var image = outputs[0]
      var text = outputs[1]
      val imageModulation = outputs[2]
      val textModulation = outputs[3]
      val singleModulation = outputs[4]
      for (index in 0..1) {
        outputs = execute(step, "kce_double$index.tflite", graphs, listOf(image, text, cos, sin, imageModulation, textModulation), completedGraphs, onProgress, graphTimings)
        completedGraphs++
        image = outputs[0]
        text = outputs[1]
      }
      var joint = FloatArray(3_145_728)
      text.copyInto(joint)
      image.copyInto(joint, text.size)
      for (index in 0..3) {
        joint = execute(step, "kce_single$index.tflite", graphs, listOf(joint, cos, sin, singleModulation), completedGraphs, onProgress, graphTimings)[0]
        completedGraphs++
      }
      val prediction = execute(step, "kce_final.tflite", graphs, listOf(joint, timestep), completedGraphs, onProgress, graphTimings)[0]
      completedGraphs++
      coroutineContext.ensureActive()
      for (index in latents.indices) {
        if (index and 1023 == 0) coroutineContext.ensureActive()
        latents[index] += delta * prediction[index]
      }
      validate("latent update", step, "kce_final.tflite", latents, 32_768)
      coroutineContext.ensureActive()
      stepTimings += millisSince(stepStart)
    }
    coroutineContext.ensureActive()
    FluxTransformerCoreResult.checked(
      finalLatents = latents.copyOf(),
      graphSequence = FluxTransformerDenoisingContracts.GRAPH_ORDER,
      completedSteps = FluxTransformerDenoisingContracts.STEPS,
      allFinite = latents.all(Float::isFinite),
      stepDurationsMillis = stepTimings.toList(),
      graphTimings = graphTimings.toList(),
    )
  }

  private suspend fun execute(step: Int, name: String, graphs: Map<String, File>, inputs: List<FloatArray>, completed: Int, onProgress: suspend (FluxDenoisingProgress) -> Unit, timings: MutableList<FluxGraphTiming>): List<FloatArray> {
    coroutineContext.ensureActive()
    val expectedInputs = FluxTransformerDenoisingContracts.inputElements.getValue(name)
    if (inputs.size != expectedInputs.size) fail("input validation", step, name)
    inputs.forEachIndexed { index, values -> validate("input validation", step, "$name input $index", values, expectedInputs[index]) }
    onProgress(FluxDenoisingProgress(step, name, completed))
    coroutineContext.ensureActive()
    val start = nanoTime()
    val result = try { runner.run(graphs.getValue(name), inputs) } catch (cancelled: kotlinx.coroutines.CancellationException) { throw cancelled }
      catch (_: Exception) { fail("graph execution", step, name) }
    coroutineContext.ensureActive()
    val expectedOutputs = FluxTransformerDenoisingContracts.outputElements.getValue(name)
    if (result.graph != name || result.outputs.size != expectedOutputs.size) fail("output validation", step, name)
    result.outputs.forEachIndexed { index, values -> validate("output validation", step, "$name output $index", values, expectedOutputs[index]) }
    timings += FluxGraphTiming(step, name, millisSince(start))
    coroutineContext.ensureActive()
    return result.outputs
  }

  private fun validate(boundary: String, step: Int, identity: String, values: FloatArray, size: Int) {
    if (values.size != size) throw FluxTransformerDenoisingException("Step $step, $identity, $boundary: invalid shape or element count.")
    if (values.any { !it.isFinite() }) throw FluxTransformerDenoisingException("Step $step, $identity, $boundary: non-finite tensor.")
  }
  private fun fail(boundary: String, step: Int, graph: String): Nothing = throw FluxTransformerDenoisingException("Step $step, $graph, $boundary: transformer verification failed.")
  private fun millisSince(start: Long) = (nanoTime() - start) / 1_000_000

}
