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

object FluxTransformerDenoisingContracts {
  val GRAPH_ORDER = listOf("kce_prep.tflite", "kce_double0.tflite", "kce_double1.tflite", "kce_single0.tflite", "kce_single1.tflite", "kce_single2.tflite", "kce_single3.tflite", "kce_final.tflite")
}

data class FluxDenoisingReport(val initialShape: List<Int>, val finalShape: List<Int>, val steps: Int, val allFinite: Boolean)
class FluxTransformerDenoisingException(message: String) : IllegalStateException(message)

/** Exact editing loop transcribed from immutable companion revision f48a89e. */
class FluxTransformerDenoiser(private val runner: FluxGraphRunner) {
  suspend fun run(root: File, manifest: FluxModelManifest, evidence: FluxPhase2gEvidence, prompt: FluxTextConditioning, reference: FluxReferenceTokens, onGraph: suspend (Int, String) -> Unit = { _, _ -> }): FluxDenoisingReport {
    require(manifest.revision == FLUX_MODEL_REVISION && FluxTransformerDenoisingContracts.GRAPH_ORDER.all { it in manifest.files })
    require(prompt.shape == listOf(1, 512, 7680) && prompt.values.size == 3_932_160 && prompt.values.all(Float::isFinite))
    require(reference.shape == listOf(1, 256, 128) && reference.size == 32_768)
    val canonical = root.canonicalFile
    val graphs = FluxTransformerDenoisingContracts.GRAPH_ORDER.associateWith { name -> File(canonical, name).canonicalFile.also { require(it.parentFile == canonical && it.isFile) } }
    var latents = evidence.initialLatents().also { validate("initial latents", it, 32_768) }
    val referenceValues = reference.copyValues().also { validate("reference tokens", it, 32_768) }
    val cos = evidence.cos().also { validate("cos", it, 65_536) }
    val sin = evidence.sin().also { validate("sin", it, 65_536) }
    for (step in 0 until 4) {
      coroutineContext.ensureActive()
      val tokens = FloatArray(65_536)
      latents.copyInto(tokens); referenceValues.copyInto(tokens, latents.size)
      var result = graph(step, "kce_prep.tflite", graphs, listOf(tokens, prompt.values.copyOf(), evidence.timestep(step)), onGraph, intArrayOf(1_572_864, 1_572_864, 18_432, 18_432, 9_216))
      var image = result[0]; var text = result[1]
      val imageMod = result[2]; val textMod = result[3]; val singleMod = result[4]
      for (index in 0..1) {
        result = graph(step, "kce_double$index.tflite", graphs, listOf(image, text, cos, sin, imageMod, textMod), onGraph, intArrayOf(1_572_864, 1_572_864))
        image = result[0]; text = result[1]
      }
      var joint = FloatArray(3_145_728)
      text.copyInto(joint); image.copyInto(joint, text.size)
      for (index in 0..3) joint = graph(step, "kce_single$index.tflite", graphs, listOf(joint, cos, sin, singleMod), onGraph, intArrayOf(3_145_728))[0]
      val prediction = graph(step, "kce_final.tflite", graphs, listOf(joint, evidence.timestep(step)), onGraph, intArrayOf(65_536))[0]
      coroutineContext.ensureActive()
      val delta = evidence.dsigma(step)
      for (i in latents.indices) { if (i and 1023 == 0) coroutineContext.ensureActive(); latents[i] += delta * prediction[i] }
      validate("latent update at step ${step + 1}", latents, 32_768)
    }
    return FluxDenoisingReport(listOf(1, 256, 128), listOf(1, 256, 128), 4, true)
  }

  private suspend fun graph(step: Int, name: String, graphs: Map<String, File>, inputs: List<FloatArray>, onGraph: suspend (Int, String) -> Unit, sizes: IntArray): List<FloatArray> {
    coroutineContext.ensureActive(); inputs.forEachIndexed { i, v -> validate("$name input $i at step ${step + 1}", v, v.size) }
    onGraph(step + 1, name); coroutineContext.ensureActive()
    val result = runner.run(graphs.getValue(name), inputs)
    if (result.graph != name || result.outputs.size != sizes.size) throw FluxTransformerDenoisingException("$name returned an invalid output contract at step ${step + 1}.")
    result.outputs.forEachIndexed { i, values -> validate("$name output $i at step ${step + 1}", values, sizes[i]) }
    return result.outputs
  }
  private fun validate(identity: String, values: FloatArray, size: Int) {
    if (values.size != size) throw FluxTransformerDenoisingException("$identity has an invalid shape.")
    if (values.any { !it.isFinite() }) throw FluxTransformerDenoisingException("$identity contains a non-finite value.")
  }
}
