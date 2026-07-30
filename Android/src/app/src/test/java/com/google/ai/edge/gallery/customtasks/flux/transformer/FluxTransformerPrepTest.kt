/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.transformer

import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceTokens
import com.google.ai.edge.gallery.customtasks.flux.prompt.FluxTextConditioning
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxGraphResult
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxGraphRunner
import java.io.File
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest

class FluxTransformerPrepTest {
  @Test fun syntheticContractsAreExactZerosAndDefensivelyOwned() {
    val noise = FluxSyntheticDiagnosticInputs.syntheticZeroNoiseTokens()
    val timestep = FluxSyntheticDiagnosticInputs.syntheticZeroTimestepEmbedding()
    assertEquals(listOf(1, 256, 128), noise.shape)
    assertEquals(32_768, noise.size)
    assertTrue(noise.copyValues().all { it.toRawBits() == 0 })
    assertEquals(listOf(1, 3072), timestep.shape)
    assertTrue(timestep.copyValues().all { it == 0f })
    noise.copyValues()[0] = 9f
    assertEquals(0f, noise.copyValues()[0])
  }

  @Test fun invalidDiagnosticInputsAreRejected() {
    assertFailsWith<FluxTransformerPrepException> { FluxDiagnosticNoiseTokens.checked(FloatArray(1)) }
    assertFailsWith<FluxTransformerPrepException> { FluxDiagnosticNoiseTokens.checked(FloatArray(32_768).also { it[2] = Float.NaN }) }
    assertFailsWith<FluxTransformerPrepException> { FluxTimestepEmbedding.checked(FloatArray(1)) }
    assertFailsWith<FluxTransformerPrepException> { FluxTimestepEmbedding.checked(FloatArray(3_072).also { it[3] = Float.POSITIVE_INFINITY }) }
  }

  @Test fun assemblyIsNoiseFirstReferenceSecondAndDoesNotMutateInputs() = runTest {
    val noiseValues = FloatArray(32_768) { it.toFloat() }
    val referenceValues = FloatArray(32_768) { (100_000 + it).toFloat() }
    val noise = FluxDiagnosticNoiseTokens.checked(noiseValues)
    val reference = FluxReferenceTokens.checked(referenceValues)
    val result = FluxEditingImageTokens.assemble(noise, reference)
    assertEquals(65_536, result.size)
    assertContentEquals(noiseValues, result.copyValues().copyOfRange(0, 32_768))
    assertContentEquals(referenceValues, result.copyValues().copyOfRange(32_768, 65_536))
    assertContentEquals(FloatArray(32_768) { it.toFloat() }, noiseValues)
    result.copyValues()[0] = -1f
    assertEquals(0f, result.copyValues()[0])
  }

  @Test fun runnerUsesExactGraphInputAndOutputOrder() = runTest {
    val calls = mutableListOf<List<FloatArray>>()
    val fake = object : FluxGraphRunner {
      override suspend fun run(modelFile: File, inputs: List<FloatArray>): FluxGraphResult {
        calls += inputs
        return FluxGraphResult(modelFile.name, FluxTransformerPrepContracts.OUTPUTS.map { FloatArray(it.elements) })
      }
    }
    val editing = FluxEditingImageTokens.assemble(FluxSyntheticDiagnosticInputs.syntheticZeroNoiseTokens(), FluxReferenceTokens.checked(FloatArray(32_768) { 1f }))
    val prompt = FluxTextConditioning(FloatArray(3_932_160) { 2f }, listOf(1, 512, 7680), emptyList())
    val timestep = FluxSyntheticDiagnosticInputs.syntheticZeroTimestepEmbedding()
    val report = FluxTransformerPrepRunner(fake).run(File(FluxTransformerPrepContracts.GRAPH), editing, prompt, timestep)
    assertEquals(1, calls.size)
    assertEquals(listOf(65_536, 3_932_160, 3_072), calls.single().map { it.size })
    assertEquals(5, report.outputContracts.size)
  }

  @Test fun outputCountSizeAndFinitenessAreValidatedWithIdentity() = runTest {
    suspend fun failure(outputs: List<FloatArray>) = assertFailsWith<FluxTransformerPrepException> {
      val fake = object : FluxGraphRunner {
        override suspend fun run(modelFile: File, inputs: List<FloatArray>) =
          FluxGraphResult(FluxTransformerPrepContracts.GRAPH, outputs)
      }
      val editing = FluxEditingImageTokens.assemble(FluxSyntheticDiagnosticInputs.syntheticZeroNoiseTokens(), FluxReferenceTokens.checked(FloatArray(32_768)))
      fake.let { FluxTransformerPrepRunner(it).run(File(FluxTransformerPrepContracts.GRAPH), editing, FluxTextConditioning(FloatArray(3_932_160), listOf(1,512,7680), emptyList()), FluxSyntheticDiagnosticInputs.syntheticZeroTimestepEmbedding()) }
    }
    failure(emptyList())
    val outputs = FluxTransformerPrepContracts.OUTPUTS.map { FloatArray(it.elements) }.toMutableList()
    outputs[3][0] = Float.NaN
    assertTrue(failure(outputs).message!!.contains("text modulation"))
  }
}
