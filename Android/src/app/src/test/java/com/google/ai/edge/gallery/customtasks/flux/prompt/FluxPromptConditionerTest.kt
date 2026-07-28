/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.prompt

import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxGraphResult
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxGraphRunner
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.yield
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class FluxPromptConditionerTest {
  private val contracts = FluxTextEncoderContracts(2, 2, 2, 4)
  private val graphs = FluxTextEncoderContracts.GRAPH_NAMES.map(::File)
  private val assets = FluxPromptAssets(File("v"), File("m"), File("s"), File("e"), graphs)

  @Test fun executesExactGraphAndInputOrderAndInterleavesTaps() = runBlocking {
    val prepared = prepared()
    val runner = RecordingRunner(contracts.hiddenElements)
    val result = conditioner(runner, prepared).condition("not retained")
    assertEquals(FluxTextEncoderContracts.GRAPH_NAMES, runner.graphs)
    runner.inputs.forEachIndexed { stage, input ->
      assertTrue(input[0] === if (stage == 0) prepared.embeddings else runner.outputs[stage - 1])
      assertTrue(input[1] === prepared.mask)
      assertTrue(input[2] === prepared.cosine)
      assertTrue(input[3] === prepared.sine)
    }
    assertArrayEquals(floatArrayOf(1f, 1f, 2f, 2f, 3f, 3f, 1f, 1f, 2f, 2f, 3f, 3f), result.values, 0f)
    assertEquals(listOf(1, 2, 6), result.shape)
  }

  @Test fun rejectsWrongInputBeforeGraphExecutionAndClosesPreprocessor() = runBlocking {
    val closed = AtomicBoolean()
    val runner = RecordingRunner(contracts.hiddenElements)
    val bad = prepared().copy(embeddings = FloatArray(1))
    assertThrows(FluxTextConditioningException::class.java) { runBlocking { conditioner(runner, bad, closed).condition("x") } }
    assertTrue(runner.graphs.isEmpty())
    assertTrue(closed.get())
  }

  @Test fun rejectsWrongAndNonFiniteOutputsWithoutRunningLaterGraphs() {
    for (bad in listOf(FloatArray(1), FloatArray(contracts.hiddenElements) { Float.NaN })) {
      val runner = RecordingRunner(contracts.hiddenElements, firstOutput = bad)
      assertThrows(FluxTextConditioningException::class.java) { runBlocking { conditioner(runner, prepared()).condition("x") } }
      assertEquals(1, runner.graphs.size)
    }
  }

  @Test fun rejectsNonFiniteEmbeddingBeforeGraphExecution() {
    val runner = RecordingRunner(contracts.hiddenElements)
    val bad = prepared().also { it.embeddings[0] = Float.POSITIVE_INFINITY }
    assertThrows(FluxTextConditioningException::class.java) { runBlocking { conditioner(runner, bad).condition("x") } }
    assertTrue(runner.graphs.isEmpty())
  }

  @Test fun cancellationBeforeEncoderZeroClosesPreprocessor() = runBlocking {
    val closed = AtomicBoolean()
    val runner = RecordingRunner(contracts.hiddenElements)
    val job = Job().also { it.cancel() }
    assertThrows(CancellationException::class.java) {
      runBlocking(job) { conditioner(runner, prepared(), closed).condition("x") }
    }
    assertTrue(runner.graphs.isEmpty())
  }

  @Test fun cancellationBetweenStagesStopsSequence() = runBlocking {
    lateinit var job: Job
    val runner = object : FluxGraphRunner {
      var calls = 0
      override suspend fun run(modelFile: File, inputs: List<FloatArray>): FluxGraphResult {
        calls++
        if (calls == 1) job.cancel()
        return FluxGraphResult(modelFile.name, listOf(FloatArray(contracts.hiddenElements)))
      }
    }
    job = launch { conditioner(runner, prepared()).condition("x") }
    job.join()
    assertEquals(1, runner.calls)
    assertTrue(job.isCancelled)
  }

  @Test fun exactMaskRepresentativeValuesAndHeadExpansion() {
    val mask = FluxPromptConditioner.buildMask(booleanArrayOf(true, false), contracts)
    // q0,k0 valid; q0,k1 future+padded; q1,k0 valid; q1,k1 padded.
    assertArrayEquals(floatArrayOf(0f, -2e9f, 0f, -1e9f), mask.copyOfRange(0, 4), 0f)
    assertArrayEquals(mask.copyOfRange(0, 4), mask.copyOfRange(4, 8), 0f)
  }

  @Test fun exactRotaryValuesAndMemoryAccounting() {
    val cos = FluxPromptConditioner.buildRotary(contracts, false)
    val sin = FluxPromptConditioner.buildRotary(contracts, true)
    assertArrayEquals(floatArrayOf(1f, 1f, 1f, 1f), cos.copyOfRange(0, 4), 0f)
    assertEquals(kotlin.math.cos(1.0).toFloat(), cos[4], 1e-6f)
    assertEquals(kotlin.math.sin(1.0).toFloat(), sin[4], 1e-6f)
    assertEquals(FluxTextEncoderContracts.EXPECTED_PEAK_BYTES, FluxTextEncoderContracts().estimatedPeakBytes())
    assertThrows(ArithmeticException::class.java) { FluxTextEncoderContracts.checkedBytes(Long.MAX_VALUE) }
  }

  private fun prepared() = FluxPreparedText(
    FloatArray(contracts.hiddenElements), FloatArray(contracts.maskElements),
    FloatArray(contracts.rotaryElements), FloatArray(contracts.rotaryElements),
  )

  private fun conditioner(runner: FluxGraphRunner, value: FluxPreparedText, closed: AtomicBoolean = AtomicBoolean()) =
    FluxPromptConditioner(assets, runner, contracts) { _, _ ->
      object : FluxPromptPreprocessor {
        override fun prepare(prompt: String) = value
        override fun close() { closed.set(true) }
      }
    }

  private class RecordingRunner(private val size: Int, private val firstOutput: FloatArray? = null) : FluxGraphRunner {
    val graphs = mutableListOf<String>()
    val inputs = mutableListOf<List<FloatArray>>()
    val outputs = mutableListOf<FloatArray>()
    override suspend fun run(modelFile: File, inputs: List<FloatArray>): FluxGraphResult {
      graphs += modelFile.name
      this.inputs += inputs
      val output = if (outputs.isEmpty() && firstOutput != null) firstOutput else FloatArray(size) { (outputs.size + 1).toFloat() }
      outputs += output
      yield()
      return FluxGraphResult(modelFile.name, listOf(output))
    }
  }
}
