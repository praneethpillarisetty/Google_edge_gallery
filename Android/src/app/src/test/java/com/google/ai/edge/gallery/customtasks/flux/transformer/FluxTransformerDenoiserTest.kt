/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.transformer

import com.google.ai.edge.gallery.customtasks.flux.FLUX_MODEL_REVISION
import com.google.ai.edge.gallery.customtasks.flux.FLUX_REPOSITORY
import com.google.ai.edge.gallery.customtasks.flux.FluxModelManifest
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceTokens
import com.google.ai.edge.gallery.customtasks.flux.prompt.FluxTextConditioning
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxGraphResult
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxGraphRunner
import java.io.File
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.assertEquals
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.runBlocking
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class FluxTransformerDenoiserTest {
  @get:Rule val temporary = TemporaryFolder()

  @Test fun `four steps select exact evidence rows preserve graph order wiring and Euler update`() = runBlocking {
    val fixture = fixture()
    val runner = RecordingRunner(fixture.evidence)
    val clock = AtomicLong()
    val result = FluxTransformerDenoiser(runner) { clock.addAndGet(1_000_000) }
      .run(fixture.root, fixture.manifest, fixture.evidence, fixture.evidence.initialLatents(), fixture.prompt, fixture.reference)
    assertEquals(4, result.completedSteps)
    assertEquals(32, runner.calls.size)
    assertEquals(List(4) { FluxTransformerDenoisingContracts.GRAPH_ORDER }.flatten(), runner.calls)
    assertEquals((0 until 4).map { fixture.evidence.timestep(it)[0] }, runner.timestepFirst)
    var expected = fixture.evidence.initialLatents()[0]
    runner.prepLatentFirst.forEachIndexed { step, observed ->
      assertEquals(expected, observed)
      expected += fixture.evidence.dsigma(step)
    }
    assertEquals(4, result.stepDurationsMillis.size)
    assertEquals(32, result.graphTimings.size)
    assertTrue(runner.authoritativeWiringObserved)
    val expectedFinal = fixture.evidence.initialLatents().also { values -> repeat(4) { step ->
      for (index in values.indices) values[index] += fixture.evidence.dsigma(step)
    } }
    assertContentEquals(expectedFinal, result.copyFinalLatents())
    assertTrue(result.copyFinalLatents().none { it == 5f }, "reference-token marker must not escape")
  }

  @Test fun `typed final latent handoff validates and defensively owns values`() {
    val original = FloatArray(32_768) { it.toFloat() }
    val result = FluxTransformerCoreResult.checked(original, emptyList(), 4, true, emptyList(), emptyList())
    original[0] = -1f
    assertEquals(0f, result.copyFinalLatents()[0])
    val first = result.copyFinalLatents(); first[1] = -1f
    assertEquals(1f, result.copyFinalLatents()[1])
    assertFailsWith<IllegalArgumentException> { FluxTransformerCoreResult.checked(FloatArray(1), emptyList(), 4, true, emptyList(), emptyList()) }
    assertFailsWith<IllegalArgumentException> { FluxTransformerCoreResult.checked(FloatArray(32_768).also { it[0] = Float.NaN }, emptyList(), 4, false, emptyList(), emptyList()) }
  }

  @Test fun `wrong output count shape and non finite output stop later graphs`() = runBlocking {
    for (failure in Failure.values()) {
      val fixture = fixture(); val runner = RecordingRunner(fixture.evidence, failure = failure)
      val error = assertFailsWith<FluxTransformerDenoisingException> {
        FluxTransformerDenoiser(runner).run(fixture.root, fixture.manifest, fixture.evidence, fixture.evidence.initialLatents(), fixture.prompt, fixture.reference)
      }
      assertTrue(error.message!!.contains("Step 1")); assertTrue(error.message!!.contains("kce_prep.tflite"))
      assertEquals(listOf("kce_prep.tflite"), runner.calls)
    }
  }

  @Test fun `graph failure prevents later graph and later step`() = runBlocking {
    val fixture = fixture(); val runner = RecordingRunner(fixture.evidence, failInvocation = 3)
    assertFailsWith<FluxTransformerDenoisingException> { FluxTransformerDenoiser(runner).run(fixture.root, fixture.manifest, fixture.evidence, fixture.evidence.initialLatents(), fixture.prompt, fixture.reference) }
    assertEquals(3, runner.calls.size)
  }

  @Test fun `cancellation between graphs prevents next graph`() = runBlocking {
    val fixture = fixture(); val runner = RecordingRunner(fixture.evidence)
    val job = async {
      FluxTransformerDenoiser(runner).run(fixture.root, fixture.manifest, fixture.evidence, fixture.evidence.initialLatents(), fixture.prompt, fixture.reference) { progress ->
        if (progress.completedGraphs == 1) throw CancellationException("test")
      }
    }
    assertFailsWith<CancellationException> { job.await() }
    assertEquals(1, runner.calls.size)
  }

  @Test fun `cancellation before run prevents graph creation`() = runBlocking {
    val fixture = fixture(); val runner = RecordingRunner(fixture.evidence)
    val job = async { FluxTransformerDenoiser(runner).run(fixture.root, fixture.manifest, fixture.evidence, fixture.evidence.initialLatents(), fixture.prompt, fixture.reference) }
    job.cancelAndJoin()
    assertTrue(runner.calls.size < 32)
  }

  @Test fun `complete transformer verification gate serializes concurrent jobs`() = runBlocking {
    val active = AtomicInteger(); val maximum = AtomicInteger()
    suspend fun gated() = FluxTransformerExecutionGate.exclusive {
      maximum.updateAndGet { maxOf(it, active.incrementAndGet()) }
      delay(25)
      active.decrementAndGet()
    }
    val first = async { gated() }; val second = async { gated() }
    first.await(); second.await()
    assertEquals(1, maximum.get())
  }

  private fun fixture(): Fixture {
    val root = temporary.newFolder()
    FluxTransformerDenoisingContracts.GRAPH_ORDER.forEach { File(root, it).writeBytes(byteArrayOf(1)) }
    val manifest = FluxModelManifest(1, FLUX_REPOSITORY, FLUX_MODEL_REVISION, "test", FluxTransformerDenoisingContracts.GRAPH_ORDER)
    val assets = File("src/main/assets/flux/phase2g")
    val evidence = FluxPhase2gEvidenceParser.parse(open = { assets.resolve(it).inputStream() })
    return Fixture(root, manifest, evidence, FluxTextConditioning(FloatArray(3_932_160), listOf(1, 512, 7680), emptyList()), FluxReferenceTokens.checked(FloatArray(32_768) { 5f }))
  }
  private data class Fixture(val root: File, val manifest: FluxModelManifest, val evidence: FluxPhase2gEvidence, val prompt: FluxTextConditioning, val reference: FluxReferenceTokens)
}

private enum class Failure { COUNT, SHAPE, NONFINITE }
private class RecordingRunner(
  private val evidence: FluxPhase2gEvidence,
  private val failure: Failure? = null,
  private val failInvocation: Int = -1,
) : FluxGraphRunner {
  val calls = mutableListOf<String>()
  val timestepFirst = mutableListOf<Float>()
  val prepLatentFirst = mutableListOf<Float>()
  var authoritativeWiringObserved = true
  override suspend fun run(modelFile: File, inputs: List<FloatArray>): FluxGraphResult {
    val name = modelFile.name
    calls += name
    if (calls.size == failInvocation) error("fake failure")
    val step = calls.size / 8 + if (calls.size % 8 == 0) 0 else 1
    when {
      name == "kce_prep.tflite" -> {
        prepLatentFirst += inputs[0][0]; timestepFirst += inputs[2][0]
        authoritativeWiringObserved = authoritativeWiringObserved && inputs[0][32_768] == 5f && inputs[2][0] == evidence.timestep(step - 1)[0]
      }
      name.startsWith("kce_double") -> authoritativeWiringObserved = authoritativeWiringObserved && inputs[2].size == 65_536 && inputs[4].size == 18_432
      name == "kce_single0.tflite" -> authoritativeWiringObserved = authoritativeWiringObserved && inputs[0][0] == 20f && inputs[0][1_572_864] == 10f
      name.startsWith("kce_single") -> authoritativeWiringObserved = authoritativeWiringObserved && inputs[0][0] == 20f && inputs[0][1_572_864] == 20f
      name == "kce_final.tflite" -> authoritativeWiringObserved = authoritativeWiringObserved && inputs[1][0] == evidence.timestep(step - 1)[0]
    }
    if (failure != null && calls.size == 1) return when (failure) {
      Failure.COUNT -> FluxGraphResult(name, emptyList())
      Failure.SHAPE -> FluxGraphResult(name, listOf(FloatArray(1), FloatArray(1), FloatArray(1), FloatArray(1), FloatArray(1)))
      Failure.NONFINITE -> FluxGraphResult(name, listOf(FloatArray(1_572_864) { Float.NaN }, FloatArray(1_572_864), FloatArray(18_432), FloatArray(18_432), FloatArray(9_216)))
    }
    val sizes = FluxTransformerDenoisingContracts.outputElements.getValue(name)
    val markers = when {
      name == "kce_prep.tflite" -> listOf(10f, 20f, 30f, 40f, 50f)
      name.startsWith("kce_double") -> listOf(10f, 20f)
      name.startsWith("kce_single") -> listOf(20f)
      else -> listOf(1f)
    }
    return FluxGraphResult(name, sizes.mapIndexed { index, size -> FloatArray(size) { markers[index] } })
  }
}
