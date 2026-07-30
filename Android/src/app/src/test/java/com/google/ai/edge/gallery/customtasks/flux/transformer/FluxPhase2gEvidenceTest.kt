/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.transformer

import java.io.ByteArrayInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotSame
import org.junit.Test

class FluxPhase2gEvidenceTest {
  private val directory = File("src/main/assets/flux/phase2g")
  private fun parse(overrides: Map<String, ByteArray> = emptyMap()) =
    FluxPhase2gEvidenceParser.parse { name -> ByteArrayInputStream(overrides[name] ?: File(directory, name).readBytes()) }

  @Test fun authoritativeEvidenceHasExactContractsAndIsDeterministic() {
    val first = parse(); val second = parse()
    assertEquals(32_768, first.initialLatents().size)
    assertEquals(3_072, first.timestep(3).size)
    assertEquals(65_536, first.cos().size)
    assertEquals(65_536, first.sin().size)
    assertContentEquals(first.initialLatents(), second.initialLatents())
    repeat(4) { assert(first.dsigma(it) < 0f) }
  }

  @Test fun arraysAreDefensivelyOwned() {
    val evidence = parse(); val one = evidence.initialLatents(); val two = evidence.initialLatents()
    assertNotSame(one, two); val expected = two[0]; one[0] = expected + 1f; assertEquals(expected, evidence.initialLatents()[0])
  }

  @Test fun unknownMetadataIsRejected() {
    val text = File(directory, "flux_phase2g_constants.json").readText().replaceFirst("{", "{\"unknown\":true,")
    assertFailsWith<FluxPhase2gEvidenceException> { parse(mapOf("flux_phase2g_constants.json" to text.encodeToByteArray())) }
  }

  @Test fun sizeAndHashAreRejected() {
    assertFailsWith<FluxPhase2gEvidenceException> { parse(mapOf("dsigma.bin" to ByteArray(15))) }
    val corrupt = File(directory, "dsigma.bin").readBytes().also { it[0] = (it[0].toInt() xor 1).toByte() }
    assertFailsWith<FluxPhase2gEvidenceException> { parse(mapOf("dsigma.bin" to corrupt)) }
  }

  @Test fun littleEndianNonFiniteAndNonNegativeAreRejected() {
    val bytes = File(directory, "latents0.bin").readBytes()
    assertEquals(ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).float, parse().initialLatents()[0])
    val nonFinite = bytes.copyOf().also { ByteBuffer.wrap(it).order(ByteOrder.LITTLE_ENDIAN).putFloat(Float.NaN) }
    assertFailsWith<FluxPhase2gEvidenceException> { parse(mapOf("latents0.bin" to nonFinite)) }
    val sigma = File(directory, "dsigma.bin").readBytes().also { ByteBuffer.wrap(it).order(ByteOrder.LITTLE_ENDIAN).putFloat(0f) }
    assertFailsWith<FluxPhase2gEvidenceException> { parse(mapOf("dsigma.bin" to sigma)) }
  }

  @Test fun graphOrderIsPinnedCompanionOrder() {
    assertEquals(listOf("kce_prep.tflite", "kce_double0.tflite", "kce_double1.tflite", "kce_single0.tflite", "kce_single1.tflite", "kce_single2.tflite", "kce_single3.tflite", "kce_final.tflite"), FluxTransformerDenoisingContracts.GRAPH_ORDER)
  }
}
