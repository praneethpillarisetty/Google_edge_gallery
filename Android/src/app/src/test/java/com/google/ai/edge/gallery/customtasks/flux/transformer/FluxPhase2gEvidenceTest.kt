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
import kotlin.test.assertSame
import org.junit.Test

class FluxPhase2gEvidenceTest {
  private val directory = File("src/main/assets/flux/phase2g")
  private fun files() = directory.listFiles()!!.associate { it.name to it.readBytes() }
  private fun parse(values: Map<String, ByteArray> = files()) = FluxPhase2gEvidenceParser.parse(open = { ByteArrayInputStream(values.getValue(it)) })
  private fun metadata(transform: (String) -> String) = files().toMutableMap().also { map ->
    map["flux_phase2g_constants.json"] = transform(map.getValue("flux_phase2g_constants.json").decodeToString()).encodeToByteArray()
  }

  @Test fun `supplied strict metadata and every exact tensor contract validate`() {
    val evidence = parse()
    assertEquals(32_768, evidence.initialLatents().size)
    assertEquals(3_072, evidence.timestep(0).size)
    assertEquals(3_072, evidence.timestep(3).size)
    assertEquals(65_536, evidence.cos().size)
    assertEquals(65_536, evidence.sin().size)
    repeat(4) { assert(evidence.dsigma(it) < 0f) }
  }

  @Test fun `unknown and missing metadata fields are rejected`() {
    assertFailsWith<FluxPhase2gEvidenceException> { parse(metadata { it.replaceFirst("{", "{\"unknown\":true,") }) }
    assertFailsWith<FluxPhase2gEvidenceException> { parse(metadata { it.replace(Regex("\\s*\"purpose\"\\s*:\\s*\"[^\"]+\",?"), "") }) }
  }

  @Test fun `provenance mismatch and non UTC timestamp are rejected`() {
    assertFailsWith<FluxPhase2gEvidenceException> { parse(metadata { it.replace("\"seed\": 1234", "\"seed\": 1235") }) }
    assertFailsWith<FluxPhase2gEvidenceException> { parse(metadata { it.replace("+00:00", "+01:00") }) }
  }

  @Test fun `every truncated trailing and hash changed artifact is rejected`() {
    listOf("latents0.bin", "temb.bin", "dsigma.bin", "cos.bin", "sin.bin").forEach { name ->
      val original = files()
      assertFailsWith<FluxPhase2gEvidenceException> { parse(original.toMutableMap().also { it[name] = original.getValue(name).copyOf(original.getValue(name).size - 1) }) }
      assertFailsWith<FluxPhase2gEvidenceException> { parse(original.toMutableMap().also { it[name] = original.getValue(name) + 0 }) }
      assertFailsWith<FluxPhase2gEvidenceException> { parse(original.toMutableMap().also { map -> map[name] = original.getValue(name).copyOf().also { it[0] = (it[0].toInt() xor 1).toByte() } }) }
    }
  }

  @Test fun `little endian parsing is explicit and non finite values are rejected`() {
    val original = files()
    val first = ByteBuffer.wrap(original.getValue("latents0.bin")).order(ByteOrder.LITTLE_ENDIAN).float
    assertEquals(first, parse(original).initialLatents()[0])
    val changed = original.toMutableMap()
    changed["latents0.bin"] = original.getValue("latents0.bin").copyOf().also { ByteBuffer.wrap(it).order(ByteOrder.LITTLE_ENDIAN).putFloat(0, Float.NaN) }
    assertFailsWith<FluxPhase2gEvidenceException> { parse(changed) }
  }

  @Test fun `zero positive and non finite dsigma are rejected`() {
    assertFailsWith<FluxPhase2gEvidenceException> { FluxPhase2gEvidenceParser.validateDsigma(floatArrayOf(-1f, -1f, 0f, -1f)) }
    assertFailsWith<FluxPhase2gEvidenceException> { FluxPhase2gEvidenceParser.validateDsigma(floatArrayOf(-1f, 1f, -1f, -1f)) }
    assertFailsWith<FluxPhase2gEvidenceException> { FluxPhase2gEvidenceParser.validateDsigma(floatArrayOf(-1f, Float.NaN, -1f, -1f)) }
  }

  @Test fun `successful load is cached failed load is not cached and access is defensive`() {
    val source = files()
    var opens = 0
    val loader = FluxPhase2gEvidenceSourceLoader { name -> opens++; ByteArrayInputStream(source.getValue(name)) }
    val first = loader.load(); val successfulOpens = opens; val second = loader.load()
    assertSame(first, second); assertEquals(successfulOpens, opens)
    val a = first.initialLatents(); val b = first.initialLatents(); assertNotSame(a, b)
    val value = b[0]; a[0] += 1f; assertEquals(value, first.initialLatents()[0])

    var invalid = true
    val retrying = FluxPhase2gEvidenceSourceLoader { name ->
      val bytes = if (invalid && name == "dsigma.bin") ByteArray(15) else source.getValue(name)
      ByteArrayInputStream(bytes)
    }
    assertFailsWith<FluxPhase2gEvidenceException> { retrying.load() }
    invalid = false
    assertContentEquals(first.initialLatents(), retrying.load().initialLatents())
  }

  @Test fun `repeated uncached parsing is deterministic and cancellation is checked during streams`() {
    assertContentEquals(parse().initialLatents(), parse().initialLatents())
    var checks = 0
    FluxPhase2gEvidenceParser.parse({ ByteArrayInputStream(files().getValue(it)) }) { checks++ }
    assert(checks > 100)
  }
}
