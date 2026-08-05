/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import com.google.ai.edge.gallery.customtasks.flux.diagnostics.FluxPromptInfluenceInterpretation
import com.google.ai.edge.gallery.customtasks.flux.diagnostics.FluxPromptInfluenceMetrics
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxProductionBoundaryPolicy
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxProductionNoiseFactory
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxSeedParser
import com.google.ai.edge.gallery.customtasks.flux.generation.FluxSeedSelection
import java.security.MessageDigest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class FluxPhase2jPromptNoiseDiagnosticsTest {
  @Test fun `production boundary names factory rather than evidence or synthetic fixtures`() {
    val source = FluxProductionBoundaryPolicy.productionInitialNoiseSource
    assertTrue(source.contains("FluxProductionNoiseFactory"))
    assertNotEquals(FluxProductionBoundaryPolicy.DIAGNOSTIC_LATENTS0, source)
    assertNotEquals(FluxProductionBoundaryPolicy.SYNTHETIC_ZERO_NOISE, source)
    assertNotEquals(FluxProductionBoundaryPolicy.DIAGNOSTIC_TIMESTEP, source)
  }

  @Test fun `fixed seeds are deterministic and distinct`() {
    val factory = FluxProductionNoiseFactory()
    val a = factory.create(FluxSeedSelection.Fixed(123)).copyValues()
    val b = factory.create(FluxSeedSelection.Fixed(123)).copyValues()
    val c = factory.create(FluxSeedSelection.Fixed(124)).copyValues()
    assertTrue(a.contentEquals(b))
    assertFalse(a.contentEquals(c))
  }

  @Test fun `random mode selects fresh actual seed`() {
    var next = 10L
    val factory = FluxProductionNoiseFactory { next++ }
    assertEquals(10L, factory.create().seed)
    assertEquals(11L, factory.create().seed)
  }

  @Test fun `malformed fixed seed is rejected`() { assertFailsWith<IllegalArgumentException> { FluxSeedParser.parseFixed("not-a-long") } }

  @Test fun `prompt A and B receive byte-identical defensive copies`() {
    val latents = FluxProductionNoiseFactory().create(FluxSeedSelection.Fixed(7))
    val a = latents.copyValues(); val b = latents.copyValues(); a[0] = 99f
    assertNotEquals(a[0].toBits(), b[0].toBits())
    assertTrue(latents.copyValues().contentEquals(b))
  }

  @Test fun `token comparison detects changed positions`() {
    val diff = FluxPromptInfluenceMetrics.compareTokens(longArrayOf(1,2,3), longArrayOf(1,9,3))
    assertEquals(1, diff.differingPositions)
  }

  @Test fun `fp32 hashing is explicit little endian and metrics are deterministic`() {
    val onePointZeroLittleEndianBytes = byteArrayOf(0x00, 0x00, 0x80.toByte(), 0x3f)
    val expectedSha = MessageDigest.getInstance("SHA-256")
      .digest(onePointZeroLittleEndianBytes)
      .joinToString("") { "%02x".format(it) }
    assertEquals("e00e5eb9444182f352323374ef4e08ebcb784725fdd4fd612d7730540b3e0c8c", expectedSha)
    val s = FluxPromptInfluenceMetrics.summarizeFp32(floatArrayOf(1f))
    assertEquals(expectedSha, s.sha256LittleEndian)
    val same = FluxPromptInfluenceMetrics.compareFp32(floatArrayOf(1f,2f), floatArrayOf(1f,2f))
    assertEquals(0.0, same.mae); assertEquals(0.0, same.rmse); assertEquals(1.0, same.cosineSimilarity)
    val different = FluxPromptInfluenceMetrics.compareFp32(floatArrayOf(1f,2f), floatArrayOf(2f,4f))
    assertTrue(different.different); assertTrue(different.mae > 0.0)
  }

  @Test fun `non finite and zero norm cosine handled`() {
    assertFailsWith<IllegalArgumentException> { FluxPromptInfluenceMetrics.summarizeFp32(floatArrayOf(Float.NaN)) }
    assertNull(FluxPromptInfluenceMetrics.compareFp32(floatArrayOf(0f), floatArrayOf(1f)).cosineSimilarity)
  }

  @Test fun `bitmap comparison detects identical and different pixels`() {
    val same = FluxPromptInfluenceMetrics.compareArgb(1, 1, intArrayOf(-1), intArrayOf(-1))
    assertEquals(0, same.differingPixels)
    val diff = FluxPromptInfluenceMetrics.compareArgb(1, 1, intArrayOf(0xff000000.toInt()), intArrayOf(0xffffffff.toInt()))
    assertEquals(1, diff.differingPixels); assertTrue(diff.rgbRmse > 0.0)
  }


  @Test fun `debug prompt comparison is sequential sanitized and release has no comparison action`() {
    val debug = java.io.File("src/debug/java/com/google/ai/edge/gallery/customtasks/flux/FluxPromptInfluenceComparison.kt").readText()
    assertTrue(debug.contains("runOne(\"A\""))
    assertTrue(debug.contains("runOne(\"B\""))
    assertTrue(debug.indexOf("runOne(\"A\"") < debug.indexOf("runOne(\"B\""))
    assertTrue(debug.contains("contentEquals(noiseB)"))
    assertTrue(debug.contains("FluxReferenceImageSourceStager"))
    assertTrue(debug.contains("withModelFilesLocked"))
    assertTrue(debug.contains("no prompt text, URI, path, image bytes"))
    val release = java.io.File("src/release/java/com/google/ai/edge/gallery/customtasks/flux/FluxDeveloperVerificationSection.kt").readText()
    assertFalse(release.contains("Run comparison"))
    assertFalse(release.contains("Prompt influence"))
  }

  @Test fun `interpretation branches are deterministic`() {
    val sameTokens = FluxPromptInfluenceMetrics.compareTokens(longArrayOf(1), longArrayOf(1))
    val differentTokens = FluxPromptInfluenceMetrics.compareTokens(longArrayOf(1), longArrayOf(2))
    val sameTensor = FluxPromptInfluenceMetrics.compareFp32(floatArrayOf(1f), floatArrayOf(1f))
    val differentTensor = FluxPromptInfluenceMetrics.compareFp32(floatArrayOf(1f), floatArrayOf(2f))
    val sameBitmap = FluxPromptInfluenceMetrics.compareArgb(1, 1, intArrayOf(-1), intArrayOf(-1))
    assertEquals(FluxPromptInfluenceInterpretation.PROMPT_DIFFERENCE_LOST_BEFORE_TEXT_CONDITIONING, FluxPromptInfluenceMetrics.interpret(sameTokens, differentTensor, differentTensor, null))
    assertEquals(FluxPromptInfluenceInterpretation.PROMPT_DIFFERENCE_LOST_BEFORE_TEXT_CONDITIONING, FluxPromptInfluenceMetrics.interpret(differentTokens, sameTensor, differentTensor, null))
    assertEquals(FluxPromptInfluenceInterpretation.PROMPT_DIFFERENCE_LOST_DURING_TRANSFORMER, FluxPromptInfluenceMetrics.interpret(differentTokens, differentTensor, sameTensor, null))
    assertEquals(FluxPromptInfluenceInterpretation.FINAL_BITMAPS_NEARLY_IDENTICAL, FluxPromptInfluenceMetrics.interpret(differentTokens, differentTensor, differentTensor, sameBitmap))
    assertEquals(FluxPromptInfluenceInterpretation.PROMPT_INFLUENCE_OBSERVED, FluxPromptInfluenceMetrics.interpret(differentTokens, differentTensor, differentTensor, null))
  }
}
