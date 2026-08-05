/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.generation

import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class FluxEditPromptCompilerTest {
  private val compiler = FluxEditPromptCompiler()

  @Test fun simpleIsOnePositiveLiteralPromptWithoutWrapper() {
    val decomposed = "recolor the cafe\u0301 sign"
    val prompt = compiler.compile(FluxSimpleEditRequest("  $decomposed  ")).positivePrompt
    assertContains(prompt, decomposed)
    assertFalse(prompt.contains("negative prompt", true))
    assertFalse(prompt.contains("<|im_start|>"))
    assertFalse(prompt.contains("new woman", true))
  }

  @Test fun emptySimpleAndCustomAreRejected() {
    assertFailsWith<IllegalArgumentException> { compiler.compile(FluxSimpleEditRequest(" \n ")) }
    assertFailsWith<IllegalArgumentException> { compiler.compile(FluxFigureEditRequest(FluxFigureAction.Custom("  "))) }
  }

  @Test fun intentBoundariesAndPreservationAvoidContradictions() {
    val changed = compiler.compile(FluxSimpleEditRequest("Change the outfit and move to a beach", preserveBackground = true))
    assertEquals(setOf(FluxPromptIntent.OUTFIT, FluxPromptIntent.BACKGROUND), changed.intents)
    assertFalse(changed.positivePrompt.contains("reference outfit"))
    assertFalse(changed.positivePrompt.contains("reference location"))
    assertTrue(FluxPromptIntent.POSE !in FluxPromptIntentDetector.detect("compose a still life"))
    assertContains(compiler.compile(FluxSimpleEditRequest("remove a sign", preserveBackground = true)).positivePrompt, "reference location")
  }

  @Test fun preserveCurrentAndPresetHaveExclusiveFigureClauses() {
    val current = compiler.compile(FluxFigureEditRequest(FluxFigureAction.PreserveCurrent)).positivePrompt
    assertContains(current, "current overall silhouette")
    assertFalse(current.contains("Modify only the subject's body proportions"))
    val description = "balanced natural proportions"
    val preset = compiler.compile(FluxFigureEditRequest(FluxFigureAction.ApplyPreset("builtin.balanced"), description), description).positivePrompt
    assertEquals(1, Regex("Modify only the subject's body proportions").findAll(preset).count())
    assertEquals(1, Regex(Regex.escape(description)).findAll(preset).count())
    assertContains(preset, "minimum fit deformation")
    assertContains(preset, "head orientation")
  }

  @Test fun everyLockDefaultsOnAndConflictsAreExplicit() {
    val request = FluxFigureEditRequest(FluxFigureAction.PreserveCurrent, additionalInstruction = "Change the hair and camera angle")
    assertTrue(request.preserveOutfit && request.preservePose && request.preserveBackground && request.preserveCamera && request.preserveLighting && request.preserveHair && request.preserveMakeupAndExpression && request.preserveAccessories)
    assertEquals(setOf("hairstyle", "framing and camera"), compiler.compile(request).conflicts.map { it.lockName }.toSet())
  }

  @Test fun presetsAreNeutralDescriptiveAndBuiltInsImmutableByType() {
    assertEquals(6, FluxBuiltInFigurePresets.all.size)
    assertTrue(FluxBuiltInFigurePresets.all.all { it.builtIn && it.attributes.description().isNotBlank() })
    assertTrue(FluxBuiltInFigurePresets.all.none { Regex("\\d+-\\d+-\\d+").containsMatchIn(it.attributes.description()) })
  }
}
