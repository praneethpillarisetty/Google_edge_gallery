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
  private fun count(text: String, value: String) = Regex(Regex.escape(value)).findAll(text).count()

  @Test fun simpleHasOneInvariantOneRealismAndOneLiteralInstruction() {
    val literal = "recolor the cafe\u0301 sign"
    val request = FluxSimpleEditRequest("  $literal  ")
    val prompt = compiler.compile(request).positivePrompt
    assertEquals(FluxRealismProfile.NATURAL_PHOTO, request.realismProfile)
    assertEquals(1, count(prompt, FluxEditPromptCompiler.ANATOMY_INVARIANT))
    assertEquals(1, count(prompt, FluxEditPromptCompiler.NATURAL_PHOTO))
    assertEquals(1, count(prompt, literal))
    assertFalse(prompt.contains("negative prompt", true))
    assertFalse(prompt.contains("<|im_start|>"))
  }

  @Test fun figureHasOneInvariantAndExactlyOneSelectedRealismClause() {
    FluxRealismProfile.entries.forEach { profile ->
      val prompt = compiler.compile(FluxFigureEditRequest(FluxFigureAction.PreserveCurrent, realismProfile = profile)).positivePrompt
      assertEquals(1, count(prompt, FluxEditPromptCompiler.ANATOMY_INVARIANT))
      assertEquals(1, listOf(FluxEditPromptCompiler.NATURAL_PHOTO, FluxEditPromptCompiler.EDITORIAL_PHOTO, FluxEditPromptCompiler.CINEMATIC_PHOTO).count(prompt::contains))
    }
  }

  @Test fun neverEmitsMalformedAnatomyOrInflatedQualityLists() {
    val prompt = compiler.compile(FluxSimpleEditRequest("Change the shirt")).positivePrompt.lowercase()
    listOf("extra limbs", "bad anatomy", "malformed hands", "mutation", "deformed", "disfigured", "masterpiece", "award-winning", "flawless", "8k", "16k", "hyper-detailed", "ultra-realistic").forEach { assertFalse(prompt.contains(it), it) }
  }

  @Test fun framingRulesAreContextRelevant() {
    val full = compiler.compile(FluxSimpleEditRequest("Make this a full-body portrait")).positivePrompt
    assertContains(full, FluxEditPromptCompiler.FULL_BODY)
    val waist = compiler.compile(FluxSimpleEditRequest("Use a waist-up crop")).positivePrompt
    assertContains(waist, FluxEditPromptCompiler.WAIST_UP)
    assertFalse(waist.contains(FluxEditPromptCompiler.FULL_BODY))
    val close = compiler.compile(FluxSimpleEditRequest("Make a close-up portrait")).positivePrompt
    assertContains(close, FluxEditPromptCompiler.CLOSE_UP)
    assertFalse(close.contains(FluxEditPromptCompiler.FULL_BODY))
    assertFalse(close.contains(FluxEditPromptCompiler.WAIST_UP))
  }

  @Test fun handAndOverlapRulesAreConditional() {
    val unrelated = compiler.compile(FluxSimpleEditRequest("Change the wall color")).positivePrompt
    assertFalse(unrelated.contains(FluxEditPromptCompiler.HANDS))
    assertFalse(unrelated.contains(FluxEditPromptCompiler.LIMB_OVERLAP))
    assertContains(compiler.compile(FluxSimpleEditRequest("Pose with hands holding a cup")).positivePrompt, FluxEditPromptCompiler.HANDS)
    assertContains(compiler.compile(FluxSimpleEditRequest("Pose with crossed arms")).positivePrompt, FluxEditPromptCompiler.LIMB_OVERLAP)
  }

  @Test fun onlyTheRelevantSpecificPoseRuleIsIncluded() {
    val cases = listOf(
      "Change the pose to sitting" to FluxEditPromptCompiler.SITTING,
      "Change the pose to standing" to FluxEditPromptCompiler.STANDING,
      "Change the pose to walking" to FluxEditPromptCompiler.GAIT,
      "Change the pose to raised arms" to FluxEditPromptCompiler.RAISED_ARMS,
      "Change the pose to leaning" to FluxEditPromptCompiler.LEANING,
    )
    val specifics = cases.map { it.second }
    cases.forEach { (instruction, expected) ->
      val prompt = compiler.compile(FluxSimpleEditRequest(instruction)).positivePrompt
      assertContains(prompt, FluxEditPromptCompiler.POSE_COHERENCE)
      assertContains(prompt, expected)
      specifics.filterNot { it == expected }.forEach { assertFalse(prompt.contains(it)) }
    }
  }

  @Test fun figureActionsRemainExclusiveAndOutfitFitsNaturally() {
    val current = compiler.compile(FluxFigureEditRequest(FluxFigureAction.PreserveCurrent)).positivePrompt
    assertContains(current, FluxEditPromptCompiler.PRESERVE_CURRENT)
    assertFalse(current.contains(FluxEditPromptCompiler.FIGURE_CHANGE))
    val description = "balanced natural proportions"
    val changed = compiler.compile(FluxFigureEditRequest(FluxFigureAction.ApplyPreset("builtin.balanced"), description), description).positivePrompt
    assertEquals(1, count(changed, FluxEditPromptCompiler.FIGURE_CHANGE))
    assertEquals(1, count(changed, description))
    assertContains(changed, FluxEditPromptCompiler.CLOTHING_BODY)
  }

  @Test fun changedOutfitAndLocationDoNotReceiveContradictoryPreservation() {
    val prompt = compiler.compile(FluxSimpleEditRequest("Change the outfit and move to a beach", preserveBackground = true)).positivePrompt
    assertFalse(prompt.contains("Preserve the reference outfit"))
    assertFalse(prompt.contains("Preserve the reference location"))
    assertContains(prompt, FluxEditPromptCompiler.NEW_LOCATION)
  }

  @Test fun figureFaceRulesSupportExternalWorkflowWithoutReplacementRequest() {
    val regular = compiler.compile(FluxFigureEditRequest(FluxFigureAction.PreserveCurrent)).positivePrompt
    assertContains(regular, FluxEditPromptCompiler.FACE_COHERENCE)
    listOf("new woman", "celebrity", "identity fusion", "face replacement", "new face", "different identity").forEach { assertFalse(regular.contains(it, true)) }
    val contextual = compiler.compile(FluxFigureEditRequest(FluxFigureAction.PreserveCurrent, additionalInstruction = "Keep the turned face partially hidden" , visualContext = FluxEditVisualContext(faceTurned = true, faceOccluded = true))).positivePrompt
    assertContains(contextual, FluxEditPromptCompiler.TURNED_FACE)
    assertContains(contextual, FluxEditPromptCompiler.FACE_OCCLUSION)
  }

  @Test fun generatedSectionsAreUniqueAndOrderedAfterLiteralRequest() {
    val literal = "Change the lighting"
    val compiled = compiler.compile(FluxSimpleEditRequest(literal))
    assertEquals(compiled.sectionNames.size, compiled.sectionNames.toSet().size)
    assertTrue(compiled.positivePrompt.indexOf(literal) < compiled.positivePrompt.indexOf(FluxEditPromptCompiler.ANATOMY_INVARIANT))
    assertEquals(1, count(compiled.positivePrompt, literal))
  }

  @Test fun emptySimpleAndCustomAreRejectedAndLocksDefaultOn() {
    assertFailsWith<IllegalArgumentException> { compiler.compile(FluxSimpleEditRequest(" \n ")) }
    assertFailsWith<IllegalArgumentException> { compiler.compile(FluxFigureEditRequest(FluxFigureAction.Custom("  "))) }
    val request = FluxFigureEditRequest(FluxFigureAction.PreserveCurrent, additionalInstruction = "Change the hair and camera angle")
    assertTrue(request.preserveOutfit && request.preservePose && request.preserveBackground && request.preserveCamera && request.preserveLighting && request.preserveHair && request.preserveMakeupAndExpression && request.preserveAccessories)
    assertEquals(setOf("hairstyle", "framing and camera"), compiler.compile(request).conflicts.map { it.lockName }.toSet())
  }

  @Test fun presetsRemainNeutralAndMeasurementFree() {
    assertEquals(6, FluxBuiltInFigurePresets.all.size)
    assertTrue(FluxBuiltInFigurePresets.all.all { it.builtIn })
    assertTrue(FluxBuiltInFigurePresets.all.single { it.id == "builtin.preserve" }.attributes.description().isBlank())
    assertTrue(FluxBuiltInFigurePresets.all.filterNot { it.id == "builtin.preserve" }.all { it.attributes.description().isNotBlank() })
    assertTrue(FluxBuiltInFigurePresets.all.none { Regex("\\d+-\\d+-\\d+").containsMatchIn(it.attributes.description()) })
  }
}
