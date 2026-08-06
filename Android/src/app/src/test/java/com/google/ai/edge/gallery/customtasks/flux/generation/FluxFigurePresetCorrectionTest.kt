/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.generation

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FluxFigurePresetCorrectionTest {
  @Test fun `every changing built-in supplies all seven attributes`() {
    FluxBuiltInFigurePresets.all.filterNot { it.id == "builtin.preserve" }.forEach {
      val a = it.attributes
      assertTrue(listOf(a.overallBuild, a.shoulders, a.torso, a.waist, a.hips, a.legs, a.heightImpression).all(String::isNotBlank))
    }
  }

  @Test fun `preserve and soft curvy select their matching exclusive actions`() {
    assertEquals(FluxFigureAction.PreserveCurrent, selectBuiltInFigurePreset(FluxBuiltInFigurePresets.byId("builtin.preserve")!!).action)
    val soft = selectBuiltInFigurePreset(FluxBuiltInFigurePresets.byId("builtin.soft_curvy")!!)
    assertEquals(FluxFigureAction.ApplyPreset("builtin.soft_curvy"), soft.action)
    assertEquals("builtin.soft_curvy", soft.presetId)
  }

  @Test fun `selection replaces every stale displayed attribute`() {
    val athletic = selectBuiltInFigurePreset(FluxBuiltInFigurePresets.byId("builtin.athletic")!!)
    val soft = selectBuiltInFigurePreset(FluxBuiltInFigurePresets.byId("builtin.soft_curvy")!!)
    assertEquals(FluxBuiltInFigurePresets.byId("builtin.soft_curvy")!!.attributes, soft.attributes)
    assertFalse(soft.attributes == athletic.attributes)
  }

  @Test fun `all displayed soft curvy attributes reach prioritized figure section`() {
    val preset = FluxBuiltInFigurePresets.byId("builtin.soft_curvy")!!
    val compiled = FluxEditPromptCompiler(FluxBodyTokenCounter { it.split(Regex("\\s+")).size }, 4096).compile(
      FluxFigureEditRequest(FluxFigureAction.ApplyPreset(preset.id), preset.attributes.description()), preset.attributes.description())
    assertTrue("figure_action" in compiled.sectionNames)
    assertFalse("figure_action" in compiled.omittedGeneratedSectionNames)
    listOf(preset.attributes.overallBuild, preset.attributes.shoulders, preset.attributes.torso, preset.attributes.waist,
      preset.attributes.hips, preset.attributes.legs, preset.attributes.heightImpression).forEach { assertTrue(compiled.positivePrompt.contains(it)) }
  }
}
