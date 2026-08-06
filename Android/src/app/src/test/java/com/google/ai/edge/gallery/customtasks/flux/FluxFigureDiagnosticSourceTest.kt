/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import java.io.File
import org.junit.Test

class FluxFigureDiagnosticSourceTest {
  @Test fun `interpretation separates missing compilation and numerical influence`() {
    assertEquals("FIGURE_PRESET_NOT_COMPILED", interpretFigureInfluence(false, "builtin.soft_curvy", 4, true, true, 90.0))
    assertEquals("FIGURE_PROMPT_INFLUENCE_NOT_OBSERVED", interpretFigureInfluence(true, "builtin.soft_curvy", 0, false, false, 0.0))
    assertEquals("FIGURE_PROMPT_INFLUENCE_OBSERVED", interpretFigureInfluence(true, "builtin.soft_curvy", 4, true, true, 90.0))
  }

  @Test fun `debug A B is fixed seed same reference and release has no verification or prompt disclosure`() {
    val root = File("src")
    val debug = File(root, "debug/java/com/google/ai/edge/gallery/customtasks/flux/FluxPromptInfluenceComparison.kt").readText()
    val release = File(root, "release/java/com/google/ai/edge/gallery/customtasks/flux/FluxDeveloperVerificationSection.kt").readText()
    assertTrue(debug.contains("noiseA = initial.copyValues()"))
    assertTrue(debug.contains("noiseB = initial.copyValues()"))
    assertTrue(debug.contains("withStagedSource"))
    assertTrue(debug.contains("figureRequest.copy"))
    assertFalse(release.contains("Verify figure preset influence"))
    assertFalse(release.contains("positivePrompt"))
  }
}
