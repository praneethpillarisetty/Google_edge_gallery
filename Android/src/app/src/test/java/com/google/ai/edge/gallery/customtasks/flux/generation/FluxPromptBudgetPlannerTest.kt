package com.google.ai.edge.gallery.customtasks.flux.generation

import kotlin.test.*
import org.junit.Test

class FluxPromptBudgetPlannerTest {
  private val chars = FluxBodyTokenCounter { it.codePointCount(0, it.length) }

  @Test fun literalSurvivesPressureAndMetadataIsExact() {
    val literal = "Requested edit:\nKeep café 😀 unchanged"
    val sections = linkedMapOf("reference" to "reference", "requested_edit" to literal,
      "anatomy" to "anatomy", "irrelevant" to "x".repeat(200))
    val result = FluxPromptBudgetPlanner(chars, 70).plan(sections, setOf("requested_edit"))
    assertContains(result.text, literal)
    assertFalse("irrelevant" in result.includedSectionNames)
    assertTrue("irrelevant" in result.omittedGeneratedSectionNames)
    assertEquals(chars.count(result.text), result.bodyTokenCount)
    assertEquals(70, result.maximumBodyTokens)
    assertTrue(result.text.toByteArray().toString(Charsets.UTF_8).contains("😀"))
  }

  @Test fun truncationIsDeterministicAndNeverSplitsSurrogates() {
    val sections = linkedMapOf("requested_edit" to "😀".repeat(30))
    val first = FluxPromptBudgetPlanner(chars, 9).plan(sections, setOf("requested_edit"))
    val second = FluxPromptBudgetPlanner(chars, 9).plan(sections, setOf("requested_edit"))
    assertEquals(first, second)
    assertEquals(9, first.text.codePointCount(0, first.text.length))
    assertFalse(first.text.any { it.isSurrogate() } && first.text.length % 2 != 0)
  }

  @Test fun outfitAndAccessoriesAreIndependent() {
    assertEquals(setOf(FluxPromptIntent.OUTFIT), FluxPromptIntentDetector.detect("Change the dress"))
    assertEquals(setOf(FluxPromptIntent.ACCESSORIES), FluxPromptIntentDetector.detect("Remove the necklace"))
    val compiler = FluxEditPromptCompiler(chars, 10_000)
    val dress = compiler.compile(FluxFigureEditRequest(FluxFigureAction.PreserveCurrent, additionalInstruction = "Change the dress"))
    assertEquals(setOf("outfit"), dress.conflicts.map { it.lockName }.toSet())
    val necklace = compiler.compile(FluxFigureEditRequest(FluxFigureAction.PreserveCurrent, additionalInstruction = "Remove the necklace"))
    assertEquals(setOf("accessories"), necklace.conflicts.map { it.lockName }.toSet())
  }

  @Test fun explicitVisualContextAddsOnlySelectedRules() {
    val compiler = FluxEditPromptCompiler(chars, 10_000)
    val unspecified = compiler.compile(FluxFigureEditRequest(FluxFigureAction.PreserveCurrent)).positivePrompt
    assertFalse(unspecified.contains(FluxEditPromptCompiler.FULL_BODY))
    assertFalse(unspecified.contains(FluxEditPromptCompiler.HANDS))
    val explicit = compiler.compile(FluxFigureEditRequest(FluxFigureAction.PreserveCurrent,
      visualContext = FluxEditVisualContext(FluxFramingCategory.FULL_BODY, handsVisible = true))).positivePrompt
    assertContains(explicit, FluxEditPromptCompiler.FULL_BODY)
    assertContains(explicit, FluxEditPromptCompiler.HANDS)
  }

  @Test fun structuredAttributesRoundTripWithoutCollapsingFields() {
    val attributes = FluxFigureAttributes("build", "shoulders", "torso", "waist", "hips", "legs", "height")
    val copy = attributes.copy()
    assertEquals(attributes, copy)
    listOf(copy.overallBuild, copy.shoulders, copy.torso, copy.waist, copy.hips, copy.legs, copy.heightImpression)
      .forEach { assertTrue(it.isNotBlank()) }
  }
}
