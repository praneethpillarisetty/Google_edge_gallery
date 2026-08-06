package com.google.ai.edge.gallery.customtasks.flux

import com.google.ai.edge.gallery.customtasks.flux.generation.*
import kotlin.test.*
import org.junit.Test

class FluxGenerationStatePreservationTest {
  @Test fun cancellationAndFailurePreserveSuccessfulMetadata() {
    val previous = FluxProductionGenerationState(actualSeed = 42, seedSelection = FluxSeedSelection.Fixed(42),
      fixedSeedText = "42", editingMode = FluxEditMode.FIGURE, visibleInstruction = "edit",
      figurePresetName = "Balanced")
    listOf(FluxGenerationStage.CANCELLED, FluxGenerationStage.ERROR).forEach { stage ->
      val terminal = fluxTerminalGenerationState(previous, stage, 9, "safe")
      assertEquals(previous.actualSeed, terminal.actualSeed); assertEquals(previous.seedSelection, terminal.seedSelection)
      assertEquals(previous.fixedSeedText, terminal.fixedSeedText); assertEquals(previous.editingMode, terminal.editingMode)
      assertEquals(previous.iterativeReference, terminal.iterativeReference); assertEquals(previous.savedOutputUri, terminal.savedOutputUri)
    }
  }
}
