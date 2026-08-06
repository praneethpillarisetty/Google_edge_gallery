package com.google.ai.edge.gallery.customtasks.flux

import kotlin.test.*
import kotlinx.coroutines.async
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.Test

class FluxPromptCompilationRunnerTest {
  @Test fun workUsesInjectedDispatcherAndReturnsToCaller() = runTest {
    val dispatcher = StandardTestDispatcher(testScheduler)
    var worked = false
    val deferred = async { FluxPromptCompilationRunner(dispatcher).run { worked = true; "compiled" } }
    assertFalse(worked)
    testScheduler.advanceUntilIdle()
    assertTrue(worked)
    assertEquals("compiled", deferred.await())
  }

  @Test fun cancellationPreventsCompletion() = runTest {
    val dispatcher = StandardTestDispatcher(testScheduler)
    var completed = false
    val job = launch { FluxPromptCompilationRunner(dispatcher).run { completed = true } }
    job.cancelAndJoin()
    testScheduler.advanceUntilIdle()
    assertFalse(completed)
  }

  @Test fun repeatedTapsAreSingleFlight() {
    val gate = FluxCompilationGate()
    assertTrue(gate.tryStart())
    assertFalse(gate.tryStart())
    gate.finish()
    assertTrue(gate.tryStart())
  }

  @Test fun terminalStateReducerPreservesPreviousResultOnPreparationFailure() {
    val previous = FluxProductionGenerationState(actualSeed = 77, visibleInstruction = "previous")
    val failed = fluxTerminalGenerationState(previous, com.google.ai.edge.gallery.customtasks.flux.generation.FluxGenerationStage.ERROR, 5, "safe")
    assertEquals(77, failed.actualSeed)
    assertEquals("previous", failed.visibleInstruction)
    assertEquals("safe", failed.sanitizedError)
  }
}
