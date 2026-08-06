package com.google.ai.edge.gallery.customtasks.flux.output

import java.nio.file.Files
import kotlin.test.*
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import org.junit.Test

class FluxAtomicFileStagerTest {
  @Test fun successfulStagePublishesOnlyFinalPng() = runTest {
    val directory = Files.createTempDirectory("flux-stage").toFile()
    val final = FluxAtomicFileStager(directory).stage { it.writeBytes(byteArrayOf(1, 2, 3)) }
    assertTrue(final.name.endsWith(".png")); assertContentEquals(byteArrayOf(1, 2, 3), final.readBytes())
    assertTrue(directory.listFiles().orEmpty().none { it.name.endsWith(".tmp") })
    directory.deleteRecursively()
  }

  @Test fun encodingAndFinalizationFailuresDeletePartials() = runTest {
    val directory = Files.createTempDirectory("flux-stage").toFile()
    assertFailsWith<IllegalStateException> { FluxAtomicFileStager(directory).stage { error("encode") } }
    assertTrue(directory.listFiles().orEmpty().isEmpty())
    assertFailsWith<IllegalStateException> { FluxAtomicFileStager(directory) { _, _ -> false }.stage { it.writeText("png") } }
    assertTrue(directory.listFiles().orEmpty().isEmpty())
    directory.deleteRecursively()
  }

  @Test fun cancellationBeforeFinalizationLeavesNoPartial() = runTest {
    val directory = Files.createTempDirectory("flux-stage").toFile()
    val job = launch { FluxAtomicFileStager(directory).stage { coroutineContext.cancel(); it.writeText("partial") } }
    job.cancelAndJoin()
    assertTrue(directory.listFiles().orEmpty().isEmpty())
    directory.deleteRecursively()
  }
}
