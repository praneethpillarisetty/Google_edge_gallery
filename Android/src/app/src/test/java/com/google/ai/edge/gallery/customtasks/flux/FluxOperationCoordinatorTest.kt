/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import java.io.Closeable
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.yield
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FluxOperationCoordinatorTest {
  @Test fun activeDownloadReceivesCancelWithoutWaitingForFileMutex() = runTest {
    val ownership = FluxDownloadOwnership()
    val writerEntered = CompletableDeferred<Unit>()
    val releaseWriter = CompletableDeferred<Unit>()
    val closed = CompletableDeferred<Unit>()
    val writer = launch { ownership.exclusive { writerEntered.complete(Unit); releaseWriter.await() } }
    writerEntered.await()
    ownership.cancellation.attach(Closeable { closed.complete(Unit) })
    val cancel = launch { ownership.cancel({}, {}) }
    closed.await()
    assertFalse(cancel.isCompleted)
    releaseWriter.complete(Unit)
    cancel.join(); writer.join()
  }

  @Test fun cancelWaitsForWriterCleanupBeforeDeletingAndRechecking() = runTest {
    val ownership = FluxDownloadOwnership()
    val entered = CompletableDeferred<Unit>(); val release = CompletableDeferred<Unit>()
    val order = mutableListOf<String>()
    val writer = launch { ownership.exclusive { entered.complete(Unit); release.await(); order += "writer-closed" } }
    entered.await()
    val cancel = launch { ownership.cancel({ order += "partial-deleted" }, { order += "rechecked" }) }
    yield(); assertTrue(order.isEmpty())
    release.complete(Unit); writer.join(); cancel.join()
    assertEquals(listOf("writer-closed", "partial-deleted", "rechecked"), order)
  }

  @Test fun repeatedCancelIsSafe() = runTest {
    val ownership = FluxDownloadOwnership(); var cleanups = 0; var checks = 0
    repeat(2) { ownership.cancel({ cleanups++ }, { checks++ }) }
    assertEquals(2, cleanups); assertEquals(2, checks)
  }

  @Test fun pauseIsImmediateWhileWriterOwnsMutex() = runTest {
    val ownership = FluxDownloadOwnership(); val entered = CompletableDeferred<Unit>(); val release = CompletableDeferred<Unit>()
    val writer = launch { ownership.exclusive { entered.complete(Unit); release.await() } }
    entered.await(); ownership.pause(); assertTrue(ownership.cancellation.isRequested)
    release.complete(Unit); writer.join()
  }

  @Test fun verificationWaitingForMutexCanBeCancelled() = runTest {
    val gate = FluxFileOperationCoordinator(); val entered = CompletableDeferred<Unit>(); val release = CompletableDeferred<Unit>()
    val mutation = launch { gate.exclusive { entered.complete(Unit); release.await() } }; entered.await()
    var read = false
    val verification = launch { gate.exclusive { read = true } }
    verification.cancelAndJoin(); assertFalse(read)
    release.complete(Unit); mutation.join()
  }

  @Test fun verificationCancellationReleasesExclusiveAccess() = runTest {
    val gate = FluxFileOperationCoordinator(); val entered = CompletableDeferred<Unit>(); val never = CompletableDeferred<Unit>()
    val verification = launch { gate.exclusive { entered.complete(Unit); never.await() } }; entered.await()
    verification.cancelAndJoin()
    var mutated = false; gate.exclusive { mutated = true }; assertTrue(mutated)
  }

  @Test fun downloadCannotMutateDuringVerification() = runTest { assertExclusiveOrder(readFirst = true) }

  @Test fun verificationCannotReadDuringActiveMutation() = runTest { assertExclusiveOrder(readFirst = false) }

  private suspend fun kotlinx.coroutines.test.TestScope.assertExclusiveOrder(readFirst: Boolean) {
    val gate = FluxFileOperationCoordinator(); val entered = CompletableDeferred<Unit>(); val release = CompletableDeferred<Unit>()
    var secondEntered = false
    val first = launch { gate.exclusive { entered.complete(Unit); release.await() } }; entered.await()
    val second = launch { gate.exclusive { secondEntered = true } }
    yield(); assertFalse("readFirst=$readFirst", secondEntered)
    release.complete(Unit); first.join(); second.join(); assertTrue(secondEntered)
  }
}
