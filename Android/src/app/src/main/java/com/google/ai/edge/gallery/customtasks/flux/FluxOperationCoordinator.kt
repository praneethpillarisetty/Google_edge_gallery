/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import java.io.Closeable
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/** Cancellation signal owned by one download, independent of model-file exclusivity. */
internal class FluxDownloadCancellation {
  private val requested = AtomicBoolean(false)
  private val active = ConcurrentHashMap.newKeySet<Closeable>()

  val isRequested: Boolean get() = requested.get()

  fun reset() {
    requested.set(false)
    active.clear()
  }

  fun attach(resource: Closeable) {
    active += resource
    if (requested.get() && active.remove(resource)) resource.closeQuietly()
  }

  fun detach(resource: Closeable) { active -= resource }

  /** Never waits for the file-operation mutex. */
  fun signal() {
    requested.set(true)
    active.toList().forEach { if (active.remove(it)) it.closeQuietly() }
  }
}

/** One cancellable exclusive gate for model reads, mutations, checks, and cleanup. */
internal class FluxFileOperationCoordinator {
  private val mutex = Mutex()

  suspend fun <T> exclusive(block: suspend () -> T): T = mutex.withLock { block() }
}

/** Defines the ordering contract between out-of-band cancellation and exclusive cleanup. */
internal class FluxDownloadOwnership(
  val cancellation: FluxDownloadCancellation = FluxDownloadCancellation(),
  private val files: FluxFileOperationCoordinator = FluxFileOperationCoordinator(),
) {
  fun pause() = cancellation.signal()

  suspend fun cancel(cleanup: suspend () -> Unit, recheck: suspend () -> Unit) {
    cancellation.signal()
    files.exclusive { cleanup(); recheck() }
  }

  suspend fun <T> exclusive(block: suspend () -> T): T = files.exclusive(block)
}

private fun Closeable.closeQuietly() = runCatching { close() }.getOrDefault(Unit)
