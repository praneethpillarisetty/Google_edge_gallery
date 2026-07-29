/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import java.io.InputStream
import java.security.MessageDigest
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.ensureActive

internal data class FluxHashIdentity(val filename: String, val size: Long, val modifiedMillis: Long) {
  val key: String get() = "$filename:$size:$modifiedMillis"
}

internal fun interface FluxHashCache {
  fun get(identity: FluxHashIdentity): String?

  fun put(identity: FluxHashIdentity, hash: String) = Unit
}

/** JVM-testable streaming hashing; structured cancellation always closes the input. */
internal class FluxStreamingHasher(private val cache: FluxHashCache) {
  suspend fun hash(
    identity: FluxHashIdentity,
    open: () -> InputStream,
    progress: (Long) -> Unit = {},
  ): String {
    cache.get(identity)?.let { return it }
    val digest = MessageDigest.getInstance("SHA-256")
    var read = 0L
    open().buffered().use { input ->
      val buffer = ByteArray(1024 * 1024)
      while (true) {
        coroutineContext.ensureActive()
        val amount = input.read(buffer)
        if (amount < 0) break
        digest.update(buffer, 0, amount)
        read += amount
        progress(read)
      }
    }
    coroutineContext.ensureActive()
    return digest.digest().joinToString("") { "%02x".format(it) }
      .also { cache.put(identity, it) }
  }
}
