/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import java.io.ByteArrayInputStream
import java.io.InputStream
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.Job
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FluxStreamingHasherTest {
  @Test fun streamingHashCancellationClosesInput() = runTest {
    val cache = MemoryCache(); val started = CompletableDeferred<Unit>(); val closed = CompletableDeferred<Unit>()
    val blocking = object : InputStream() {
      override fun read(): Int = error("unused")
      override fun read(bytes: ByteArray, offset: Int, length: Int): Int { started.complete(Unit); return 1 }
      override fun close() { closed.complete(Unit) }
    }
    lateinit var job: Job
    job = launch { FluxStreamingHasher(cache).hash(id(), { blocking }) { job.cancel() } }
    started.await(); job.cancelAndJoin(); assertTrue(closed.isCompleted); assertTrue(cache.values.isEmpty())
  }

  @Test fun identicalIdentityUsesHashCache() = runTest {
    val cache = MemoryCache(); val hasher = FluxStreamingHasher(cache); val identity = id()
    val expected = hasher.hash(identity, { ByteArrayInputStream(byteArrayOf(1)) })
    var opened = false
    assertEquals(expected, hasher.hash(identity, { opened = true; ByteArrayInputStream(byteArrayOf(2)) }))
    assertFalse(opened)
  }

  @Test fun sizeChangeInvalidatesHashCache() = runTest { assertInvalidated(id().copy(size = 2)) }

  @Test fun modificationTimeChangeInvalidatesHashCache() = runTest { assertInvalidated(id().copy(modifiedMillis = 2)) }

  private suspend fun assertInvalidated(changed: FluxHashIdentity) {
    val cache = MemoryCache(); val hasher = FluxStreamingHasher(cache); var opens = 0
    hasher.hash(id(), { opens++; ByteArrayInputStream(byteArrayOf(1)) })
    hasher.hash(changed, { opens++; ByteArrayInputStream(byteArrayOf(1)) })
    assertEquals(2, opens)
  }

  private fun id() = FluxHashIdentity("enc0.tflite", 1, 1)
  private class MemoryCache : FluxHashCache {
    val values = mutableMapOf<FluxHashIdentity, String>()
    override fun get(identity: FluxHashIdentity) = values[identity]
    override fun put(identity: FluxHashIdentity, hash: String) { values[identity] = hash }
  }
}
