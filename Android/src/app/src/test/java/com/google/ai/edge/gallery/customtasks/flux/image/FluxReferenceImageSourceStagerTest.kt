/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.image

import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileNotFoundException
import java.io.InputStream
import java.io.OutputStream
import java.nio.file.Files
import kotlin.test.assertFailsWith
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FluxReferenceImageSourceStagerTest {
  @Test fun sourceIsOpenedExactlyOnceAndStreamedInBoundedReads() = runTest {
    var opens = 0
    var largestRead = 0
    val source = ByteArray(FluxReferenceImageSourceStager.COPY_BUFFER_BYTES * 3 + 7) { 1 }
    val input = object : ByteArrayInputStream(source) {
      override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
        largestRead = maxOf(largestRead, length)
        return super.read(buffer, offset, length)
      }
    }
    stager({ opens++; input }).withStagedSource { assertEquals(source.size.toLong(), it.length()) }
    assertEquals(1, opens)
    assertTrue(largestRead <= FluxReferenceImageSourceStager.COPY_BUFFER_BYTES)
  }

  @Test fun emptySourceIsRejectedAndTemporaryFileRemoved() = runTest {
    val directory = temporaryDirectory()
    val failure = assertFailsWith<FluxReferenceImageException> {
      stager({ ByteArrayInputStream(byteArrayOf()) }, directory).withStagedSource {}
    }
    assertEquals(FluxReferenceImageSourceStager.EMPTY, failure.message)
    assertTrue(directory.listFiles().orEmpty().isEmpty())
  }

  @Test fun sourceOverLimitIsRejectedAndTemporaryFileRemoved() = runTest {
    val directory = temporaryDirectory()
    val failure = assertFailsWith<FluxReferenceImageException> {
      stager({ ByteArrayInputStream(ByteArray(9)) }, directory, 8).withStagedSource {}
    }
    assertEquals(FluxReferenceImageSourceStager.TOO_LARGE, failure.message)
    assertTrue(directory.listFiles().orEmpty().isEmpty())
  }

  @Test fun cancellationClosesBothStreamsAndRemovesTemporaryFile() = runTest {
    val directory = temporaryDirectory()
    var inputClosed = false
    var outputClosed = false
    val input = object : InputStream() {
      override fun read(): Int = throw CancellationException("stop")
      override fun read(buffer: ByteArray, offset: Int, length: Int): Int = read()
      override fun close() { inputClosed = true }
    }
    val output = object : OutputStream() {
      override fun write(value: Int) = Unit
      override fun write(buffer: ByteArray, offset: Int, length: Int) = Unit
      override fun close() { outputClosed = true }
    }
    assertFailsWith<CancellationException> {
      stager({ input }, directory, openTarget = { output }).withStagedSource {}
    }
    assertTrue(inputClosed)
    assertTrue(outputClosed)
    assertTrue(directory.listFiles().orEmpty().isEmpty())
    // A successfully closed output leaves no open handle preventing directory removal.
    assertTrue(directory.delete())
  }

  @Test fun temporaryFileIsRemovedAfterSuccessPreprocessingFailureAndVaeFailure() = runTest {
    listOf<Throwable?>(null, IllegalArgumentException("preprocess"), IllegalStateException("vae"))
      .forEach { failure ->
        val directory = temporaryDirectory()
        if (failure == null) {
          stager({ ByteArrayInputStream(byteArrayOf(1)) }, directory).withStagedSource { assertTrue(it.exists()) }
        } else {
          assertFailsWith<Throwable> {
            stager({ ByteArrayInputStream(byteArrayOf(1)) }, directory).withStagedSource { throw failure }
          }
        }
        assertTrue(directory.listFiles().orEmpty().isEmpty())
      }
  }

  @Test fun temporaryFileIsRemovedWhenVerificationBlockIsCancelled() = runTest {
    val directory = temporaryDirectory()
    assertFailsWith<CancellationException> {
      stager({ ByteArrayInputStream(byteArrayOf(1)) }, directory).withStagedSource {
        throw CancellationException("stop")
      }
    }
    assertTrue(directory.listFiles().orEmpty().isEmpty())
  }

  @Test fun providerFailuresAreSanitizedWithoutSourceOrTemporaryDetails() = runTest {
    val sensitive = "content://provider/private-name.jpg"
    val directory = temporaryDirectory()
    val denied = assertFailsWith<FluxReferenceImageException> {
      stager({ throw SecurityException(sensitive) }, directory).withStagedSource {}
    }
    val missing = assertFailsWith<FluxReferenceImageException> {
      stager({ throw FileNotFoundException(sensitive) }, directory).withStagedSource {}
    }
    val nullStream = assertFailsWith<FluxReferenceImageException> {
      stager({ null }, directory).withStagedSource {}
    }
    assertEquals(FluxReferenceImageSourceStager.ACCESS_DENIED, denied.message)
    assertEquals(FluxReferenceImageSourceStager.UNAVAILABLE, missing.message)
    assertEquals(FluxReferenceImageSourceStager.UNAVAILABLE, nullStream.message)
    listOf(denied, missing, nullStream).forEach {
      assertFalse(it.message.orEmpty().contains(sensitive))
      assertFalse(it.message.orEmpty().contains(directory.path))
    }
  }

  private fun stager(
    source: () -> InputStream?,
    directory: File = temporaryDirectory(),
    maximumBytes: Long = FluxReferenceImageSourceStager.MAXIMUM_SOURCE_BYTES,
    openTarget: (File) -> OutputStream = { it.outputStream() },
  ) = FluxReferenceImageSourceStager(directory, source, maximumBytes, openTarget)

  private fun temporaryDirectory(): File = Files.createTempDirectory("flux-stager-test").toFile()
}
