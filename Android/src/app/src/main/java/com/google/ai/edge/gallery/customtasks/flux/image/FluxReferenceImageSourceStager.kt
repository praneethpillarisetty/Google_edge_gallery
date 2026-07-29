/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.image

import android.content.ContentResolver
import android.net.Uri
import java.io.File
import java.io.FileNotFoundException
import java.io.InputStream
import java.io.OutputStream
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

internal class FluxReferenceImageSourceStager internal constructor(
  private val cacheDirectory: File,
  private val openSource: () -> InputStream?,
  private val maximumBytes: Long = MAXIMUM_SOURCE_BYTES,
  private val openTarget: (File) -> OutputStream = { it.outputStream() },
) {
  constructor(cacheDirectory: File, resolver: ContentResolver, uri: Uri) :
    this(cacheDirectory, { resolver.openInputStream(uri) })

  suspend fun <T> withStagedSource(block: suspend (File) -> T): T = withContext(Dispatchers.IO) {
    var staged: File? = null
    try {
      staged = stage()
      block(staged)
    } finally {
      staged?.delete()
    }
  }

  private suspend fun stage(): File {
    val target = File.createTempFile("flux-reference-", ".source", cacheDirectory)
    var complete = false
    try {
      val source = try {
        openSource()
      } catch (denied: SecurityException) {
        throw FluxReferenceImageException(ACCESS_DENIED, denied)
      } catch (missing: FileNotFoundException) {
        throw FluxReferenceImageException(UNAVAILABLE, missing)
      } ?: throw FluxReferenceImageException(UNAVAILABLE)

      source.use { input ->
        openTarget(target).buffered().use { output ->
          val buffer = ByteArray(COPY_BUFFER_BYTES)
          var total = 0L
          while (true) {
            coroutineContext.ensureActive()
            val count = input.read(buffer)
            coroutineContext.ensureActive()
            if (count < 0) break
            if (count == 0) continue
            if (total > maximumBytes - count.toLong()) {
              throw FluxReferenceImageException(TOO_LARGE)
            }
            output.write(buffer, 0, count)
            total += count.toLong()
          }
          if (total == 0L) throw FluxReferenceImageException(EMPTY)
        }
      }
      complete = true
      return target
    } catch (denied: SecurityException) {
      throw FluxReferenceImageException(ACCESS_DENIED, denied)
    } catch (missing: FileNotFoundException) {
      throw FluxReferenceImageException(UNAVAILABLE, missing)
    } catch (cancelled: CancellationException) {
      throw cancelled
    } finally {
      if (!complete) target.delete()
    }
  }

  companion object {
    const val MAXIMUM_SOURCE_BYTES = 128L * 1024L * 1024L
    internal const val COPY_BUFFER_BYTES = 32 * 1024
    const val ACCESS_DENIED = "Access to the selected image was denied. Select the image again."
    const val UNAVAILABLE = "The selected image is no longer available. Select it again."
    const val EMPTY = "The selected image is empty."
    const val TOO_LARGE = "The selected image is too large to import safely."
  }
}
