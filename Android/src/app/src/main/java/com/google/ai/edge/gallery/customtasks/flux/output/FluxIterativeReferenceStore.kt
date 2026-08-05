/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.output

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.core.content.FileProvider
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import javax.inject.Inject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

/** Owns only generated iterative inputs. Provider-selected source URIs are never touched. */
class FluxIterativeReferenceStore @Inject constructor(@ApplicationContext private val context: Context) {
  private val directory get() = File(context.filesDir, "flux_iterative").apply { mkdirs() }
  suspend fun stage(bitmap: Bitmap, previousOwned: Uri?): Uri = withContext(Dispatchers.IO) {
    val temp = File.createTempFile("iterative_", ".tmp", directory)
    val final = File(directory, "reference_${System.nanoTime()}.png")
    try {
      temp.outputStream().buffered().use { output ->
        check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)) { "PNG encoding failed." }
        output.flush()
      }
      ensureActive()
      check(temp.renameTo(final)) { "Could not finalize iterative reference." }
      val uri = FileProvider.getUriForFile(context, "${context.packageName}.provider", final)
      previousOwned?.let(::deleteOwned)
      uri
    } catch (failure: Throwable) {
      temp.delete(); final.delete(); throw failure
    }
  }
  fun deleteOwned(uri: Uri) {
    if (uri.authority != "${context.packageName}.provider") return
    val name = uri.lastPathSegment?.substringAfterLast('/') ?: return
    File(directory, name).takeIf { it.parentFile == directory && it.name.endsWith(".png") }?.delete()
  }
}
