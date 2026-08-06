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

internal class FluxAtomicFileStager(
  private val directory: File,
  private val finalize: (File, File) -> Boolean = File::renameTo,
) {
  suspend fun stage(write: (File) -> Unit): File {
    directory.mkdirs()
    val temp = File.createTempFile("iterative_", ".tmp", directory)
    val final = File(directory, "reference_${System.nanoTime()}.png")
    try {
      write(temp)
      kotlinx.coroutines.currentCoroutineContext().ensureActive()
      check(finalize(temp, final)) { "Could not finalize iterative reference." }
      return final
    } catch (failure: Throwable) {
      temp.delete(); final.delete(); throw failure
    }
  }
}

/** Owns only generated iterative inputs. Provider-selected source URIs are never touched. */
class FluxIterativeReferenceStore @Inject constructor(@ApplicationContext private val context: Context) {
  private val directory get() = File(context.filesDir, "flux_iterative").apply { mkdirs() }
  suspend fun stage(bitmap: Bitmap, previousOwned: Uri?): Uri = withContext(Dispatchers.IO) {
    val final = FluxAtomicFileStager(directory).stage { temp ->
      temp.outputStream().buffered().use { output ->
        check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)) { "PNG encoding failed." }
        output.flush()
      }
    }
    try {
      val uri = FileProvider.getUriForFile(context, "${context.packageName}.provider", final)
      previousOwned?.let(::deleteOwned)
      uri
    } catch (failure: Throwable) {
      final.delete(); throw failure
    }
  }
  fun deleteOwned(uri: Uri) {
    if (uri.authority != "${context.packageName}.provider") return
    val segments = uri.pathSegments
    if (segments.size != 2 || segments[0] != "flux_iterative") return
    val name = segments[1]
    if (!name.startsWith("reference_") || !name.endsWith(".png") || name.contains('/') || name.contains('\\')) return
    val root = directory.canonicalFile
    val target = File(root, name).canonicalFile
    if (target.parentFile == root && target.path.startsWith(root.path + File.separator)) target.delete()
  }
}
