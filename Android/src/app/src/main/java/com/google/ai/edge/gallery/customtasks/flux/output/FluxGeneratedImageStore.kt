/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.output

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import dagger.hilt.android.qualifiers.ApplicationContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.inject.Inject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class FluxOutputException(val safeMessage: String, cause: Throwable? = null) : Exception(safeMessage, cause)

class FluxGeneratedImageStore @Inject constructor(@ApplicationContext private val context: Context) {
  suspend fun savePng(bitmap: Bitmap, resized: Boolean = false): Uri = withContext(Dispatchers.IO) {
    require(bitmap.config == Bitmap.Config.ARGB_8888)
    val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    val name = if (resized) "FLUX_Edit_1024_${stamp}.png" else "FLUX_Edit_${stamp}.png"
    val values = ContentValues().apply {
      put(MediaStore.Images.Media.DISPLAY_NAME, name)
      put(MediaStore.Images.Media.MIME_TYPE, "image/png")
      put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/GoogleEdgeGallery")
      if (Build.VERSION.SDK_INT >= 29) put(MediaStore.Images.Media.IS_PENDING, 1)
    }
    val resolver = context.contentResolver
    var uri: Uri? = null
    try {
      uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        ?: throw FluxOutputException("Could not create the image in MediaStore.")
      resolver.openOutputStream(uri, "w")?.use { output ->
        if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)) throw FluxOutputException("PNG encoding failed.")
      } ?: throw FluxOutputException("Could not write the image.")
      if (Build.VERSION.SDK_INT >= 29) resolver.update(uri, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
      uri
    } catch (failure: Throwable) {
      uri?.let { resolver.delete(it, null, null) }
      throw FluxOutputException("The PNG could not be saved.", failure)
    }
  }

  suspend fun saveResized1024(source: Bitmap): Uri = withContext(Dispatchers.Default) {
    var resized: Bitmap? = null
    try {
      resized = Bitmap.createScaledBitmap(source, 1024, 1024, true)
      savePng(requireNotNull(resized), resized = true)
    } catch (failure: OutOfMemoryError) {
      throw FluxOutputException("The 1024×1024 resized export needs more memory.", failure)
    } finally {
      if (resized !== source) resized?.recycle()
    }
  }

  fun shareIntent(uri: Uri): Intent = Intent(Intent.ACTION_SEND).apply {
    type = "image/png"
    putExtra(Intent.EXTRA_STREAM, uri)
    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    clipData = android.content.ClipData.newUri(context.contentResolver, "FLUX image", uri)
  }

  fun canShare(intent: Intent): Boolean = intent.resolveActivity(context.packageManager) != null
}
