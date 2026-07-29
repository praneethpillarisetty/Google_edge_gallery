/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.image

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

class FluxReferenceImagePreprocessor(private val resolver: ContentResolver) {
  suspend fun preprocess(uri: Uri, progress: (String) -> Unit = {}): FluxReferenceImageTensor =
    withContext(Dispatchers.IO) {
      var decoded: Bitmap? = null
      var oriented: Bitmap? = null
      var cropped: Bitmap? = null
      var opaque: Bitmap? = null
      var scaled: Bitmap? = null
      try {
        coroutineContext.ensureActive()
        progress("READING_ORIENTATION")
        val orientation = resolver.openInputStream(uri)?.use {
          ExifInterface(it).getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL,
          )
        } ?: throw FluxReferenceImageException("The selected reference image cannot be opened.")
        coroutineContext.ensureActive()

        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
          ?: throw FluxReferenceImageException("The selected reference image cannot be opened.")
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
          throw FluxReferenceImageException("The selected reference image is empty or unsupported.")
        }
        coroutineContext.ensureActive()
        val plan = FluxReferenceImageContracts.decodePlan(bounds.outWidth, bounds.outHeight)

        progress("DECODING_IMAGE")
        decoded = resolver.openInputStream(uri)?.use {
          BitmapFactory.decodeStream(
            it,
            null,
            BitmapFactory.Options().apply {
              inSampleSize = plan.inSampleSize
              inPreferredConfig = Bitmap.Config.ARGB_8888
            },
          )
        } ?: throw FluxReferenceImageException("The selected reference image cannot be decoded.")
        coroutineContext.ensureActive()

        progress("APPLYING_ORIENTATION")
        val transformed = Bitmap.createBitmap(
          decoded!!, 0, 0, decoded!!.width, decoded!!.height,
          FluxReferenceImageContracts.orientationMatrix(orientation), true,
        )
        oriented = transformed
        if (transformed !== decoded) {
          decoded!!.recycle()
          decoded = null
        } else {
          decoded = null // ownership transferred to oriented
        }
        coroutineContext.ensureActive()
        val orientedWidth = oriented!!.width
        val orientedHeight = oriented!!.height

        progress("CENTER_CROPPING")
        val crop = FluxReferenceImageContracts.centerCrop(orientedWidth, orientedHeight)
        val square = Bitmap.createBitmap(oriented!!, crop.left, crop.top, crop.side, crop.side)
        cropped = square
        if (square !== oriented) {
          oriented!!.recycle()
          oriented = null
        } else {
          oriented = null // ownership transferred to cropped
        }
        coroutineContext.ensureActive()

        // Alpha is removed before interpolation so hidden transparent RGB cannot affect resize.
        opaque = Bitmap.createBitmap(crop.side, crop.side, Bitmap.Config.ARGB_8888)
        val row = IntArray(crop.side)
        for (y in 0 until crop.side) {
          if (y and 31 == 0) coroutineContext.ensureActive()
          cropped!!.getPixels(row, 0, crop.side, 0, y, crop.side, 1)
          for (x in row.indices) row[x] = FluxReferenceImageContracts.compositeBlack(row[x])
          opaque!!.setPixels(row, 0, crop.side, 0, y, crop.side, 1)
        }
        cropped!!.recycle()
        cropped = null

        progress("RESIZING")
        val resized = Bitmap.createScaledBitmap(
          opaque!!,
          FluxReferenceImageContracts.SIZE,
          FluxReferenceImageContracts.SIZE,
          true,
        )
        scaled = resized
        if (resized !== opaque) {
          opaque!!.recycle()
          opaque = null
        } else {
          opaque = null // ownership transferred to scaled
        }
        coroutineContext.ensureActive()

        progress("BUILDING_INPUT")
        val pixels = IntArray(FluxReferenceImageContracts.SIZE * FluxReferenceImageContracts.SIZE)
        scaled!!.getPixels(
          pixels, 0, FluxReferenceImageContracts.SIZE, 0, 0,
          FluxReferenceImageContracts.SIZE, FluxReferenceImageContracts.SIZE,
        )
        scaled!!.recycle()
        scaled = null
        val values = FluxReferenceImageContracts.pixelsToNchw(pixels) {
          coroutineContext.ensureActive()
        }
        coroutineContext.ensureActive()
        FluxReferenceImageTensor.checked(values, orientedWidth, orientedHeight, crop, plan)
      } catch (cancelled: CancellationException) {
        throw cancelled
      } catch (oom: OutOfMemoryError) {
        throw FluxReferenceImageException(
          "Not enough memory to preprocess the selected reference image.",
          oom,
        )
      } catch (failure: FluxReferenceImageException) {
        throw failure
      } catch (failure: Exception) {
        throw FluxReferenceImageException(
          "The selected reference image could not be processed.",
          failure,
        )
      } finally {
        listOfNotNull(scaled, opaque, cropped, oriented, decoded)
          .distinctBy { System.identityHashCode(it) }
          .forEach { if (!it.isRecycled) it.recycle() }
      }
    }
}
