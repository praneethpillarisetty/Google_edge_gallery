package com.google.ai.edge.gallery.customtasks.flux.image

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlin.coroutines.coroutineContext

class FluxReferenceImagePreprocessor(private val resolver: ContentResolver) {
  suspend fun preprocess(uri: Uri, progress: (String) -> Unit = {}): FluxReferenceImageTensor = withContext(Dispatchers.IO) {
    try {
      coroutineContext.ensureActive()
      progress("READING_ORIENTATION")
      val orientation = resolver.openInputStream(uri)?.use { ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }
        ?: throw FluxReferenceImageException("The selected reference image cannot be opened.")
      val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
      resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        ?: throw FluxReferenceImageException("The selected reference image cannot be opened.")
      if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw FluxReferenceImageException("The selected reference image is empty or unsupported.")
      var sample = 1
      while (bounds.outWidth / (sample * 2) >= FluxReferenceImageContracts.SIZE && bounds.outHeight / (sample * 2) >= FluxReferenceImageContracts.SIZE) sample *= 2
      progress("DECODING_IMAGE")
      val options = BitmapFactory.Options().apply { inSampleSize = sample; inPreferredConfig = Bitmap.Config.ARGB_8888 }
      var decoded = resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, options) }
        ?: throw FluxReferenceImageException("The selected reference image cannot be decoded.")
      coroutineContext.ensureActive()
      progress("APPLYING_ORIENTATION")
      var oriented = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, FluxReferenceImageContracts.orientationMatrix(orientation), true)
      if (oriented !== decoded) decoded.recycle()
      decoded = oriented
      val width = oriented.width; val height = oriented.height
      progress("CENTER_CROPPING")
      val crop = FluxReferenceImageContracts.centerCrop(width, height)
      var square = Bitmap.createBitmap(oriented, crop.left, crop.top, crop.side, crop.side)
      if (square !== oriented) oriented.recycle()
      progress("RESIZING")
      var scaled = Bitmap.createScaledBitmap(square, FluxReferenceImageContracts.SIZE, FluxReferenceImageContracts.SIZE, true)
      if (scaled !== square) square.recycle()
      progress("BUILDING_INPUT")
      coroutineContext.ensureActive()
      val pixels = IntArray(FluxReferenceImageContracts.SIZE * FluxReferenceImageContracts.SIZE)
      scaled.getPixels(pixels, 0, FluxReferenceImageContracts.SIZE, 0, 0, FluxReferenceImageContracts.SIZE, FluxReferenceImageContracts.SIZE)
      scaled.recycle()
      val plane = pixels.size
      val values = FloatArray(plane * 3)
      pixels.forEachIndexed { i, raw ->
        val pixel = FluxReferenceImageContracts.compositeBlack(raw)
        values[i] = FluxReferenceImageContracts.normalize(android.graphics.Color.red(pixel))
        values[plane + i] = FluxReferenceImageContracts.normalize(android.graphics.Color.green(pixel))
        values[2 * plane + i] = FluxReferenceImageContracts.normalize(android.graphics.Color.blue(pixel))
      }
      FluxReferenceImageTensor.checked(values, width, height, crop)
    } catch (cancelled: CancellationException) { throw cancelled
    } catch (oom: OutOfMemoryError) {
      throw FluxReferenceImageException("Not enough memory to preprocess the selected reference image.", oom)
    } catch (e: FluxReferenceImageException) { throw e }
      catch (e: Exception) { throw FluxReferenceImageException("The selected reference image could not be processed.", e) }
  }
}
