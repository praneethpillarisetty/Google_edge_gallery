package com.google.ai.edge.gallery.customtasks.flux.image

import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface

object FluxReferenceImageContracts {
  const val SIZE = 256
  const val INPUT_ELEMENTS = 3 * SIZE * SIZE
  const val OUTPUT_ELEMENTS = 32 * 32 * 32
  const val GRAPH = "kv_vae_enc.tflite"
  val INPUT_SHAPE = listOf(1, 3, SIZE, SIZE)
  val OUTPUT_SHAPE = listOf(1, 32, 32, 32)

  data class Crop(val left: Int, val top: Int, val side: Int)
  fun centerCrop(width: Int, height: Int): Crop {
    require(width > 0 && height > 0)
    val side = minOf(width, height)
    return Crop((width - side) / 2, (height - side) / 2, side)
  }

  fun orientationMatrix(orientation: Int): Matrix = Matrix().apply {
    when (orientation) {
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> setScale(-1f, 1f)
      ExifInterface.ORIENTATION_ROTATE_180 -> setRotate(180f)
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> setScale(1f, -1f)
      ExifInterface.ORIENTATION_TRANSPOSE -> { setRotate(90f); postScale(-1f, 1f) }
      ExifInterface.ORIENTATION_ROTATE_90 -> setRotate(90f)
      ExifInterface.ORIENTATION_TRANSVERSE -> { setRotate(-90f); postScale(-1f, 1f) }
      ExifInterface.ORIENTATION_ROTATE_270 -> setRotate(270f)
    }
  }

  fun compositeChannel(channel: Int, alpha: Int): Int = (channel * alpha + 127) / 255
  fun compositeBlack(argb: Int): Int {
    val alpha = argb ushr 24 and 0xff
    val red = compositeChannel(argb ushr 16 and 0xff, alpha)
    val green = compositeChannel(argb ushr 8 and 0xff, alpha)
    val blue = compositeChannel(argb and 0xff, alpha)
    return (0xff shl 24) or (red shl 16) or (green shl 8) or blue
  }
  fun normalize(channel: Int): Float = channel / 127.5f - 1f
}

data class FluxReferenceImageTensor private constructor(
  val values: FloatArray,
  val orientedWidth: Int,
  val orientedHeight: Int,
  val crop: FluxReferenceImageContracts.Crop,
) {
  val shape = FluxReferenceImageContracts.INPUT_SHAPE
  val layout = "NCHW RGB"
  val dataType = "FP32"
  companion object {
    fun checked(values: FloatArray, width: Int, height: Int, crop: FluxReferenceImageContracts.Crop): FluxReferenceImageTensor {
      require(values.size == FluxReferenceImageContracts.INPUT_ELEMENTS) { "Reference VAE input has the wrong element count." }
      require(values.all { it.isFinite() && it in -1f..1f }) { "Reference VAE input contains an invalid value." }
      return FluxReferenceImageTensor(values, width, height, crop)
    }
  }
}
