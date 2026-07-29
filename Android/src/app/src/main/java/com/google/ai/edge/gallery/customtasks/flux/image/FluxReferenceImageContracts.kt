/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.image

import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface

object FluxReferenceImageContracts {
  const val SIZE = 256
  const val INPUT_ELEMENTS = 3 * SIZE * SIZE
  const val OUTPUT_ELEMENTS = 32 * 32 * 32
  const val GRAPH = "kv_vae_enc.tflite"
  const val MAX_DECODE_BYTES = 64L * 1024 * 1024
  val INPUT_SHAPE = listOf(1, 3, SIZE, SIZE)
  val OUTPUT_SHAPE = listOf(1, 32, 32, 32)

  data class Crop(val left: Int, val top: Int, val side: Int)
  data class DecodePlan(
    val inSampleSize: Int,
    val estimatedWidth: Int,
    val estimatedHeight: Int,
    val estimatedArgbBytes: Long,
  )
  data class Point(val x: Int, val y: Int)

  fun centerCrop(width: Int, height: Int): Crop {
    require(width > 0 && height > 0)
    val side = minOf(width, height)
    return Crop((width - side) / 2, (height - side) / 2, side)
  }

  fun decodePlan(width: Int, height: Int): DecodePlan {
    require(width > 0 && height > 0) { "Invalid reference image dimensions." }
    val widthLong = width.toLong()
    val heightLong = height.toLong()
    var sampleLong = 1L
    while (
      widthLong / Math.multiplyExact(sampleLong, 2L) >= SIZE &&
        heightLong / Math.multiplyExact(sampleLong, 2L) >= SIZE
    ) {
      sampleLong = Math.multiplyExact(sampleLong, 2L)
    }
    val sampledWidth: Long
    val sampledHeight: Long
    val bytes: Long
    try {
      sampledWidth = Math.addExact(widthLong, sampleLong - 1L) / sampleLong
      sampledHeight = Math.addExact(heightLong, sampleLong - 1L) / sampleLong
      require(sampledWidth > 0L && sampledHeight > 0L) {
        "Invalid sampled reference image dimensions."
      }
      val pixels = Math.multiplyExact(sampledWidth, sampledHeight)
      bytes = Math.multiplyExact(pixels, 4L)
    } catch (overflow: ArithmeticException) {
      throw IllegalArgumentException(
        "The selected reference image is too large to preprocess safely.",
        overflow,
      )
    }
    require(bytes in 1L..MAX_DECODE_BYTES) {
      "The selected reference image is too large to preprocess safely."
    }
    require(sampledWidth <= Int.MAX_VALUE.toLong() && sampledHeight <= Int.MAX_VALUE.toLong()) {
      "The selected reference image is too large to preprocess safely."
    }
    require(sampleLong <= Int.MAX_VALUE.toLong()) {
      "The selected reference image is too large to preprocess safely."
    }
    return DecodePlan(sampleLong.toInt(), sampledWidth.toInt(), sampledHeight.toInt(), bytes)
  }

  fun orientedSize(orientation: Int, width: Int, height: Int): Pair<Int, Int> =
    if (orientation in setOf(
        ExifInterface.ORIENTATION_TRANSPOSE,
        ExifInterface.ORIENTATION_ROTATE_90,
        ExifInterface.ORIENTATION_TRANSVERSE,
        ExifInterface.ORIENTATION_ROTATE_270,
      )) height to width else width to height

  /** Platform-neutral source-to-oriented coordinate mapping, used to verify all EXIF policies. */
  fun orientPoint(orientation: Int, width: Int, height: Int, x: Int, y: Int): Point =
    when (orientation) {
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> Point(width - 1 - x, y)
      ExifInterface.ORIENTATION_ROTATE_180 -> Point(width - 1 - x, height - 1 - y)
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> Point(x, height - 1 - y)
      ExifInterface.ORIENTATION_TRANSPOSE -> Point(y, x)
      ExifInterface.ORIENTATION_ROTATE_90 -> Point(height - 1 - y, x)
      ExifInterface.ORIENTATION_TRANSVERSE -> Point(height - 1 - y, width - 1 - x)
      ExifInterface.ORIENTATION_ROTATE_270 -> Point(y, width - 1 - x)
      else -> Point(x, y)
    }

  fun orientationMatrix(orientation: Int): Matrix = Matrix().apply {
    val (a, b, c, d) = when (orientation) {
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> listOf(-1f, 0f, 0f, 1f)
      ExifInterface.ORIENTATION_ROTATE_180 -> listOf(-1f, 0f, 0f, -1f)
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> listOf(1f, 0f, 0f, -1f)
      ExifInterface.ORIENTATION_TRANSPOSE -> listOf(0f, 1f, 1f, 0f)
      ExifInterface.ORIENTATION_ROTATE_90 -> listOf(0f, -1f, 1f, 0f)
      ExifInterface.ORIENTATION_TRANSVERSE -> listOf(0f, -1f, -1f, 0f)
      ExifInterface.ORIENTATION_ROTATE_270 -> listOf(0f, 1f, -1f, 0f)
      else -> listOf(1f, 0f, 0f, 1f)
    }
    setValues(floatArrayOf(a, b, 0f, c, d, 0f, 0f, 0f, 1f))
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

  internal fun pixelsToNchw(pixels: IntArray, cancellationCheck: () -> Unit = {}): FloatArray {
    require(pixels.size == SIZE * SIZE)
    val plane = pixels.size
    val values = FloatArray(INPUT_ELEMENTS)
    pixels.forEachIndexed { index, pixel ->
      if (index and 4095 == 0) cancellationCheck()
      values[index] = normalize(pixel ushr 16 and 0xff)
      values[plane + index] = normalize(pixel ushr 8 and 0xff)
      values[2 * plane + index] = normalize(pixel and 0xff)
    }
    cancellationCheck()
    return values
  }
}

data class FluxReferenceImageTensor private constructor(
  val values: FloatArray,
  val orientedWidth: Int,
  val orientedHeight: Int,
  val crop: FluxReferenceImageContracts.Crop,
  val decodePlan: FluxReferenceImageContracts.DecodePlan,
) {
  val shape = FluxReferenceImageContracts.INPUT_SHAPE
  val layout = "NCHW RGB"
  val dataType = "FP32"
  companion object {
    fun checked(
      values: FloatArray,
      width: Int,
      height: Int,
      crop: FluxReferenceImageContracts.Crop,
      decodePlan: FluxReferenceImageContracts.DecodePlan =
        FluxReferenceImageContracts.DecodePlan(1, width, height, width.toLong() * height * 4L),
    ): FluxReferenceImageTensor {
      require(values.size == FluxReferenceImageContracts.INPUT_ELEMENTS) {
        "Reference VAE input has the wrong element count."
      }
      require(values.all { it.isFinite() && it in -1f..1f }) {
        "Reference VAE input contains an invalid value."
      }
      return FluxReferenceImageTensor(values, width, height, crop, decodePlan)
    }
  }
}
