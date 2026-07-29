/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.image

import androidx.exifinterface.media.ExifInterface
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxGraphResult
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxGraphRunner
import java.io.File
import kotlin.test.assertFailsWith
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class FluxReferenceImageContractsTest {
  @Test fun allEightExifCoordinateTransformsAreDistinctAndCorrect() {
    val w = 3; val h = 2; val x = 0; val y = 0
    val expected = mapOf(
      ExifInterface.ORIENTATION_NORMAL to FluxReferenceImageContracts.Point(0, 0),
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL to FluxReferenceImageContracts.Point(2, 0),
      ExifInterface.ORIENTATION_ROTATE_180 to FluxReferenceImageContracts.Point(2, 1),
      ExifInterface.ORIENTATION_FLIP_VERTICAL to FluxReferenceImageContracts.Point(0, 1),
      ExifInterface.ORIENTATION_TRANSPOSE to FluxReferenceImageContracts.Point(0, 0),
      ExifInterface.ORIENTATION_ROTATE_90 to FluxReferenceImageContracts.Point(1, 0),
      ExifInterface.ORIENTATION_TRANSVERSE to FluxReferenceImageContracts.Point(1, 2),
      ExifInterface.ORIENTATION_ROTATE_270 to FluxReferenceImageContracts.Point(0, 2),
    )
    expected.forEach { (orientation, point) ->
      assertEquals(point, FluxReferenceImageContracts.orientPoint(orientation, w, h, x, y))
    }
    // A second asymmetric point distinguishes normal from transpose.
    assertEquals(
      FluxReferenceImageContracts.Point(1, 2),
      FluxReferenceImageContracts.orientPoint(ExifInterface.ORIENTATION_TRANSPOSE, w, h, 2, 1),
    )
    assertEquals(
      FluxReferenceImageContracts.Point(0, 0),
      FluxReferenceImageContracts.orientPoint(ExifInterface.ORIENTATION_TRANSVERSE, w, h, 2, 1),
    )
  }

  @Test fun orientationChangesDimensionsBeforeCrop() {
    val oriented = FluxReferenceImageContracts.orientedSize(ExifInterface.ORIENTATION_ROTATE_90, 7, 4)
    assertEquals(4 to 7, oriented)
    assertEquals(FluxReferenceImageContracts.Crop(0, 1, 4), FluxReferenceImageContracts.centerCrop(oriented.first, oriented.second))
  }

  @Test fun centerCropsLandscapePortraitSquareAndOddDifferences() {
    assertEquals(FluxReferenceImageContracts.Crop(2, 0, 4), FluxReferenceImageContracts.centerCrop(8, 4))
    assertEquals(FluxReferenceImageContracts.Crop(0, 2, 4), FluxReferenceImageContracts.centerCrop(4, 8))
    assertEquals(FluxReferenceImageContracts.Crop(0, 0, 4), FluxReferenceImageContracts.centerCrop(4, 4))
    assertEquals(FluxReferenceImageContracts.Crop(1, 0, 4), FluxReferenceImageContracts.centerCrop(7, 4))
  }

  @Test fun alphaCompositesBeforeResizeDeterministically() {
    assertEquals(0xff000000.toInt(), FluxReferenceImageContracts.compositeBlack(0x00ff6301))
    assertEquals(0xff000000.toInt(), FluxReferenceImageContracts.compositeBlack(0x000102ff))
    assertEquals(0xffff6301.toInt(), FluxReferenceImageContracts.compositeBlack(0xffff6301.toInt()))
    assertEquals(50, FluxReferenceImageContracts.compositeChannel(100, 128))
  }

  @Test fun nchwIsRgbPlaneOrderedAndRowMajorWithExactEndpoints() {
    val pixels = IntArray(256 * 256)
    pixels[0] = 0xffff0000.toInt()
    pixels[1] = 0xff00ff00.toInt()
    pixels[256] = 0xff0000ff.toInt()
    val values = FluxReferenceImageContracts.pixelsToNchw(pixels)
    val plane = 256 * 256
    assertEquals(1f, values[0])
    assertEquals(-1f, values[1])
    assertEquals(-1f, values[256])
    assertEquals(-1f, values[plane])
    assertEquals(1f, values[plane + 1])
    assertEquals(1f, values[2 * plane + 256])
    assertEquals(-1f, FluxReferenceImageContracts.normalize(0))
    assertEquals(1f, FluxReferenceImageContracts.normalize(255))
  }

  @Test fun tensorConstructionIsCancellableInChunks() {
    assertFailsWith<CancellationException> {
      FluxReferenceImageContracts.pixelsToNchw(IntArray(256 * 256)) {
        throw CancellationException("test")
      }
    }
  }

  @Test fun tensorRejectsWrongSizeRangeAndNonFinite() {
    val crop = FluxReferenceImageContracts.Crop(0, 0, 1)
    assertFailsWith<IllegalArgumentException> { FluxReferenceImageTensor.checked(floatArrayOf(), 1, 1, crop) }
    val values = FloatArray(FluxReferenceImageContracts.INPUT_ELEMENTS)
    values[0] = Float.NaN
    assertFailsWith<IllegalArgumentException> { FluxReferenceImageTensor.checked(values, 1, 1, crop) }
    values[0] = 1.01f
    assertFailsWith<IllegalArgumentException> { FluxReferenceImageTensor.checked(values, 1, 1, crop) }
  }

  @Test fun decodePlanRejectsExtremeAspectRatioAllocation() {
    assertFailsWith<IllegalArgumentException> { FluxReferenceImageContracts.decodePlan(Int.MAX_VALUE, 256) }
  }

  @Test fun encoderInvokesExactGraphOnceWithOneOrderedInput() = runTest {
    var calls = 0
    val runner = object : FluxGraphRunner {
      override suspend fun run(modelFile: File, inputs: List<FloatArray>): FluxGraphResult {
        calls++
        assertEquals(FluxReferenceImageContracts.GRAPH, modelFile.name)
        assertEquals(1, inputs.size)
        return FluxGraphResult(modelFile.name, listOf(FloatArray(FluxReferenceImageContracts.OUTPUT_ELEMENTS)))
      }
    }
    val tensor = validTensor()
    val latent = FluxReferenceVaeEncoder(runner).encode(File(FluxReferenceImageContracts.GRAPH), tensor)
    assertEquals(1, calls)
    assertEquals(FluxReferenceImageContracts.OUTPUT_SHAPE, latent.shape)
  }

  @Test fun encoderRejectsWrongOutputCount() = runTest {
    val runner = object : FluxGraphRunner {
      override suspend fun run(modelFile: File, inputs: List<FloatArray>) =
        FluxGraphResult(modelFile.name, emptyList())
    }
    assertFailsWith<IllegalArgumentException> {
      FluxReferenceVaeEncoder(runner).encode(File(FluxReferenceImageContracts.GRAPH), validTensor())
    }
  }

  @Test fun latentRejectsWrongCountAndNonFinite() {
    assertFailsWith<IllegalArgumentException> { FluxReferenceLatent.checked(FloatArray(1)) }
    val values = FloatArray(FluxReferenceImageContracts.OUTPUT_ELEMENTS); values[2] = Float.POSITIVE_INFINITY
    assertFailsWith<IllegalArgumentException> { FluxReferenceLatent.checked(values) }
  }

  private fun validTensor() = FluxReferenceImageTensor.checked(
    FloatArray(FluxReferenceImageContracts.INPUT_ELEMENTS), 256, 256,
    FluxReferenceImageContracts.Crop(0, 0, 256),
  )
}
