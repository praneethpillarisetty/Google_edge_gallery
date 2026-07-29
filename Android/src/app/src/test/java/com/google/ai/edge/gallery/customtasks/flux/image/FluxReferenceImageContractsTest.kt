package com.google.ai.edge.gallery.customtasks.flux.image

import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxGraphResult
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxGraphRunner
import java.io.File
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.test.assertFailsWith

class FluxReferenceImageContractsTest {
  @Test fun centerCropsLandscapePortraitSquareAndOddDifferences() {
    assertEquals(FluxReferenceImageContracts.Crop(2, 0, 4), FluxReferenceImageContracts.centerCrop(8, 4))
    assertEquals(FluxReferenceImageContracts.Crop(0, 2, 4), FluxReferenceImageContracts.centerCrop(4, 8))
    assertEquals(FluxReferenceImageContracts.Crop(0, 0, 4), FluxReferenceImageContracts.centerCrop(4, 4))
    assertEquals(FluxReferenceImageContracts.Crop(1, 0, 4), FluxReferenceImageContracts.centerCrop(7, 4))
  }
  @Test fun alphaCompositesOntoBlackWithIntegerRounding() {
    assertEquals(0xff000000.toInt(), FluxReferenceImageContracts.compositeBlack(0x00ff6301))
    assertEquals(0xffff6301.toInt(), FluxReferenceImageContracts.compositeBlack(0xffff6301.toInt()))
    assertEquals(50, FluxReferenceImageContracts.compositeChannel(100, 128))
  }
  @Test fun normalizationHasExactEndpoints() {
    assertEquals(-1f, FluxReferenceImageContracts.normalize(0))
    assertEquals(1f, FluxReferenceImageContracts.normalize(255))
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
    val tensor = FluxReferenceImageTensor.checked(FloatArray(FluxReferenceImageContracts.INPUT_ELEMENTS), 256, 256, FluxReferenceImageContracts.Crop(0, 0, 256))
    val latent = FluxReferenceVaeEncoder(runner).encode(File(FluxReferenceImageContracts.GRAPH), tensor)
    assertEquals(1, calls)
    assertEquals(FluxReferenceImageContracts.OUTPUT_SHAPE, latent.shape)
  }
  @Test fun latentRejectsWrongCountAndNonFinite() {
    assertFailsWith<IllegalArgumentException> { FluxReferenceLatent.checked(FloatArray(1)) }
    val values = FloatArray(FluxReferenceImageContracts.OUTPUT_ELEMENTS); values[2] = Float.POSITIVE_INFINITY
    assertFailsWith<IllegalArgumentException> { FluxReferenceLatent.checked(values) }
  }
}
