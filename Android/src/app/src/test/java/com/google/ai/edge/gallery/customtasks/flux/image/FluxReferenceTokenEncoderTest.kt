package com.google.ai.edge.gallery.customtasks.flux.image

import java.io.ByteArrayInputStream
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.cancel
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.Job
import kotlinx.coroutines.withContext

class FluxReferenceTokenEncoderTest {
  @Test fun metadataAndProvenanceAreStrictlyValidated() {
    val constants = parse(bundle())
    assertEquals(FluxReferenceConstantsParser.REVISION, constants.revision)
    val bad = bundle().also { it["flux_reference_constants.json"] = it.getValue("flux_reference_constants.json").decodeToString().replace("\"schemaVersion\": 1", "\"schemaVersion\": 2").encodeToByteArray() }
    assertFailsWith<FluxReferenceTokenException> { parse(bad) }
  }

  @Test fun unknownMetadataIsRejected() {
    val bad = bundle().also { it["flux_reference_constants.json"] = it.getValue("flux_reference_constants.json").decodeToString().replaceFirst("{", "{\"unexpected\":true,").encodeToByteArray() }
    assertFailsWith<FluxReferenceTokenException> { parse(bad) }
  }

  @Test fun authoritativeBinarySizesCountsHashesEndianAndValuesValidate() {
    val files = bundle()
    assertEquals(512, files.getValue("bn_mean.bin").size)
    assertEquals(512, files.getValue("bn_std.bin").size)
    assertEquals(131_072, files.getValue("patch_perm.bin").size)
    assertEquals(FluxReferenceConstantsParser.MEAN_SHA256, sha(files.getValue("bn_mean.bin")))
    assertEquals(FluxReferenceConstantsParser.STD_SHA256, sha(files.getValue("bn_std.bin")))
    assertEquals(FluxReferenceConstantsParser.PERM_SHA256, sha(files.getValue("patch_perm.bin")))
    val constants = parse(files)
    assertEquals(128, constants.mean.size)
    assertEquals(128, constants.standardDeviation.size)
    assertEquals(32_768, constants.patchPermutation.size)
    assertEquals(-0.0673828125f, constants.mean.first())
    assertEquals(1.8028034f, constants.standardDeviation.first())
    assertTrue(constants.mean.all(Float::isFinite))
    assertTrue(constants.standardDeviation.all { it.isFinite() && it > 0f })
  }

  @Test fun everyWrongBinarySizeOrHashIsRejected() {
    for (name in listOf("bn_mean.bin", "bn_std.bin", "patch_perm.bin")) {
      val short = bundle().also { it[name] = it.getValue(name).copyOf(it.getValue(name).size - 1) }
      assertFailsWith<FluxReferenceTokenException>(name) { parse(short) }
      val corrupt = bundle().also { it.getValue(name)[0] = (it.getValue(name)[0].toInt() xor 1).toByte() }
      assertFailsWith<FluxReferenceTokenException>(name) { parse(corrupt) }
    }
  }

  @Test fun permutationIsExactBijectionWithAuthoritativeEnds() {
    val permutation = parse(bundle()).patchPermutation
    assertEquals(0, permutation.min())
    assertEquals(32_767, permutation.max())
    assertEquals(32_768, permutation.toSet().size)
    assertContentEquals(IntArray(16) { it * 2 }, permutation.copyOfRange(0, 16))
    assertContentEquals(IntArray(10) { 32_749 + it * 2 }, permutation.copyOfRange(32_758, 32_768))
    assertContentEquals((0..32_767).toList(), permutation.sorted())
  }

  @Test fun invalidPermutationsAreRejected() {
    val identity = IntArray(32_768) { it }
    FluxReferenceConstantsParser.validatePermutation(identity)
    assertFailsWith<FluxReferenceTokenException> { FluxReferenceConstantsParser.validatePermutation(identity.copyOf().also { it[1] = 0 }) }
    assertFailsWith<FluxReferenceTokenException> { FluxReferenceConstantsParser.validatePermutation(identity.copyOf().also { it[0] = -1 }) }
    assertFailsWith<FluxReferenceTokenException> { FluxReferenceConstantsParser.validatePermutation(identity.copyOf(32_767)) }
  }

  @Test fun patchCornersOffsetsAndTokenOrderingMatchAuthoritativeGather() = runTest {
    val constants = parse(bundle())
    val source = FloatArray(32_768) { it.toFloat() }
    val output = FluxReferenceTokenEncoder(constants).encode(FluxReferenceLatent.checked(source)).copyValues()
    fun expected(spatial: Int, channel: Int): Float {
      val packedIndex = channel * 256 + spatial
      return (constants.patchPermutation[packedIndex] - constants.mean[channel]) / constants.standardDeviation[channel]
    }
    for (channel in 0..3) assertEquals(expected(0, channel), output[channel])
    assertEquals(expected(255, 127), output[255 * 128 + 127])
    assertEquals(expected(1, 0), output[128])
    assertEquals(listOf(0, 1, 32, 33), (0..3).map { constants.patchPermutation[it * 256] })
    assertEquals(listOf(990, 991, 1022, 1023), (0..3).map { constants.patchPermutation[it * 256 + 255] })
  }

  @Test fun outputContractNormalizationNoSecondEpsilonAndOwnership() = runTest {
    val constants = parse(bundle())
    val input = FloatArray(32_768)
    val original = input.copyOf()
    val tokens = FluxReferenceTokenEncoder(constants).encode(FluxReferenceLatent.checked(input))
    val first = (0f - constants.mean[0]) / constants.standardDeviation[0]
    assertTrue(abs(tokens.copyValues()[0] - first) < 0.000001f)
    assertFalse(abs(tokens.copyValues()[0] - ((0f - constants.mean[0]) / kotlin.math.sqrt(constants.standardDeviation[0] * constants.standardDeviation[0] + 0.0001f))) < 0.00000001f)
    assertEquals(listOf(1, 256, 128), tokens.shape)
    assertEquals(32_768, tokens.size)
    assertTrue(tokens.allFinite())
    assertContentEquals(original, input)
    val leaked = tokens.copyValues().also { it[0] = 999f }
    assertFalse(tokens.copyValues()[0] == leaked[0])
    assertContentEquals(tokens.copyValues(), FluxReferenceTokenEncoder(constants).encode(FluxReferenceLatent.checked(input)).copyValues())
  }

  @Test fun typedContractsRejectWrongCountsAndNonFiniteValues() {
    assertFailsWith<IllegalArgumentException> { FluxReferenceLatent.checked(FloatArray(1)) }
    assertFailsWith<IllegalArgumentException> { FluxReferenceLatent.checked(FloatArray(32_768).also { it[4] = Float.NaN }) }
    assertFailsWith<FluxReferenceTokenException> { FluxReferenceTokens.checked(FloatArray(1)) }
    assertFailsWith<FluxReferenceTokenException> { FluxReferenceTokens.checked(FloatArray(32_768).also { it[4] = Float.POSITIVE_INFINITY }) }
  }

  @Test fun invalidConstantsAndNonFiniteNormalizationAreRejected() = runTest {
    val valid = parse(bundle())
    for (bad in listOf(0f, -1f, Float.NaN, Float.POSITIVE_INFINITY)) {
      assertFailsWith<FluxReferenceTokenException> {
        FluxReferenceConstantsParser.validateStandardDeviation(
          valid.standardDeviation.copyOf().also { it[0] = bad }
        )
      }
    }
    assertFailsWith<FluxReferenceTokenException> {
      FluxReferenceConstantsParser.validateMean(valid.mean.copyOf().also { it[0] = Float.NaN })
    }
    val tiny = valid.standardDeviation.copyOf().also { it[0] = Float.MIN_VALUE }
    val constants = valid.copy(standardDeviation = tiny)
    assertFailsWith<FluxReferenceTokenException> { FluxReferenceTokenEncoder(constants).encode(FluxReferenceLatent.checked(FloatArray(32_768).also { it[0] = Float.MAX_VALUE })) }
  }

  @Test fun cancellationIsObservedBeforeAndAtEveryProcessingBoundary() = runTest {
    assertFailsWith<CancellationException> {
      withContext(Job().also { it.cancel() }) {
        FluxReferenceTokenEncoder(parse(bundle())).encode(FluxReferenceLatent.checked(FloatArray(32_768)))
      }
    }
    // Separate child contexts below verify pre-start, gather, normalization, and pre-return checks.
    for (stage in FluxReferenceTokenStage.entries) {
      assertFailsWith<CancellationException> {
        kotlinx.coroutines.coroutineScope {
          FluxReferenceTokenEncoder(parse(bundle())).encode(FluxReferenceLatent.checked(FloatArray(32_768))) {
            if (it == stage) cancel()
          }
        }
      }
    }
  }

  private fun parse(files: Map<String, ByteArray>) = FluxReferenceConstantsParser.parse { ByteArrayInputStream(files.getValue(it)) }
  private fun bundle() = listOf("flux_reference_constants.json", "bn_mean.bin", "bn_std.bin", "patch_perm.bin").associateWith { Files.readAllBytes(evidenceDirectory().resolve(it)) }.toMutableMap()
  private fun sha(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
  private fun evidenceDirectory(): Path {
    var current = Path.of("").toAbsolutePath()
    while (current.parent != null) {
      current.resolve("evidence/flux-reference-constants").takeIf(Files::isDirectory)?.let { return it }
      current = current.parent
    }
    error("Authoritative evidence directory not found.")
  }
}
