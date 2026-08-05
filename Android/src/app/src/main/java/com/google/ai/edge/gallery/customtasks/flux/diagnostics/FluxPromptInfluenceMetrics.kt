/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.diagnostics

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.sqrt

data class FluxFp32Summary(val elementCount: Int, val allFinite: Boolean, val sha256LittleEndian: String, val min: Float, val max: Float, val mean: Double, val stddev: Double)
data class FluxTensorDiff(val equalCount: Int, val equalPercent: Double, val mae: Double, val maxAbsError: Float, val rmse: Double, val cosineSimilarity: Double?, val different: Boolean)
data class FluxTokenDiff(val tokenCount: Int, val aSha256LittleEndianInt64: String, val bSha256LittleEndianInt64: String, val differingPositions: Int, val differingPercent: Double)
data class FluxBitmapDiff(val width: Int, val height: Int, val aArgbSha256: String, val bArgbSha256: String, val differingPixels: Int, val differingPercent: Double, val meanAbsA: Double, val meanAbsR: Double, val meanAbsG: Double, val meanAbsB: Double, val rgbRmse: Double)

enum class FluxPromptInfluenceInterpretation { PROMPT_INFLUENCE_OBSERVED, PROMPT_DIFFERENCE_LOST_BEFORE_TEXT_CONDITIONING, PROMPT_DIFFERENCE_LOST_DURING_TRANSFORMER, FINAL_BITMAPS_NEARLY_IDENTICAL, COMPARISON_INCONCLUSIVE, CANCELLED, ERROR }

object FluxPromptInfluenceMetrics {
  const val TENSOR_DIFFERENT_MAE_THRESHOLD = 1.0e-6
  const val BITMAP_DIFFERENT_PIXEL_PERCENT_THRESHOLD = 0.1

  fun summarizeFp32(values: FloatArray): FluxFp32Summary {
    require(values.isNotEmpty() && values.all(Float::isFinite)) { "FP32 diagnostic tensors must be finite." }
    var min = Float.POSITIVE_INFINITY; var max = Float.NEGATIVE_INFINITY; var sum = 0.0
    values.forEach { if (it < min) min = it; if (it > max) max = it; sum += it.toDouble() }
    val mean = sum / values.size
    val variance = values.fold(0.0) { acc, v -> val d = v - mean; acc + d * d } / values.size
    return FluxFp32Summary(values.size, true, shaFloats(values), min, max, mean, sqrt(variance))
  }

  fun compareFp32(a: FloatArray, b: FloatArray): FluxTensorDiff {
    require(a.size == b.size && a.isNotEmpty()) { "Tensor sizes must match." }
    require(a.all(Float::isFinite) && b.all(Float::isFinite)) { "FP32 diagnostic tensors must be finite." }
    var equal = 0; var absSum = 0.0; var maxAbs = 0f; var sq = 0.0; var dot = 0.0; var na = 0.0; var nb = 0.0
    for (i in a.indices) { val av=a[i]; val bv=b[i]; if (av.toBits()==bv.toBits()) equal++; val d=kotlin.math.abs(av-bv); absSum += d; if (d>maxAbs) maxAbs=d; sq += d.toDouble()*d; dot += av.toDouble()*bv; na += av.toDouble()*av; nb += bv.toDouble()*bv }
    val cosine = if (na == 0.0 || nb == 0.0) null else dot / sqrt(na * nb)
    val mae = absSum / a.size
    return FluxTensorDiff(equal, 100.0 * equal / a.size, mae, maxAbs, sqrt(sq / a.size), cosine, mae > TENSOR_DIFFERENT_MAE_THRESHOLD || maxAbs > TENSOR_DIFFERENT_MAE_THRESHOLD)
  }

  fun compareTokens(a: LongArray, b: LongArray): FluxTokenDiff {
    require(a.size == b.size && a.isNotEmpty())
    val diff = a.indices.count { a[it] != b[it] }
    return FluxTokenDiff(a.size, shaLongs(a), shaLongs(b), diff, 100.0 * diff / a.size)
  }

  fun compareArgb(width: Int, height: Int, a: IntArray, b: IntArray): FluxBitmapDiff {
    require(width > 0 && height > 0 && a.size == width * height && b.size == a.size)
    var diff = 0; val sums = DoubleArray(4); var rgbSq = 0.0
    for (i in a.indices) { if (a[i] != b[i]) diff++; for (c in 0..3) { val d = kotlin.math.abs(((a[i] ushr ((3-c)*8)) and 255) - ((b[i] ushr ((3-c)*8)) and 255)); sums[c] += d; if (c>0) rgbSq += d*d } }
    return FluxBitmapDiff(width,height,shaInts(a),shaInts(b),diff,100.0*diff/a.size,sums[0]/a.size,sums[1]/a.size,sums[2]/a.size,sums[3]/a.size,sqrt(rgbSq/(a.size*3)))
  }

  fun interpret(tokenDiff: FluxTokenDiff?, text: FluxTensorDiff?, transformer: FluxTensorDiff?, bitmap: FluxBitmapDiff?): FluxPromptInfluenceInterpretation = when {
    tokenDiff == null || text == null -> FluxPromptInfluenceInterpretation.COMPARISON_INCONCLUSIVE
    tokenDiff.differingPositions == 0 -> FluxPromptInfluenceInterpretation.PROMPT_DIFFERENCE_LOST_BEFORE_TEXT_CONDITIONING
    !text.different -> FluxPromptInfluenceInterpretation.PROMPT_DIFFERENCE_LOST_BEFORE_TEXT_CONDITIONING
    transformer != null && !transformer.different -> FluxPromptInfluenceInterpretation.PROMPT_DIFFERENCE_LOST_DURING_TRANSFORMER
    bitmap != null && bitmap.differingPercent <= BITMAP_DIFFERENT_PIXEL_PERCENT_THRESHOLD -> FluxPromptInfluenceInterpretation.FINAL_BITMAPS_NEARLY_IDENTICAL
    else -> FluxPromptInfluenceInterpretation.PROMPT_INFLUENCE_OBSERVED
  }

  private fun shaFloats(v: FloatArray)=digest(v.size*4){ bb -> v.forEach { bb.putInt(it.toBits()) } }
  private fun shaLongs(v: LongArray)=digest(v.size*8){ bb -> v.forEach { bb.putLong(it) } }
  private fun shaInts(v: IntArray)=digest(v.size*4){ bb -> v.forEach { bb.putInt(it) } }
  private fun digest(bytes:Int, fill:(ByteBuffer)->Unit):String { val bb=ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN); fill(bb); return MessageDigest.getInstance("SHA-256").digest(bb.array()).joinToString("") { "%02x".format(it) } }
}
