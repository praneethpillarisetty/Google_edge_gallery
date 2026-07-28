/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.gallery.customtasks.flux.prompt

import java.io.Closeable
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.Path
import java.nio.file.StandardOpenOption

/** Owns a read-only channel and mapping of the row-major little-endian FP16 embedding table. */
class FluxEmbeddingTable private constructor(
  private val channel: FileChannel,
  private val mapping: MappedByteBuffer,
  val vocabularyRows: Int,
  val valuesPerRow: Int,
) : Closeable {
  @Volatile private var closed = false

  @Synchronized
  fun lookup(tokenIds: IntArray): FloatArray {
    check(!closed) { "Embedding table is closed" }
    val result = FloatArray(Math.multiplyExact(tokenIds.size, valuesPerRow))
    tokenIds.forEachIndexed { outputRow, tokenId ->
      if (tokenId !in 0 until vocabularyRows) throw FluxPromptPreparationException("Embedding token id $tokenId is outside 0 until $vocabularyRows")
      var inputOffset = Math.multiplyExact(tokenId, valuesPerRow * BYTES_PER_VALUE)
      var outputOffset = outputRow * valuesPerRow
      repeat(valuesPerRow) {
        val bits = mapping.getShort(inputOffset).toInt() and 0xffff
        result[outputOffset++] = halfToFloat(bits)
        inputOffset += BYTES_PER_VALUE
      }
    }
    return result
  }

  override fun close() { if (!closed) { closed = true; channel.close() } }

  companion object {
    const val VOCABULARY_ROWS = 151_936
    const val VALUES_PER_ROW = 2_560
    const val BYTES_PER_VALUE = 2
    const val EXPECTED_BYTES: Long = VOCABULARY_ROWS.toLong() * VALUES_PER_ROW * BYTES_PER_VALUE

    @JvmStatic fun open(path: Path): FluxEmbeddingTable = open(path, VOCABULARY_ROWS, VALUES_PER_ROW)

    /** Dimension parameters make the same file contract testable without a multi-gigabyte fixture. */
    @JvmStatic fun open(path: Path, vocabularyRows: Int, valuesPerRow: Int): FluxEmbeddingTable {
      require(vocabularyRows > 0 && valuesPerRow > 0)
      val expected = vocabularyRows.toLong() * valuesPerRow * BYTES_PER_VALUE
      val channel = try { FileChannel.open(path, StandardOpenOption.READ) }
      catch (e: Exception) { throw FluxPromptPreparationException("Unable to open embedding table", e) }
      try {
        val actual = channel.size()
        if (actual != expected) throw FluxPromptPreparationException("Embedding table length is $actual bytes; expected $expected")
        val mapping = channel.map(FileChannel.MapMode.READ_ONLY, 0, expected)
        mapping.order(ByteOrder.LITTLE_ENDIAN)
        return FluxEmbeddingTable(channel, mapping, vocabularyRows, valuesPerRow)
      } catch (e: Exception) {
        channel.close()
        if (e is FluxPromptPreparationException) throw e
        throw FluxPromptPreparationException("Unable to map embedding table", e)
      }
    }

    /** Converts the low 16 bits of [bits] from IEEE-754 binary16 to binary32. */
    @JvmStatic fun halfToFloat(bits: Int): Float {
      val half = bits and 0xffff
      val sign = (half and 0x8000) shl 16
      var exponent = (half ushr 10) and 0x1f
      var fraction = half and 0x3ff
      val floatBits = when (exponent) {
        0 -> {
          if (fraction == 0) sign
          else {
            var shift = 0
            while ((fraction and 0x400) == 0) { fraction = fraction shl 1; shift++ }
            fraction = fraction and 0x3ff
            sign or ((113 - shift) shl 23) or (fraction shl 13)
          }
        }
        0x1f -> sign or 0x7f800000 or (fraction shl 13)
        else -> sign or ((exponent + 112) shl 23) or (fraction shl 13)
      }
      return Float.fromBits(floatBits)
    }
  }
}
