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

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class FluxEmbeddingTableTest {
  @Test fun `converts all IEEE 754 binary16 classes`() {
    fun convert(bits: Int) = FluxEmbeddingTable.halfToFloat(bits)
    assertEquals(0.0f.toRawBits(), convert(0x0000).toRawBits())
    assertEquals((-0.0f).toRawBits(), convert(0x8000).toRawBits())
    assertEquals(1.0f, convert(0x3c00))
    assertEquals(-1.0f, convert(0xbc00))
    assertEquals(6.1035156e-5f, convert(0x0400))
    assertEquals(5.9604645e-8f, convert(0x0001))
    assertEquals(Float.POSITIVE_INFINITY, convert(0x7c00))
    assertEquals(Float.NEGATIVE_INFINITY, convert(0xfc00))
    assertTrue(convert(0x7e01).isNaN())
  }

  @Test fun `looks up rows from a little endian read-only mapping`() {
    val file = Files.createTempFile("embedding", ".bin")
    val bytes = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN)
    shortArrayOf(0x3c00, 0x4000, 0x4200, 0xc400.toShort(), 0x0001, 0x7c00).forEach { bytes.putShort(it) }
    Files.write(file, bytes.array())
    FluxEmbeddingTable.open(file, vocabularyRows = 3, valuesPerRow = 2).use { table ->
      val rows = table.lookup(intArrayOf(2, 0))
      assertEquals(4, rows.size)
      assertEquals(5.9604645e-8f, rows[0])
      assertEquals(Float.POSITIVE_INFINITY, rows[1])
      assertEquals(1f, rows[2])
      assertEquals(2f, rows[3])
    }
  }

  @Test fun `rejects invalid file length and out of range ids`() {
    val file = Files.createTempFile("embedding", ".bin")
    Files.write(file, ByteArray(4))
    assertThrows(FluxPromptPreparationException::class.java) { FluxEmbeddingTable.open(file, 2, 2) }
    Files.write(file, ByteArray(8))
    FluxEmbeddingTable.open(file, 2, 2).use { table ->
      assertThrows(FluxPromptPreparationException::class.java) { table.lookup(intArrayOf(-1)) }
      assertThrows(FluxPromptPreparationException::class.java) { table.lookup(intArrayOf(2)) }
    }
  }

  @Test fun `closed table rejects lookup`() {
    val file = Files.createTempFile("embedding", ".bin")
    Files.write(file, ByteArray(2))
    val table = FluxEmbeddingTable.open(file, 1, 1)
    table.close()
    assertThrows(IllegalStateException::class.java) { table.lookup(intArrayOf(0)) }
    table.close()
  }
}
