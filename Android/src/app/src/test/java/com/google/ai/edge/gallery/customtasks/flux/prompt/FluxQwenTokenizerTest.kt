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

import java.nio.file.Files
import java.nio.file.Path
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class FluxQwenTokenizerTest {
  private fun fixture(name: String): Path =
    Path.of(requireNotNull(javaClass.getResource("/flux/tokenizer/$name")).toURI())

  private fun tokenizer() = FluxQwenTokenizer.load(
    fixture("minimal_vocab.fixture"), fixture("minimal_merges.fixture"), fixture("minimal_special.fixture"))

  @Test fun `parses implicit vocabulary ids and ranked merges`() {
    // i+n uses rank 0, then the leading encoded space + in uses rank 1.
    assertArrayEquals(intArrayOf(6), tokenizer().tokenize(" in"))
  }

  @Test fun `tokenizes ascii whitespace punctuation and unicode`() {
    assertArrayEquals(intArrayOf(1, 6, 0, 5, 9), tokenizer().tokenize("A in! é"))
  }

  @Test fun `preserves composed and decomposed unicode bytes`() {
    val instance = tokenizer()
    val composed = instance.tokenize("é")
    val decomposed = instance.tokenize("e\u0301")

    assertArrayEquals(intArrayOf(9), composed)
    assertArrayEquals(intArrayOf(12, 15), decomposed)
    assertFalse(composed.contentEquals(decomposed))
  }

  @Test fun `recognizes special token without splitting it`() {
    assertArrayEquals(intArrayOf(1, 11, 4), tokenizer().tokenize("A<|im_start|>in"))
  }

  @Test fun `repeated tokenization is deterministic`() {
    val instance = tokenizer()
    assertArrayEquals(instance.tokenize("A in! é"), instance.tokenize("A in! é"))
  }

  @Test fun `prepares exactly 512 positions without adding bos or eos`() {
    val prepared = tokenizer().prepare("A")
    assertEquals(512, prepared.tokenIds.size)
    assertEquals(1, prepared.tokenIds[0])
    assertEquals(10, prepared.tokenIds[1])
    assertTrue(prepared.validPositions[0])
    assertFalse(prepared.validPositions[1])

    val truncated = tokenizer().prepare("A".repeat(513))
    assertTrue(truncated.tokenIds.all { it == 1 })
    assertTrue(truncated.validPositions.all { it })
  }

  @Test fun `rejects malformed and duplicate entries`() {
    val directory = Files.createTempDirectory("bad-tokenizer")
    val vocab = directory.resolve("vocab").also { Files.writeString(it, "!\n!\n<|endoftext|>\n") }
    val merges = directory.resolve("merges").also { Files.writeString(it, "bad merge record\n") }
    val specials = directory.resolve("specials").also { Files.writeString(it, "<|endoftext|>\t2\n") }
    assertThrows(FluxPromptPreparationException::class.java) { FluxQwenTokenizer.load(vocab, merges, specials) }
  }

  @Test fun `rejects duplicate special ids and invalid merge references`() {
    val directory = Files.createTempDirectory("bad-tokenizer")
    val vocab = directory.resolve("vocab").also { Files.writeString(it, "!\ni\nn\nin\n<|endoftext|>\n<special>\n") }
    val merge = directory.resolve("merge").also { Files.writeString(it, "i missing\n") }
    val noMerges = directory.resolve("no-merges").also { Files.writeString(it, "") }
    val duplicate = directory.resolve("special").also { Files.writeString(it, "<|endoftext|>\t4\n<special>\t4\n") }
    assertThrows(FluxPromptPreparationException::class.java) { FluxQwenTokenizer.load(vocab, noMerges, duplicate) }
    assertThrows(FluxPromptPreparationException::class.java) { FluxQwenTokenizer.load(vocab, merge, directory.resolve("valid").also { Files.writeString(it, "<|endoftext|>\t4\n") }) }
  }
}
