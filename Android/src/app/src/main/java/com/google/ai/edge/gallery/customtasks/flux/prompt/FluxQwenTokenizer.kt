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

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.ConcurrentHashMap
import java.util.regex.Pattern

/** Qwen2 byte-level BPE backed solely by the exported FLUX tokenizer text files. */
class FluxQwenTokenizer private constructor(private val data: TokenizerData) {
  fun tokenize(text: String): IntArray {
    val result = ArrayList<Int>()
    var start = 0
    while (start < text.length) {
      val special = data.specialsByLength.firstOrNull { text.startsWith(it, start) }
      if (special != null) {
        result += data.specialTokens.getValue(special)
        start += special.length
        continue
      }
      var end = start + 1
      while (end < text.length && data.specialsByLength.none { text.startsWith(it, end) }) end++
      tokenizeOrdinary(text.substring(start, end), result)
      start = end
    }
    return result.toIntArray()
  }

  /** Literal-prompt encoding: no wrapper or BOS/EOS; right-pad/truncate to the export width. */
  fun prepare(text: String): FluxPromptTokens {
    val unpadded = tokenize(text)
    val count = minOf(unpadded.size, MAX_TOKENS)
    val ids = IntArray(MAX_TOKENS) { data.paddingId }
    val valid = BooleanArray(MAX_TOKENS)
    unpadded.copyInto(ids, endIndex = count)
    for (i in 0 until count) valid[i] = true
    return FluxPromptTokens(ids, valid)
  }

  /** Applies the authoritative Qwen chat wrapper, truncating only the literal UTF-8 body. */
  fun prepareForTextEncoder(text: String): FluxPromptTokens {
    val body = tokenize(text)
    val bodyCount = minOf(body.size, MAX_TOKENS - TEMPLATE_PREFIX.size - TEMPLATE_SUFFIX.size)
    val count = TEMPLATE_PREFIX.size + bodyCount + TEMPLATE_SUFFIX.size
    val ids = IntArray(MAX_TOKENS) { data.paddingId }
    TEMPLATE_PREFIX.copyInto(ids)
    body.copyInto(ids, TEMPLATE_PREFIX.size, 0, bodyCount)
    TEMPLATE_SUFFIX.copyInto(ids, TEMPLATE_PREFIX.size + bodyCount)
    return FluxPromptTokens(ids, BooleanArray(MAX_TOKENS) { it < count })
  }

  /** Authoritative body count for debug budgeting; does not add the Qwen wrapper. */
  fun bodyTokenCount(text: String): Int = tokenize(text).size

  private fun tokenizeOrdinary(text: String, output: MutableList<Int>) {
    val matcher = PRETOKENIZER.matcher(text)
    var consumed = 0
    while (matcher.find()) {
      if (matcher.start() != consumed) throw FluxPromptPreparationException("Qwen pre-tokenizer left input unmatched at character $consumed")
      val symbols = matcher.group().toByteArray(StandardCharsets.UTF_8).map { BYTE_ENCODER[it.toInt() and 0xff].toString() }
      for (token in applyBpe(symbols)) {
        output += data.vocabulary[token]
          ?: throw FluxPromptPreparationException("Tokenizer vocabulary has no entry for BPE token '$token'")
      }
      consumed = matcher.end()
    }
    if (consumed != text.length) throw FluxPromptPreparationException("Qwen pre-tokenizer left input unmatched at character $consumed")
  }

  private fun applyBpe(initial: List<String>): List<String> {
    val pieces = initial.toMutableList()
    while (pieces.size > 1) {
      var bestIndex = -1
      var bestRank = Int.MAX_VALUE
      for (i in 0 until pieces.lastIndex) {
        val rank = data.mergeRanks[Pair(pieces[i], pieces[i + 1])] ?: continue
        if (rank < bestRank) { bestRank = rank; bestIndex = i }
      }
      if (bestIndex < 0) break
      val left = pieces[bestIndex]
      val right = pieces[bestIndex + 1]
      // A BPE pass merges every occurrence of the selected pair, left to right.
      var i = 0
      while (i < pieces.lastIndex) {
        if (pieces[i] == left && pieces[i + 1] == right) {
          pieces[i] = left + right
          pieces.removeAt(i + 1)
        } else i++
      }
    }
    return pieces
  }

  companion object {
    const val MAX_TOKENS = 512
    val MAX_BODY_TOKENS: Int get() = MAX_TOKENS - TEMPLATE_PREFIX.size - TEMPLATE_SUFFIX.size
    const val END_OF_TEXT = "<|endoftext|>"
    private val TEMPLATE_PREFIX = intArrayOf(151644, 872, 198)
    private val TEMPLATE_SUFFIX = intArrayOf(151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271)
    private val PRETOKENIZER = Pattern.compile(
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    )
    private val BYTE_ENCODER = buildByteEncoder()
    private val cache = ConcurrentHashMap<CacheKey, TokenizerData>()

    @JvmStatic
    @Throws(FluxPromptPreparationException::class)
    fun load(vocabulary: Path, merges: Path, specials: Path): FluxQwenTokenizer {
      val key = try {
        CacheKey(vocabulary.toAbsolutePath(), merges.toAbsolutePath(), specials.toAbsolutePath(),
          Files.size(vocabulary), Files.size(merges), Files.size(specials),
          Files.getLastModifiedTime(vocabulary).toMillis(), Files.getLastModifiedTime(merges).toMillis(), Files.getLastModifiedTime(specials).toMillis())
      } catch (e: Exception) { throw FluxPromptPreparationException("Unable to inspect tokenizer files", e) }
      val parsed = cache[key] ?: parse(vocabulary, merges, specials).also { cache.putIfAbsent(key, it) }
      return FluxQwenTokenizer(cache[key] ?: parsed)
    }

    private fun parse(vocabularyFile: Path, mergesFile: Path, specialsFile: Path): TokenizerData {
      try {
        val vocab = LinkedHashMap<String, Int>()
        Files.newBufferedReader(vocabularyFile, StandardCharsets.UTF_8).useLines { lines ->
          lines.forEachIndexed { id, token ->
            if (token.isEmpty()) throw FluxPromptPreparationException("Empty vocabulary token at line ${id + 1}")
            if (vocab.put(token, id) != null) throw FluxPromptPreparationException("Duplicate vocabulary token '$token' at line ${id + 1}")
          }
        }
        if (vocab.isEmpty()) throw FluxPromptPreparationException("Vocabulary is empty")

        val ranks = HashMap<Pair<String, String>, Int>()
        Files.newBufferedReader(mergesFile, StandardCharsets.UTF_8).useLines { lines ->
          lines.forEachIndexed { rank, line ->
            val separator = line.indexOf(' ')
            if (separator <= 0 || separator != line.lastIndexOf(' ') || separator == line.lastIndex) {
              throw FluxPromptPreparationException("Invalid merge at line ${rank + 1}: expected exactly two non-empty tokens")
            }
            val pair = Pair(line.substring(0, separator), line.substring(separator + 1))
            if (ranks.put(pair, rank) != null) throw FluxPromptPreparationException("Duplicate merge at line ${rank + 1}")
            if (!vocab.containsKey(pair.first) || !vocab.containsKey(pair.second) || !vocab.containsKey(pair.first + pair.second)) {
              throw FluxPromptPreparationException("Merge at line ${rank + 1} references a missing vocabulary token")
            }
          }
        }

        val specialTokens = LinkedHashMap<String, Int>()
        val specialIds = HashSet<Int>()
        Files.newBufferedReader(specialsFile, StandardCharsets.UTF_8).useLines { lines ->
          lines.forEachIndexed { index, line ->
            val fields = line.split('\t')
            val id = fields.getOrNull(1)?.toIntOrNull()
            if (fields.size != 2 || fields[0].isEmpty() || id == null || id < 0) throw FluxPromptPreparationException("Invalid special token at line ${index + 1}")
            if (specialTokens.put(fields[0], id) != null || !specialIds.add(id)) throw FluxPromptPreparationException("Duplicate special token or id at line ${index + 1}")
            if (vocab[fields[0]] != id) throw FluxPromptPreparationException("Special token '${fields[0]}' id $id does not match the vocabulary")
          }
        }
        val paddingId = specialTokens[END_OF_TEXT] ?: throw FluxPromptPreparationException("Missing required special token $END_OF_TEXT")
        return TokenizerData(vocab.toMap(), ranks.toMap(), specialTokens.toMap(), specialTokens.keys.sortedByDescending { it.length }, paddingId)
      } catch (e: FluxPromptPreparationException) { throw e }
      catch (e: Exception) { throw FluxPromptPreparationException("Unable to parse tokenizer files", e) }
    }

    private fun buildByteEncoder(): Array<Char> {
      val safe = (('!'.code..'~'.code) + ('¡'.code..'¬'.code) + ('®'.code..'ÿ'.code)).toMutableList()
      val bytes = safe.toMutableList()
      var extra = 0
      for (byte in 0..255) if (byte !in safe) { bytes += byte; safe += 256 + extra++ }
      return Array(256) { byte -> safe[bytes.indexOf(byte)].toChar() }
    }
  }
}

private data class TokenizerData(
  val vocabulary: Map<String, Int>,
  val mergeRanks: Map<Pair<String, String>, Int>,
  val specialTokens: Map<String, Int>,
  val specialsByLength: List<String>,
  val paddingId: Int,
)

private data class CacheKey(
  val vocabulary: Path, val merges: Path, val specials: Path,
  val vocabularySize: Long, val mergesSize: Long, val specialsSize: Long,
  val vocabularyModified: Long, val mergesModified: Long, val specialsModified: Long,
)
