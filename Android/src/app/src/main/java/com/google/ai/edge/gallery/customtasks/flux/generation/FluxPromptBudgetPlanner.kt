/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.generation

fun interface FluxBodyTokenCounter { fun count(text: String): Int }

/** Deterministic JVM fallback; production can inject the installed authoritative tokenizer. */
object FluxUtf8UpperBoundTokenCounter : FluxBodyTokenCounter {
  override fun count(text: String) = Regex("\\S+").findAll(text).count()
}

data class FluxPromptBudgetResult(
  val text: String,
  val bodyTokenCount: Int,
  val maximumBodyTokens: Int,
  val truncationOccurred: Boolean,
  val includedSectionNames: Set<String>,
  val omittedGeneratedSectionNames: Set<String>,
)

/** Pure deterministic planner; callers may inject the authoritative installed tokenizer. */
class FluxPromptBudgetPlanner(
  private val counter: FluxBodyTokenCounter,
  private val maximumBodyTokens: Int,
) {
  init { require(maximumBodyTokens > 0) }

  fun plan(sections: LinkedHashMap<String, String>, literalSections: Set<String>): FluxPromptBudgetResult {
    val unique = sections.entries.distinctBy { it.value }.associateTo(linkedMapOf()) { it.toPair() }
    val included = linkedMapOf<String, String>()
    var truncated = false
    // Literal sections are admitted first and remain in their final semantic position.
    unique.filterKeys { it in literalSections }.forEach { (name, value) -> included[name] = value }
    var literalText = compose(unique, included)
    if (counter.count(literalText) > maximumBodyTokens) {
      literalSections.forEach { name ->
        val value = included[name] ?: return@forEach
        included[name] = truncateCodePoints(value) { candidate ->
          counter.count(compose(unique, LinkedHashMap(included).apply { put(name, candidate) })) <= maximumBodyTokens
        }
      }
      truncated = true
    }
    unique.filterKeys { it !in literalSections }.forEach { (name, value) ->
      val candidate = included + (name to value)
      if (counter.count(compose(unique, candidate)) <= maximumBodyTokens) included[name] = value
    }
    val text = compose(unique, included)
    val omitted = unique.keys.filterTo(linkedSetOf()) { it !in included && it !in literalSections }
    return FluxPromptBudgetResult(text, counter.count(text), maximumBodyTokens, truncated || omitted.isNotEmpty(), included.keys, omitted)
  }

  private fun compose(order: LinkedHashMap<String, String>, included: Map<String, String>): String =
    order.keys.filter { it in included }.joinToString("\n\n") { included.getValue(it) }

  private fun truncateCodePoints(value: String, fits: (String) -> Boolean): String {
    val points = value.codePoints().toArray()
    var low = 0; var high = points.size
    while (low < high) {
      val middle = (low + high + 1) / 2
      val candidate = String(points, 0, middle)
      if (fits(candidate)) low = middle else high = middle - 1
    }
    return String(points, 0, low)
  }
}
