/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.prompt

import com.google.ai.edge.gallery.customtasks.flux.generation.FluxBodyTokenCounter

/** Uses the installed authoritative vocabulary/merges without adding a chat wrapper. */
class FluxAuthoritativeBodyTokenCounter private constructor(private val tokenizer: FluxQwenTokenizer) : FluxBodyTokenCounter {
  override fun count(text: String): Int = tokenizer.bodyTokenCount(text)

  companion object {
    fun create(assets: FluxPromptAssets): FluxAuthoritativeBodyTokenCounter = FluxAuthoritativeBodyTokenCounter(
      FluxQwenTokenizer.load(assets.vocabulary.toPath(), assets.merges.toPath(), assets.specials.toPath())
    )
    val maximumBodyTokens: Int get() = FluxQwenTokenizer.MAX_BODY_TOKENS
  }
}
