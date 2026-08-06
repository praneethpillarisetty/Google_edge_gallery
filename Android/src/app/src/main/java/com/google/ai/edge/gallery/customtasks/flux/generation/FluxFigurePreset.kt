/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.generation

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

@Serializable
data class FluxFigureAttributes(
  val overallBuild: String = "",
  val shoulders: String = "",
  val torso: String = "",
  val waist: String = "",
  val hips: String = "",
  val legs: String = "",
  val heightImpression: String = "",
) {
  fun description(): String = listOf(overallBuild, shoulders, torso, waist, hips, legs, heightImpression)
    .map(String::trim).filter(String::isNotEmpty).joinToString(", ")
}

@Serializable
data class FluxFigurePreset(val id: String, val name: String, val attributes: FluxFigureAttributes, val builtIn: Boolean) {
  init { require(id.isNotBlank() && name.isNotBlank()); require(!builtIn || id.startsWith("builtin.")) }
}

object FluxBuiltInFigurePresets {
  val all: List<FluxFigurePreset> = listOf(
    preset("preserve", "Preserve reference", "preserve the reference's current figure"),
    preset("balanced", "Balanced natural", "balanced natural proportions"),
    preset("athletic", "Athletic balanced", "athletic balanced proportions"),
    preset("soft_curvy", "Soft curvy", "soft curvy proportions with a naturally defined waist and proportionate hips"),
    preset("tall", "Tall proportioned", "tall appearance with naturally longer leg proportions"),
    preset("petite", "Petite proportioned", "petite proportions consistent with the subject's frame"),
  )
  private fun preset(id: String, name: String, build: String) = FluxFigurePreset("builtin.$id", name, FluxFigureAttributes(overallBuild = build), true)
}

/** Persists descriptive configuration only; no URI, path, prompt history, image, or identity fields exist. */
class FluxFigurePresetRepository @Inject constructor(@ApplicationContext context: Context) {
  private val preferences = context.getSharedPreferences("flux_figure_presets", Context.MODE_PRIVATE)
  private val json = Json { ignoreUnknownKeys = true }

  suspend fun userPresets(): List<FluxFigurePreset> = withContext(Dispatchers.IO) { decode() }
  suspend fun save(preset: FluxFigurePreset): FluxFigurePreset = withContext(Dispatchers.IO) {
    require(!preset.builtIn && !preset.id.startsWith("builtin.")) { "Built-in presets are immutable." }
    val updated = decode().filterNot { it.id == preset.id } + preset
    preferences.edit().putString(KEY, json.encodeToString(updated)).commit()
    preset
  }
  suspend fun duplicate(source: FluxFigurePreset, id: String, name: String): FluxFigurePreset =
    save(FluxFigurePreset(id, name.trim(), source.attributes, false))
  suspend fun delete(id: String): Boolean = withContext(Dispatchers.IO) {
    require(!id.startsWith("builtin.")) { "Built-in presets cannot be deleted." }
    val old = decode(); val updated = old.filterNot { it.id == id }
    preferences.edit().putString(KEY, json.encodeToString(updated)).commit() && old.size != updated.size
  }
  private fun decode(): List<FluxFigurePreset> = runCatching {
    json.decodeFromString<List<FluxFigurePreset>>(preferences.getString(KEY, "[]") ?: "[]").filterNot { it.builtIn }
  }.getOrDefault(emptyList())
  private companion object { const val KEY = "descriptive_presets" }
}
