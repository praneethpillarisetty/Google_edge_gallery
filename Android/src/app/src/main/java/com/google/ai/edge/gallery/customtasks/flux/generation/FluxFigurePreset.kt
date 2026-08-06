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
    FluxFigurePreset("builtin.preserve", "Preserve reference", FluxFigureAttributes(), true),
    preset("balanced", "Balanced natural", "balanced natural build", "shoulders balanced with the frame", "proportionate torso", "gently defined waist", "hips balanced with the shoulders", "even natural leg proportions", "average height impression"),
    preset("athletic", "Athletic balanced", "naturally athletic build", "moderately broad shoulders", "firm proportionate torso", "subtly defined waist", "hips balanced with the torso", "strong proportionate legs", "upright height impression"),
    preset("soft_curvy", "Soft curvy", "softly rounded build with a visibly changed silhouette", "gently rounded shoulders balanced with the hips", "slightly fuller and softly contoured torso", "clearly defined but natural waist transition", "visibly fuller hips in coherent proportion to the waist and torso", "visible upper legs slightly fuller and balanced with the hips", "natural height impression with unchanged anatomical scale"),
    preset("tall", "Tall proportioned", "slender proportionate build", "shoulders balanced with the frame", "naturally elongated torso", "gently defined waist", "hips proportionate to the shoulders", "visibly longer balanced legs", "tall height impression"),
    preset("petite", "Petite proportioned", "compact proportionate build", "narrow balanced shoulders", "compact natural torso", "gently defined waist", "hips balanced with the compact frame", "shorter proportionate legs", "petite height impression"),
  )
  fun byId(id: String): FluxFigurePreset? = all.singleOrNull { it.id == id }
  private fun preset(id: String, name: String, overallBuild: String, shoulders: String, torso: String, waist: String, hips: String, legs: String, heightImpression: String) =
    FluxFigurePreset("builtin.$id", name, FluxFigureAttributes(overallBuild, shoulders, torso, waist, hips, legs, heightImpression), true)
}

data class FluxFigurePresetSelection(val presetId: String, val presetName: String, val attributes: FluxFigureAttributes, val action: FluxFigureAction)

/** One immutable value prevents navigation from retaining fields or an action from the prior preset. */
fun selectBuiltInFigurePreset(preset: FluxFigurePreset): FluxFigurePresetSelection {
  require(preset.builtIn)
  return FluxFigurePresetSelection(preset.id, preset.name, preset.attributes,
    if (preset.id == "builtin.preserve") FluxFigureAction.PreserveCurrent else FluxFigureAction.ApplyPreset(preset.id))
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
