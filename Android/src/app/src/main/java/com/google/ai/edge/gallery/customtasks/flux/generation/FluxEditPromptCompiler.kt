/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.generation

/** Deterministic, local compiler. Qwen chat wrapping remains the conditioner's responsibility. */
class FluxEditPromptCompiler {
  fun compile(request: FluxSimpleEditRequest): FluxCompiledPrompt {
    val instruction = request.userInstruction.trim()
    require(instruction.isNotEmpty()) { "Describe the requested edit." }
    val intents = FluxPromptIntentDetector.detect(instruction)
    val preserves = buildList {
      if (request.preserveIdentity) add("the same adult person's recognizable facial identity, skin tone, and realistic anatomy")
      if (request.preservePoseAndComposition && FluxPromptIntent.POSE !in intents && FluxPromptIntent.CAMERA !in intents) add("the reference pose and composition")
      if (request.preserveBackground && FluxPromptIntent.BACKGROUND !in intents) add("the reference location and background")
      if (FluxPromptIntent.OUTFIT !in intents) add("the reference outfit")
      if (FluxPromptIntent.HAIR !in intents) add("the reference hairstyle")
      if (FluxPromptIntent.MAKEUP_EXPRESSION !in intents) add("the reference makeup and expression")
    }
    val preservation = if (preserves.isEmpty()) "Preserve realistic anatomy and properties not explicitly changed." else
      "Preserve ${preserves.joinToString(", ")}, and all properties the requested edit does not explicitly change."
    return FluxCompiledPrompt("""Use the reference image as the source image.

$preservation

Requested edit:
$instruction

Apply only the requested changes. Produce one coherent photorealistic image with natural lighting, realistic textures, and anatomically consistent details.""", intents)
  }

  fun compile(request: FluxFigureEditRequest, presetDescription: String? = null): FluxCompiledPrompt {
    val additional = request.additionalInstruction?.trim()?.takeIf(String::isNotEmpty)
    val intents = additional?.let(FluxPromptIntentDetector::detect).orEmpty()
    val conflicts = locks(request).mapNotNull { (intent, name, enabled) ->
      if (enabled && intent in intents) FluxPromptConflict(intent, name) else null
    }
    val description = when (val action = request.figureAction) {
      FluxFigureAction.PreserveCurrent -> null
      is FluxFigureAction.Custom -> action.description.trim().also { require(it.isNotEmpty()) { "Enter a custom figure description." } }
      is FluxFigureAction.ApplyPreset -> (request.figureDescription ?: presetDescription).orEmpty().trim()
        .also { require(it.isNotEmpty()) { "The selected figure preset has no description." } }
    }
    val enabledLocks = locks(request).filter { it.third }.map { it.second }
    val actionClause = if (request.figureAction is FluxFigureAction.PreserveCurrent)
      "Preserve the reference's current overall silhouette and body proportions."
    else "Modify only the subject's body proportions according to this figure profile:\n$description"
    val garment = if (request.preserveOutfit) "\n\nPreserve garment identity, design, fabric, colors and accessories while allowing only the minimum fit deformation required by the requested figure adjustment." else ""
    val optional = additional?.let { "\n\nAdditional instruction:\n$it" }.orEmpty()
    val prompt = """Use the reference image as the primary visual source.

Preserve: ${enabledLocks.joinToString(", ")}.

$actionClause

Keep the result anatomically realistic and naturally compatible with the existing pose and clothing.$garment

Maintain a clear naturally rendered face with the reference head position, head orientation, expression, viewing angle, visibility, and face lighting. Do not obscure or crop the face.$optional

Do not redesign or relocate any locked property."""
    return FluxCompiledPrompt(prompt, intents, conflicts)
  }

  private fun locks(r: FluxFigureEditRequest) = listOf(
    Triple(FluxPromptIntent.OUTFIT, "outfit", r.preserveOutfit),
    Triple(FluxPromptIntent.POSE, "pose and hand placement", r.preservePose),
    Triple(FluxPromptIntent.BACKGROUND, "location and background", r.preserveBackground),
    Triple(FluxPromptIntent.CAMERA, "framing and camera", r.preserveCamera),
    Triple(FluxPromptIntent.LIGHTING, "lighting and shadows", r.preserveLighting),
    Triple(FluxPromptIntent.HAIR, "hairstyle", r.preserveHair),
    Triple(FluxPromptIntent.MAKEUP_EXPRESSION, "makeup and expression", r.preserveMakeupAndExpression),
    Triple(FluxPromptIntent.OUTFIT, "accessories", r.preserveAccessories),
  )
}

object FluxPromptIntentDetector {
  private val terms = mapOf(
    FluxPromptIntent.OUTFIT to listOf("outfit", "clothing", "clothes", "dress", "gown", "suit", "shirt", "top", "pants", "skirt", "coat", "jacket", "saree", "sari", "lehenga", "uniform", "costume", "wear", "wearing", "accessories"),
    FluxPromptIntent.BACKGROUND to listOf("background", "location", "setting", "scene", "beach", "mountain", "city", "street", "room", "studio", "forest", "park", "office", "indoor", "outdoor", "move to", "place in"),
    FluxPromptIntent.POSE to listOf("pose", "standing", "sitting", "walking", "running", "kneeling", "leaning", "arms", "hands", "turn", "facing", "posture"),
    FluxPromptIntent.CAMERA to listOf("camera", "angle", "framing", "crop", "close-up", "waist-up", "full-body", "portrait", "wide shot", "perspective", "zoom", "viewpoint"),
    FluxPromptIntent.LIGHTING to listOf("lighting", "sunlight", "sunset", "night", "studio light", "soft light", "dramatic light", "shadow", "shadows"),
    FluxPromptIntent.HAIR to listOf("hair", "hairstyle", "haircut", "ponytail", "bun", "braid", "curls", "straight hair", "hair color"),
    FluxPromptIntent.MAKEUP_EXPRESSION to listOf("makeup", "lipstick", "eyeliner", "smile", "expression", "laughing", "serious", "surprised"),
    FluxPromptIntent.BODY to listOf("body", "figure", "build", "proportions", "shoulders", "torso", "waist", "hips", "legs", "height", "silhouette", "athletic", "curvy", "petite", "tall"),
  )
  fun detect(text: String): Set<FluxPromptIntent> = terms.filterValues { words -> words.any { phrase ->
    Regex("(?i)(?<![\\p{L}\\p{N}_])${Regex.escape(phrase)}(?![\\p{L}\\p{N}_])").containsMatchIn(text)
  } }.keys
}
