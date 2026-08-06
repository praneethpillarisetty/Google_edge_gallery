/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.generation

/** Deterministic local compiler. FluxPromptConditioner remains the sole Qwen-wrapper owner. */
class FluxEditPromptCompiler(
  private val tokenCounter: FluxBodyTokenCounter = FluxUtf8UpperBoundTokenCounter,
  private val maximumBodyTokens: Int = 500,
) {
  fun compile(request: FluxSimpleEditRequest): FluxCompiledPrompt {
    val instruction = request.userInstruction.trim()
    require(instruction.isNotEmpty()) { "Describe the requested edit." }
    val intents = FluxPromptIntentDetector.detect(instruction)
    val framing = FluxPromptIntentDetector.framing(instruction, request.visualContext.framing)
    val pose = FluxPromptIntentDetector.pose(instruction)
    val sections = linkedMapOf<String, String>()
    sections["reference"] = "Use the reference image as the primary visual source."
    sections["requested_edit"] = "Requested edit:\n$instruction"
    sections["anatomy_invariant"] = ANATOMY_INVARIANT
    conditionalRules(instruction, framing, pose, request.visualContext).forEach { (name, clause) -> sections[name] = clause }
    preservation(intents, request).takeIf(String::isNotEmpty)?.let { sections["preservation"] = it }
    sceneClause(intents, request.preserveBackground, request.preservePoseAndComposition)?.let { sections["scene_perspective"] = it }
    sections["realism"] = realism(request.realismProfile)
    sections["scope"] = "Apply only the requested changes and preserve every property not explicitly changed."
    return result(sections, setOf("requested_edit"), intents, framing, pose)
  }

  fun compile(request: FluxFigureEditRequest, presetDescription: String? = null): FluxCompiledPrompt {
    val additional = request.additionalInstruction?.trim()?.takeIf(String::isNotEmpty)
    val intentText = additional.orEmpty()
    val intents = FluxPromptIntentDetector.detect(intentText)
    val framing = FluxPromptIntentDetector.framing(intentText, request.visualContext.framing)
    val pose = FluxPromptIntentDetector.pose(intentText)
    val conflicts = locks(request).mapNotNull { (intent, name, enabled) ->
      if (enabled && intent in intents) FluxPromptConflict(intent, name) else null
    }
    val sections = linkedMapOf<String, String>()
    sections["reference"] = "Use the reference image as the primary visual source."
    sections["figure_action"] = figureAction(request, presetDescription)
    sections["anatomy_invariant"] = ANATOMY_INVARIANT
    conditionalRules(intentText, framing, pose, request.visualContext).forEach { (name, clause) -> sections[name] = clause }
    enabledPreservation(request).takeIf(String::isNotEmpty)?.let { sections["preservation"] = it }
    sceneClause(intents, request.preserveBackground, request.preserveCamera)?.let { sections["scene_perspective"] = it }
    if (request.preserveOutfit) sections["clothing_body"] = CLOTHING_BODY
    sections["face_coherence"] = FACE_COHERENCE
    if (request.visualContext.faceTurned || FluxPromptIntentDetector.turnedFace(intentText)) sections["turned_face"] = TURNED_FACE
    if (request.visualContext.faceOccluded || FluxPromptIntentDetector.occludedFace(intentText)) sections["face_occlusion"] = FACE_OCCLUSION
    sections["realism"] = realism(request.realismProfile)
    additional?.let { sections["additional_instruction"] = "Additional instruction:\n$it" }
    sections["scope"] = "Do not redesign or relocate any locked property; make only the selected figure adjustment and explicit additional changes."
    return result(sections, setOf("figure_action", "additional_instruction"), intents, framing, pose, conflicts)
  }

  private fun result(sections: LinkedHashMap<String, String>, literalSections: Set<String>, intents: Set<FluxPromptIntent>, framing: FluxFramingCategory, pose: FluxPoseCategory, conflicts: List<FluxPromptConflict> = emptyList()): FluxCompiledPrompt {
    val plan = FluxPromptBudgetPlanner(tokenCounter, maximumBodyTokens).plan(sections, literalSections.filterTo(linkedSetOf()) { it in sections })
    return FluxCompiledPrompt(plan.text, intents, conflicts, plan.includedSectionNames, framing, pose,
      plan.bodyTokenCount, plan.maximumBodyTokens, plan.truncationOccurred, plan.omittedGeneratedSectionNames)
  }

  private fun preservation(intents: Set<FluxPromptIntent>, r: FluxSimpleEditRequest): String = buildList {
    if (r.preserveIdentity) add("Preserve the same adult person's recognizable facial identity and skin tone.")
    if (r.preservePoseAndComposition && FluxPromptIntent.POSE !in intents && FluxPromptIntent.CAMERA !in intents) add("Preserve the reference pose, framing and composition.")
    if (r.preserveBackground && FluxPromptIntent.BACKGROUND !in intents) add("Preserve the reference location and background.")
    if (FluxPromptIntent.OUTFIT !in intents) add("Preserve the reference outfit.")
    if (FluxPromptIntent.HAIR !in intents) add("Preserve the reference hairstyle.")
    if (FluxPromptIntent.MAKEUP_EXPRESSION !in intents) add("Preserve the reference makeup and expression.")
  }.joinToString(" ")

  private fun figureAction(r: FluxFigureEditRequest, preset: String?): String = when (val action = r.figureAction) {
    FluxFigureAction.PreserveCurrent -> PRESERVE_CURRENT
    is FluxFigureAction.Custom -> action.description.trim().also { require(it.isNotEmpty()) { "Enter a custom figure description." } }.let { "$FIGURE_CHANGE\nFigure profile: $it" }
    is FluxFigureAction.ApplyPreset -> (r.figureDescription ?: preset).orEmpty().trim().also { require(it.isNotEmpty()) { "The selected figure preset has no description." } }.let { "$FIGURE_CHANGE\nFigure profile: $it" }
  }

  private fun enabledPreservation(r: FluxFigureEditRequest) = buildList {
    if (r.preserveOutfit) add("outfit identity and styling")
    if (r.preservePose) add("pose, hand placement and body orientation")
    if (r.preserveBackground) add("location, background and scene layout")
    if (r.preserveCamera) add("framing, crop, camera angle and perspective")
    if (r.preserveLighting) add("lighting and shadows")
    if (r.preserveHair) add("hairstyle")
    if (r.preserveMakeupAndExpression) add("makeup and expression")
    if (r.preserveAccessories) add("accessories")
  }.takeIf { it.isNotEmpty() }?.joinToString(prefix = "Preserve: ", postfix = ".") ?: ""

  private fun conditionalRules(text: String, framing: FluxFramingCategory, pose: FluxPoseCategory, context: FluxEditVisualContext): LinkedHashMap<String, String> = linkedMapOf<String, String>().apply {
    when (framing) {
      FluxFramingCategory.FULL_BODY, FluxFramingCategory.THREE_QUARTER -> put("framing", FULL_BODY)
      FluxFramingCategory.WAIST_UP -> put("framing", WAIST_UP)
      FluxFramingCategory.CLOSE_UP -> put("framing", CLOSE_UP)
      FluxFramingCategory.UNSPECIFIED -> Unit
    }
    if (context.handsVisible || FluxPromptIntentDetector.handsRelevant(text)) put("hands", HANDS)
    if (context.limbsOverlap || FluxPromptIntentDetector.overlap(text)) put("limb_occlusion", LIMB_OVERLAP)
    if (pose != FluxPoseCategory.NONE) {
      put("pose_coherence", POSE_COHERENCE)
      poseSpecific(pose)?.let { put("pose_specific", it) }
    }
  }

  private fun poseSpecific(pose: FluxPoseCategory) = when (pose) {
    FluxPoseCategory.SITTING -> SITTING
    FluxPoseCategory.STANDING -> STANDING
    FluxPoseCategory.GAIT -> GAIT
    FluxPoseCategory.RAISED_ARMS -> RAISED_ARMS
    FluxPoseCategory.LEANING -> LEANING
    FluxPoseCategory.NONE, FluxPoseCategory.GENERIC -> null
  }

  private fun sceneClause(intents: Set<FluxPromptIntent>, preserveBackground: Boolean, preserveCamera: Boolean): String? = when {
    FluxPromptIntent.BACKGROUND in intents -> NEW_LOCATION
    preserveBackground && preserveCamera -> PRESERVED_SCENE
    else -> null
  }

  private fun realism(profile: FluxRealismProfile) = when (profile) {
    FluxRealismProfile.NATURAL_PHOTO -> NATURAL_PHOTO
    FluxRealismProfile.EDITORIAL_PHOTO -> EDITORIAL_PHOTO
    FluxRealismProfile.CINEMATIC_PHOTO -> CINEMATIC_PHOTO
  }

  private fun locks(r: FluxFigureEditRequest) = listOf(
    Triple(FluxPromptIntent.OUTFIT, "outfit", r.preserveOutfit), Triple(FluxPromptIntent.POSE, "pose and hand placement", r.preservePose),
    Triple(FluxPromptIntent.BACKGROUND, "location and background", r.preserveBackground), Triple(FluxPromptIntent.CAMERA, "framing and camera", r.preserveCamera),
    Triple(FluxPromptIntent.LIGHTING, "lighting and shadows", r.preserveLighting), Triple(FluxPromptIntent.HAIR, "hairstyle", r.preserveHair),
    Triple(FluxPromptIntent.MAKEUP_EXPRESSION, "makeup and expression", r.preserveMakeupAndExpression), Triple(FluxPromptIntent.ACCESSORIES, "accessories", r.preserveAccessories),
  )

  companion object {
    const val ANATOMY_INVARIANT = "Maintain one complete and anatomically coherent adult human figure. Keep the head, torso, shoulders, arms, hands, hips, legs and feet connected through physically plausible joints, with consistent left/right placement, natural balance and correct perspective. Preserve coherent anatomy for partially occluded body parts."
    const val FULL_BODY = "Keep both visible legs naturally connected to the hips, both visible arms naturally connected to the shoulders, and maintain plausible weight distribution and balance."
    const val WAIST_UP = "Preserve the visible torso, shoulders, arms and hands with natural joint placement. Do not invent body parts outside the selected crop."
    const val CLOSE_UP = "Preserve coherent head, neck and shoulder anatomy. Keep facial features, ears, hairline and visible shoulders aligned with the viewing angle."
    const val HANDS = "Keep each visible hand connected to its corresponding arm, with a natural wrist orientation and a plausible number and arrangement of visible fingers. Respect occlusion when fingers or hands are partially hidden."
    const val LIMB_OVERLAP = "Preserve correct front-to-back ordering and occlusion where limbs overlap."
    const val POSE_COHERENCE = "Apply the requested pose as one physically achievable human pose. Maintain natural joint ranges, stable balance, coherent weight distribution, consistent left/right anatomy and correct contact between the body, clothing and environment."
    const val SITTING = "Align the hips and legs naturally with the seat and preserve believable contact, compression and balance."
    const val STANDING = "Maintain stable foot placement, natural knee alignment and balanced body weight."
    const val GAIT = "Maintain a physically plausible gait, coordinated opposing arm and leg movement, stable body balance and believable ground contact."
    const val RAISED_ARMS = "Maintain natural shoulder rotation, elbow direction and wrist orientation, with each arm connected coherently to its shoulder."
    const val LEANING = "Maintain believable support, center of gravity and contact with the supporting surface."
    const val FIGURE_CHANGE = "Apply the requested figure adjustment consistently across the shoulders, torso, waist, hips and visible limbs while preserving one coherent skeletal structure. Keep the result compatible with the reference pose, camera perspective and natural human anatomy."
    const val PRESERVE_CURRENT = "Preserve the reference figure's current visible silhouette, relative body proportions and limb lengths."
    const val CLOTHING_BODY = "Preserve the garment's identity, design, fabric, color, pattern, accessories and layering. Fit the existing clothing naturally over the resulting body, with physically plausible draping, seams, folds, tension, coverage and occlusion. Permit only the minimum garment deformation required by the figure or pose adjustment. Keep limbs distinct through garment openings, fabric following the body and pose, and accessories attached at physically plausible locations."
    const val FACE_COHERENCE = "Maintain one clear, naturally structured adult face with coherent facial geometry, aligned eyes, a natural nose and mouth, consistent skin illumination and a clean jaw/neck connection. Preserve the reference head position, expression, viewing angle, facial visibility and lighting so the result remains suitable for a later external identity-replacement step."
    const val TURNED_FACE = "Keep all visible facial features consistent with the head angle and perspective."
    const val FACE_OCCLUSION = "Respect the existing facial occlusion instead of inventing additional visible features."
    const val PRESERVED_SCENE = "Preserve the reference camera viewpoint, subject scale, horizon, ground plane, lighting direction and scene perspective. Keep the subject physically grounded in the environment with coherent shadows and contact."
    const val NEW_LOCATION = "Place the subject naturally in the requested environment while preserving consistent subject scale, perspective, ground contact, lighting and shadows."
    const val NATURAL_PHOTO = "Render one coherent, anatomically realistic adult human subject as a natural photograph, with realistic skin texture, physically plausible lighting, consistent perspective, natural proportions and visually coherent details."
    const val EDITORIAL_PHOTO = "Render one coherent, anatomically realistic adult human subject as a polished editorial photograph, with realistic skin texture, controlled photographic lighting, consistent perspective, natural proportions and detailed but believable styling."
    const val CINEMATIC_PHOTO = "Render one coherent, anatomically realistic adult human subject as a cinematic photograph, with physically plausible directional lighting, consistent perspective, realistic skin texture, natural proportions and coherent environmental detail."
  }
}

object FluxPromptIntentDetector {
  private val terms = mapOf(
    FluxPromptIntent.OUTFIT to listOf("outfit", "clothing", "clothes", "dress", "gown", "suit", "shirt", "top", "pants", "skirt", "coat", "jacket", "saree", "sari", "lehenga", "uniform", "costume", "wear", "wearing"),
    FluxPromptIntent.ACCESSORIES to listOf("accessory", "accessories", "jewelry", "jewellery", "necklace", "earrings", "bracelet", "watch", "belt", "bag", "handbag", "glasses", "sunglasses", "hat"),
    FluxPromptIntent.BACKGROUND to listOf("background", "location", "setting", "scene", "beach", "mountain", "city", "street", "room", "studio", "forest", "park", "office", "indoor", "outdoor", "move to", "place in"),
    FluxPromptIntent.POSE to listOf("pose", "standing", "sitting", "walking", "running", "kneeling", "leaning", "arms", "hands", "holding", "touching", "waving", "raised arms", "turn", "facing", "posture"),
    FluxPromptIntent.CAMERA to listOf("camera", "angle", "framing", "crop", "close-up", "waist-up", "full-body", "three-quarter", "portrait", "wide shot", "perspective", "zoom", "viewpoint"),
    FluxPromptIntent.LIGHTING to listOf("lighting", "sunlight", "sunset", "night", "studio light", "soft light", "dramatic light", "shadow", "shadows"),
    FluxPromptIntent.HAIR to listOf("hair", "hairstyle", "haircut", "ponytail", "bun", "braid", "curls", "straight hair", "hair color"),
    FluxPromptIntent.MAKEUP_EXPRESSION to listOf("makeup", "lipstick", "eyeliner", "smile", "expression", "laughing", "serious", "surprised"),
    FluxPromptIntent.BODY to listOf("body", "figure", "build", "proportions", "shoulders", "torso", "waist", "hips", "legs", "height", "silhouette", "athletic", "curvy", "petite", "tall"),
  )
  fun detect(text: String): Set<FluxPromptIntent> = terms.filterValues { words -> words.any { contains(text, it) } }.keys
  fun framing(text: String, explicit: FluxFramingCategory) = if (explicit != FluxFramingCategory.UNSPECIFIED) explicit else when {
    any(text, "close-up", "close up") -> FluxFramingCategory.CLOSE_UP
    any(text, "waist-up", "waist up") -> FluxFramingCategory.WAIST_UP
    any(text, "three-quarter", "three quarter") -> FluxFramingCategory.THREE_QUARTER
    any(text, "full-body", "full body", "wide shot") -> FluxFramingCategory.FULL_BODY
    else -> FluxFramingCategory.UNSPECIFIED
  }
  fun pose(text: String) = when {
    any(text, "sitting", "sit", "seated") -> FluxPoseCategory.SITTING
    any(text, "walking", "running", "walk", "run") -> FluxPoseCategory.GAIT
    any(text, "raised arms", "raise arms", "arms raised", "waving") -> FluxPoseCategory.RAISED_ARMS
    any(text, "leaning", "lean") -> FluxPoseCategory.LEANING
    any(text, "standing", "stand") -> FluxPoseCategory.STANDING
    any(text, "pose", "posture", "kneeling", "turn", "facing") -> FluxPoseCategory.GENERIC
    else -> FluxPoseCategory.NONE
  }
  fun handsRelevant(text: String) = any(text, "hand", "hands", "arm", "arms", "holding", "touching", "waving", "raised arms")
  fun overlap(text: String) = any(text, "crossed arms", "crossing limbs", "limbs cross", "overlapping limbs", "arms overlap", "legs overlap")
  fun turnedFace(text: String) = any(text, "face turned", "turned face", "profile view", "three-quarter face")
  fun occludedFace(text: String) = any(text, "face occluded", "partially hidden face", "face partially hidden", "covered face")
  private fun any(text: String, vararg phrases: String) = phrases.any { contains(text, it) }
  private fun contains(text: String, phrase: String) = Regex("(?i)(?<![\\p{L}\\p{N}_])${Regex.escape(phrase)}(?![\\p{L}\\p{N}_])").containsMatchIn(text)
}
