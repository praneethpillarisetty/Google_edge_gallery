/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import java.io.File
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FluxReferencePickerSourceTest {
  @Test fun pickerUsesOpenDocumentPersistableReadOnlyGrantAndKeepsTemporaryGrant() {
    val source = File("src/main/java/com/google/ai/edge/gallery/customtasks/flux/FluxEditorScreen.kt").readText()
    assertTrue(source.contains("ActivityResultContracts.OpenDocument()"))
    assertTrue(source.contains("picker.launch(arrayOf(\"image/*\"))"))
    assertTrue(source.contains("takePersistableUriPermission(it, Intent.FLAG_GRANT_READ_URI_PERMISSION)"))
    assertTrue(source.contains("catch (_: SecurityException)"))
    assertTrue(source.contains("imageUri = selected"))
    assertFalse(source.contains("FLAG_GRANT_WRITE_URI_PERMISSION"))
    assertFalse(source.contains("ActivityResultContracts.GetContent()"))
  }
}
