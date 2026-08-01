/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.decoder
import com.google.ai.edge.gallery.customtasks.flux.FluxModelManifest
import com.google.ai.edge.gallery.customtasks.flux.FLUX_REPOSITORY
import com.google.ai.edge.gallery.customtasks.flux.FLUX_MODEL_REVISION
import com.google.ai.edge.gallery.customtasks.flux.runtime.*
import java.io.File
import kotlin.test.*
import kotlinx.coroutines.runBlocking
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
class FluxVaeDecoderTest { @get:Rule val temporary=TemporaryFolder()
 @Test fun `decoder uses exact graph input ordering and output contract`()=runBlocking {val root=temporary.newFolder();File(root,FluxVaeDecoder.GRAPH).writeBytes(byteArrayOf(1));var observed:FloatArray?=null;val runner=object:FluxGraphRunner{override suspend fun run(modelFile:File,inputs:List<FloatArray>):FluxGraphResult{assertEquals("kv_vae.tflite",modelFile.name);assertEquals(1,inputs.size);observed=inputs.single();return FluxGraphResult(modelFile.name,listOf(FloatArray(196608)))}};val input=FluxDecoderInput.checked(FloatArray(32768){it.toFloat()});val output=FluxVaeDecoder(runner).decode(root,FluxModelManifest(1,FLUX_REPOSITORY,FLUX_MODEL_REVISION,"p",listOf("kv_vae.tflite")),input);assertEquals(196608,output.size);assertEquals(32767f,observed!![32767])}
 @Test fun `decoder rejects nonfinite output and wrong output count`()=runBlocking {val root=temporary.newFolder();File(root,"kv_vae.tflite").writeBytes(byteArrayOf(1));val manifest=FluxModelManifest(1,FLUX_REPOSITORY,FLUX_MODEL_REVISION,"p",listOf("kv_vae.tflite"));val input=FluxDecoderInput.checked(FloatArray(32768));for(outputs in listOf(emptyList(),listOf(FloatArray(196608).also{it[0]=Float.NaN}))){val runner=object:FluxGraphRunner{override suspend fun run(modelFile:File,inputs:List<FloatArray>)=FluxGraphResult(modelFile.name,outputs)};assertFails{FluxVaeDecoder(runner).decode(root,manifest,input)}}}
}
