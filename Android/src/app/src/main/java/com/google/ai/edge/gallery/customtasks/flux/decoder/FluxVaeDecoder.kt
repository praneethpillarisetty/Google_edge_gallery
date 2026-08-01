/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.decoder
import android.graphics.Bitmap
import com.google.ai.edge.gallery.customtasks.flux.FluxModelManifest
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxGraphRunner
import java.io.File
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.ensureActive
internal class FluxVaeDecoder(private val runner:FluxGraphRunner) {
  suspend fun decode(rootFile:File,manifest:FluxModelManifest,input:FluxDecoderInput):FloatArray {
    coroutineContext.ensureActive();require(GRAPH in manifest.files){"Decoder graph is missing."};val root=rootFile.canonicalFile;val graph=File(root,GRAPH).canonicalFile
    require(graph.parentFile==root&&graph.isFile){"Decoder graph is missing."};require(input.shape==INPUT_SHAPE&&input.size==32_768)
    val values=input.copyValues();require(values.all(Float::isFinite));coroutineContext.ensureActive();val result=runner.run(graph,listOf(values));require(result.graph==GRAPH&&result.outputs.size==1)
    val output=result.outputs.single();require(output.size==196_608&&output.all(Float::isFinite)){"Decoder output must be finite [1,3,256,256]."};coroutineContext.ensureActive();return output.copyOf()
  }
  companion object{const val GRAPH="kv_vae.tflite";val INPUT_SHAPE=listOf(1,32,32,32);val OUTPUT_SHAPE=listOf(1,3,256,256)}
}
internal object FluxDecodedBitmapConverter {
  fun pixels(planar:FloatArray,check:()->Unit={}):IntArray {require(planar.size==196_608);val pixels=IntArray(65_536);for(i in pixels.indices){if(i and 1023==0)check();val r=byte(planar[i]);val g=byte(planar[65_536+i]);val b=byte(planar[131_072+i]);pixels[i]=(0xff shl 24)or(r shl 16)or(g shl 8)or b};check();return pixels}
  fun bitmap(planar:FloatArray,check:()->Unit={}):Bitmap=Bitmap.createBitmap(pixels(planar,check),256,256,Bitmap.Config.ARGB_8888)
  internal fun byte(value:Float):Int{require(value.isFinite());return(((value.coerceIn(-1f,1f)+1f)*127.5f).toInt()).coerceIn(0,255)}
}
