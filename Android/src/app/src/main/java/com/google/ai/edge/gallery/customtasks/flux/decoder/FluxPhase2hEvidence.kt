/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.decoder

import android.content.res.AssetManager
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

class FluxPhase2hEvidenceException(message: String, cause: Throwable? = null) : IllegalArgumentException(message, cause)
internal class FluxPhase2hEvidence(unpack: IntArray, unpatch: IntArray) {
  private val unpackValues = unpack.copyOf(); private val unpatchValues = unpatch.copyOf()
  fun unpackAt(index: Int) = unpackValues[index]; fun unpatchAt(index: Int) = unpatchValues[index]
}
internal object FluxPhase2hEvidenceParser {
  const val UNPACK_SHA256 = "909fbd19ae0075502700869ea294e50f2a5cd55376ace013dd19b31d4b862ce9"
  const val UNPATCH_SHA256 = "3ba8f27ee20eb995c5cc6752d02a2a7dd237213afd7059d9e403ad966b334fa7"
  private const val ELEMENTS = 32_768
  private val json = Json { ignoreUnknownKeys = false }
  fun parse(open: (String) -> InputStream): FluxPhase2hEvidence = try {
    val metadata = open("flux_phase2h_constants.json").use { json.decodeFromString<Metadata>(it.readBytes().decodeToString()) }
    validateMetadata(metadata)
    FluxPhase2hEvidence(artifact(open,"unpack_perm.bin",UNPACK_SHA256), artifact(open,"unpatch_perm.bin",UNPATCH_SHA256))
  } catch (e: FluxPhase2hEvidenceException) { throw e } catch (e: Exception) { throw FluxPhase2hEvidenceException("Phase 2H evidence validation failed.", e) }
  internal fun decodeAndValidate(bytes: ByteArray, name: String): IntArray {
    if (bytes.size != 131_072) fail("$name must contain exactly 131072 bytes.")
    val b=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN); val values=IntArray(ELEMENTS){b.int}; val seen=BooleanArray(ELEMENTS)
    var min=Int.MAX_VALUE; var max=Int.MIN_VALUE
    values.forEach { if(it !in seen.indices) fail("$name contains an out-of-range index."); if(seen[it]) fail("$name contains a duplicate index."); seen[it]=true; min=minOf(min,it); max=maxOf(max,it) }
    if(min!=0 || max!=ELEMENTS-1 || !seen.all{it}) fail("$name is not a complete permutation."); return values
  }
  private fun artifact(open:(String)->InputStream,name:String,hash:String):IntArray { val bytes=open(name).use{it.readBytes()}; if(sha(bytes)!=hash) fail("$name failed SHA-256 validation."); return decodeAndValidate(bytes,name) }
  private fun sha(bytes:ByteArray)=MessageDigest.getInstance("SHA-256").digest(bytes).joinToString(""){"%02x".format(it)}
  private fun validateMetadata(m:Metadata) {
    if(m.schemaVersion!=1 || m.baseModelRepository!="black-forest-labs/FLUX.2-klein-4B" || m.immutableBaseModelRevision!="e7b7dc27f91deacad38e78976d1f2b499d76a294" || m.companionRepository!="google-ai-edge/litert-samples" || m.immutableCompanionRevision!="f48a89e4f29a74ab51f29c311ac7a0e5e479d225" || m.generatorScriptPath!="compiled_model_api/text_to_image/flux2_klein_kotlin_gpu/conversion/gen_prep_klein.py" || m.originalGeneratorScriptSha256!="1f2b3d902d4f36281e61447d86331e08bd1c61f20f9034896817c086ae1ab61d" || m.sourceLatentTokenShape!=listOf(1,256,128) || m.unpackedShape!=listOf(1,128,16,16) || m.decoderInputShape!=listOf(1,32,32,32) || m.decoderOutputShape!=listOf(1,3,256,256)) fail("Metadata provenance or tensor contract is invalid.")
    if(m.pythonVersion!="3.12.13" || m.pytorchVersion!="2.9.0+cpu" ||
      m.extractionLibraries!=Libraries("1.14.0","0.39.0","1.25.1","2.0.2","0.8.0","5.14.1") ||
      m.purpose!="FLUX Phase 2H decoder-tail permutation evidence" ||
      m.pixelConversionPolicy!=PixelPolicy(255,listOf(-1f,1f),"(clamp(value, -1, 1) + 1) * 127.5","NCHW planar RGB FP32",listOf(-1f,1f),"truncate toward zero","ARGB_8888","Pinned Android companion implementation")) fail("Metadata environment or pixel policy is invalid.")
    if(m.artifacts.keys!=setOf("unpack_perm.bin","unpatch_perm.bin")) fail("Metadata contains an unexpected artifact set.")
    validateArtifact(m.artifacts["unpack_perm.bin"],UNPACK_SHA256); validateArtifact(m.artifacts["unpatch_perm.bin"],UNPATCH_SHA256)
  }
  private fun validateArtifact(a:Artifact?,hash:String) { if(a==null || a.byteCount!=131_072 || a.elementCount!=ELEMENTS || a.byteOrder!="little-endian" || a.dtype!="int32" || a.numpyDtype!="<i4" || a.sha256!=hash || !a.isBijection || a.minimum!=0 || a.maximum!=32767 || !a.sha256.matches(Regex("[0-9a-f]{64}"))) fail("Artifact metadata is invalid.") }
  private fun fail(message:String):Nothing=throw FluxPhase2hEvidenceException(message)
  @Serializable private data class Artifact(val byteCount:Int,val byteOrder:String,val dtype:String,val elementCount:Int,val isBijection:Boolean,val maximum:Int,val minimum:Int,val numpyDtype:String,val sha256:String)
  @Serializable private data class Libraries(val accelerate:String,val diffusers:String,val huggingface_hub:String,val numpy:String,val safetensors:String,val transformers:String)
  @Serializable private data class PixelPolicy(val alpha:Int,val clampRange:List<Float>,val formula:String,val inputLayout:String,val inputNominalRange:List<Float>,val integerConversion:String,val outputBitmap:String,val selectedAuthority:String)
  @Serializable private data class Metadata(val artifacts:Map<String,Artifact>,val baseModelRepository:String,val companionRepository:String,val decoderInputShape:List<Int>,val decoderOutputShape:List<Int>,val extractionLibraries:Libraries,val generatedAt:String,val generatorScriptPath:String,val immutableBaseModelRevision:String,val immutableCompanionRevision:String,val originalGeneratorScriptSha256:String,val patchedGeneratorScriptSha256:String,val pixelConversionPolicy:PixelPolicy,val provenanceStatement:String,val purpose:String,val pythonVersion:String,val pytorchVersion:String,val schemaVersion:Int,val sourceLatentTokenShape:List<Int>,val unpackedShape:List<Int>)
}
internal class FluxPhase2hEvidenceLoader(private val assets:AssetManager) {
  @Volatile private var cached:FluxPhase2hEvidence?=null
  fun load():FluxPhase2hEvidence=cached?:synchronized(this){cached?:FluxPhase2hEvidenceParser.parse{assets.open("flux/phase2h/$it")}.also{cached=it}}
}
