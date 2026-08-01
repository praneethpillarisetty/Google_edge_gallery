/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.decoder
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceConstants
import java.io.ByteArrayInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import kotlin.test.*
import org.junit.Test
class FluxPhase2hTest {
 @Test fun `committed evidence exact hashes sizes endian extrema and bijections`() { val d=directory();listOf("unpack_perm.bin" to FluxPhase2hEvidenceParser.UNPACK_SHA256,"unpatch_perm.bin" to FluxPhase2hEvidenceParser.UNPATCH_SHA256).forEach{(n,h)->val b=Files.readAllBytes(d.resolve(n));assertEquals(131_072,b.size);assertEquals(h,sha(b));val v=FluxPhase2hEvidenceParser.decodeAndValidate(b,n);assertEquals(0,v.minOrNull());assertEquals(32767,v.maxOrNull());assertEquals(32768,v.toSet().size)};FluxPhase2hEvidenceParser.parse{ByteArrayInputStream(Files.readAllBytes(d.resolve(it)))} }
 @Test fun `strict metadata rejects unknown missing mutable and wrong generator`() {
  val d=directory();val j=Files.readString(d.resolve("flux_phase2h_constants.json"))
  fun reject(s:String) { assertFailsWith<FluxPhase2hEvidenceException> {
   FluxPhase2hEvidenceParser.parse { n -> ByteArrayInputStream(if(n.endsWith("json")) s.encodeToByteArray() else Files.readAllBytes(d.resolve(n))) }
  } }
  reject(j.replaceFirst("{","{\"unknown\":1,"));reject(j.replaceFirst("\"byteCount\": 131072","\"unknown\":1,\"byteCount\": 131072"));reject(j.replace(Regex("\\s*\"schemaVersion\": 1,?"),""));reject(j.replace("e7b7dc27f91deacad38e78976d1f2b499d76a294","main"));reject(j.replace("1f2b3d902d4f36281e61447d86331e08bd1c61f20f9034896817c086ae1ab61d","${"0".repeat(64)}"))
 }
 @Test fun `duplicate missing and out of range permutation values reject`() {val b=identityBytes();assertContentEquals(intArrayOf(0,1,2,3),FluxPhase2hEvidenceParser.decodeAndValidate(b,"x").copyOf(4));val duplicate=b.copyOf();ByteBuffer.wrap(duplicate).order(ByteOrder.LITTLE_ENDIAN).putInt(4,0);assertFails{FluxPhase2hEvidenceParser.decodeAndValidate(duplicate,"x")};val outside=b.copyOf();ByteBuffer.wrap(outside).order(ByteOrder.LITTLE_ENDIAN).putInt(0,32768);assertFails{FluxPhase2hEvidenceParser.decodeAndValidate(outside,"x")}}
 @Test fun `tail performs unpack exact channel inverse and unpatch without aliasing`() {val unpack=IntArray(32768){32767-it};val unpatch=IntArray(32768){(it+1)%32768};val mean=FloatArray(128){it.toFloat()};val std=FloatArray(128){2f};val input=FloatArray(32768){it.toFloat()};val original=input.copyOf();val result=FluxDecoderTail(FluxPhase2hEvidence(unpack,unpatch),FluxReferenceConstants(mean,std,IntArray(32768),"r")).prepare(input).first;assertEquals(32766f*2f,result.copyValues()[0]);assertEquals(32511f*2f+1f,result.copyValues()[255]);assertEquals(listOf(1,32,32,32),result.shape);assertContentEquals(original,input);val copy=result.copyValues();copy[0]=-1f;assertNotEquals(-1f,result.copyValues()[0])}
 @Test fun `tail rejects malformed nonfinite and cooperatively cancels`() {val p=IntArray(32768){it};val tail=FluxDecoderTail(FluxPhase2hEvidence(p,p),FluxReferenceConstants(FloatArray(128),FloatArray(128){1f},p,"r"));assertFails{tail.prepare(FloatArray(1))};assertFails{tail.prepare(FloatArray(32768).also{it[1]=Float.NaN})};var checks=0;assertFails{tail.prepare(FloatArray(32768)){if(++checks==10)error("cancel")}};assertTrue(checks>=10)}
 @Test fun `Android planar RGB conversion clamps truncates and packs opaque row major`() {assertEquals(0,FluxDecodedBitmapConverter.byte(-2f));assertEquals(255,FluxDecodedBitmapConverter.byte(2f));assertEquals(0,FluxDecodedBitmapConverter.byte(-1f));assertEquals(255,FluxDecodedBitmapConverter.byte(1f));assertEquals(127,FluxDecodedBitmapConverter.byte(0f));assertFails{FluxDecodedBitmapConverter.byte(Float.NaN)};val v=FloatArray(196608){-1f};v[0]=1f;v[65536+1]=1f;v[131072+2]=1f;val p=FluxDecodedBitmapConverter.pixels(v);assertEquals(0xffff0000.toInt(),p[0]);assertEquals(0xff00ff00.toInt(),p[1]);assertEquals(0xff0000ff.toInt(),p[2]);assertTrue(p.all{it ushr 24==255})}
 private fun identityBytes()=ByteBuffer.allocate(131072).order(ByteOrder.LITTLE_ENDIAN).also{b->repeat(32768){b.putInt(it)}}.array()
 private fun sha(b:ByteArray)=MessageDigest.getInstance("SHA-256").digest(b).joinToString(""){"%02x".format(it)}
 private fun directory():Path{var p=Path.of("").toAbsolutePath();repeat(8){p.resolve("app/src/main/assets/flux/phase2h").takeIf(Files::isDirectory)?.let{return it};p=p.parent?:p};error("not found")}
}
