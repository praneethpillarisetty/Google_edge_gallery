/* Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux.decoder
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceConstants
internal class FluxDecoderInput private constructor(private val owned:FloatArray) {
  val shape=listOf(1,32,32,32); val size get()=owned.size; fun copyValues()=owned.copyOf()
  companion object { fun checked(values:FloatArray):FluxDecoderInput { require(values.size==32_768 && values.all(Float::isFinite)); return FluxDecoderInput(values.copyOf()) } }
}
internal data class FluxDecoderTailMetrics(val unpackedFinite:Boolean,val decoderInputFinite:Boolean)
internal class FluxDecoderTail(private val evidence:FluxPhase2hEvidence,private val constants:FluxReferenceConstants) {
  fun prepare(input:FloatArray,check:()->Unit={}):Pair<FluxDecoderInput,FluxDecoderTailMetrics> {
    check(); validate(input,"final Phase 2G latents"); val source=input.copyOf()
    check(); val unpacked=FloatArray(elements(1,128,16,16)); check()
    for(i in unpacked.indices){if(i and 1023==0)check();unpacked[i]=source[evidence.unpackAt(i)]};validate(unpacked,"unpacked latents")
    check(); val denormalized=FloatArray(unpacked.size); check()
    for(i in denormalized.indices){if(i and 1023==0)check();val channel=i/256;denormalized[i]=unpacked[i]*constants.standardDeviation[channel]+constants.mean[channel]};validate(denormalized,"denormalized latents")
    check(); val output=FloatArray(denormalized.size);check()
    for(i in output.indices){if(i and 1023==0)check();output[i]=denormalized[evidence.unpatchAt(i)]};validate(output,"decoder input");check()
    return FluxDecoderInput.checked(output) to FluxDecoderTailMetrics(true,true)
  }
  private fun validate(v:FloatArray,name:String){require(v.size==32_768&&v.all(Float::isFinite)){"$name must be 32768 finite FP32 values."}}
  private fun elements(vararg d:Int)=d.fold(1){p,v->Math.multiplyExact(p,v)}
}
