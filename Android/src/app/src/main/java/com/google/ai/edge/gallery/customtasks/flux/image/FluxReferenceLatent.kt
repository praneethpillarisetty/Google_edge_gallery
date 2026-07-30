package com.google.ai.edge.gallery.customtasks.flux.image

class FluxReferenceLatent private constructor(private val storage: FloatArray) {
  val shape = FluxReferenceImageContracts.OUTPUT_SHAPE
  val layout = "NCHW"
  val dataType = "FP32"
  val size get() = storage.size
  internal fun copyValues(): FloatArray = storage.copyOf()
  companion object {
    fun checked(values: FloatArray): FluxReferenceLatent {
      require(values.size == FluxReferenceImageContracts.OUTPUT_ELEMENTS) { "kv_vae_enc.tflite returned the wrong element count." }
      require(values.all(Float::isFinite)) { "kv_vae_enc.tflite returned a non-finite value." }
      return FluxReferenceLatent(values.copyOf())
    }
  }
}
