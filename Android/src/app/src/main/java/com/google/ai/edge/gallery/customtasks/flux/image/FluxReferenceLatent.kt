package com.google.ai.edge.gallery.customtasks.flux.image

data class FluxReferenceLatent private constructor(val values: FloatArray) {
  val shape = FluxReferenceImageContracts.OUTPUT_SHAPE
  val dataType = "FP32"
  companion object {
    fun checked(values: FloatArray): FluxReferenceLatent {
      require(values.size == FluxReferenceImageContracts.OUTPUT_ELEMENTS) { "kv_vae_enc.tflite returned the wrong element count." }
      require(values.all(Float::isFinite)) { "kv_vae_enc.tflite returned a non-finite value." }
      return FluxReferenceLatent(values)
    }
  }
}
