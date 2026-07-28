/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.ai.edge.gallery.customtasks.flux.runtime

enum class FluxBackendPreference { AUTO, GPU, TENSOR_TPU }

sealed interface FluxBackendAvailability {
  data object Available : FluxBackendAvailability
  data object UnavailableSdk : FluxBackendAvailability
  data object UnsupportedDevice : FluxBackendAvailability
  data object MissingCompatibleArtifacts : FluxBackendAvailability
  data object UnsupportedModel : FluxBackendAvailability
}

/** Every field must be true before a Tensor TPU runner may be created or selected. */
data class FluxTensorTpuCapabilities(
  val sdkAvailable: Boolean,
  val deviceSupported: Boolean,
  val compatibleArtifactsInstalled: Boolean,
  val modelCompatibilityValidated: Boolean,
) {
  fun availability(): FluxBackendAvailability = when {
    !sdkAvailable -> FluxBackendAvailability.UnavailableSdk
    !deviceSupported -> FluxBackendAvailability.UnsupportedDevice
    !compatibleArtifactsInstalled -> FluxBackendAvailability.MissingCompatibleArtifacts
    !modelCompatibilityValidated -> FluxBackendAvailability.UnsupportedModel
    else -> FluxBackendAvailability.Available
  }
}

/** Capability boundary for a future Google Tensor SDK implementation; none is provided here. */
interface FluxTensorTpuProvider {
  fun capabilities(): FluxTensorTpuCapabilities
  fun createGraphRunner(): FluxGraphRunner
}

sealed interface FluxBackendSelection {
  data class Selected(
    val backend: FluxBackendPreference,
    val runner: FluxGraphRunner,
  ) : FluxBackendSelection

  data class Unavailable(
    val backend: FluxBackendPreference,
    val reason: FluxBackendAvailability,
  ) : FluxBackendSelection
}

class FluxBackendSelector(
  private val gpuRunner: FluxGraphRunner,
  private val tensorTpuProvider: FluxTensorTpuProvider? = null,
) {
  fun select(preference: FluxBackendPreference): FluxBackendSelection = when (preference) {
    FluxBackendPreference.GPU -> gpuSelection()
    FluxBackendPreference.AUTO -> selectTensorTpu(fallbackToGpu = true)
    FluxBackendPreference.TENSOR_TPU -> selectTensorTpu(fallbackToGpu = false)
  }

  private fun gpuSelection() =
    FluxBackendSelection.Selected(FluxBackendPreference.GPU, gpuRunner)

  private fun selectTensorTpu(fallbackToGpu: Boolean): FluxBackendSelection {
    val provider = tensorTpuProvider
    val availability = provider?.capabilities()?.availability()
      ?: FluxBackendAvailability.UnavailableSdk
    return if (availability == FluxBackendAvailability.Available && provider != null) {
      FluxBackendSelection.Selected(
        FluxBackendPreference.TENSOR_TPU,
        provider.createGraphRunner(),
      )
    } else if (fallbackToGpu) {
      gpuSelection()
    } else {
      FluxBackendSelection.Unavailable(FluxBackendPreference.TENSOR_TPU, availability)
    }
  }
}
