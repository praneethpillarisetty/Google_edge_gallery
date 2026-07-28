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

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class FluxBackendSelectorTest {
  private val gpuRunner = StubRunner("gpu")
  private val tpuRunner = StubRunner("tpu")

  @Test fun `AUTO selects GPU when Tensor TPU is unavailable`() {
    val provider = StubTensorTpuProvider(capabilities(sdk = false), tpuRunner)
    val selected = FluxBackendSelector(gpuRunner, provider).select(FluxBackendPreference.AUTO)

    assertSelected(selected, FluxBackendPreference.GPU, gpuRunner)
    assertEquals(0, provider.runnerCreations)
  }

  @Test fun `AUTO selects Tensor TPU only when every capability is verified`() {
    listOf(
      capabilities(sdk = false),
      capabilities(device = false),
      capabilities(artifacts = false),
      capabilities(model = false),
    ).forEach { incomplete ->
      val selected = FluxBackendSelector(
        gpuRunner,
        StubTensorTpuProvider(incomplete, tpuRunner),
      ).select(FluxBackendPreference.AUTO)
      assertSelected(selected, FluxBackendPreference.GPU, gpuRunner)
    }

    val provider = StubTensorTpuProvider(capabilities(), tpuRunner)
    val selected = FluxBackendSelector(gpuRunner, provider).select(FluxBackendPreference.AUTO)
    assertSelected(selected, FluxBackendPreference.TENSOR_TPU, tpuRunner)
    assertEquals(1, provider.runnerCreations)
  }

  @Test fun `explicit unavailable Tensor TPU produces clear failure without GPU fallback`() {
    val provider = StubTensorTpuProvider(capabilities(artifacts = false), tpuRunner)
    val selected = FluxBackendSelector(gpuRunner, provider)
      .select(FluxBackendPreference.TENSOR_TPU)

    assertTrue(selected is FluxBackendSelection.Unavailable)
    selected as FluxBackendSelection.Unavailable
    assertEquals(FluxBackendPreference.TENSOR_TPU, selected.backend)
    assertEquals(FluxBackendAvailability.MissingCompatibleArtifacts, selected.reason)
    assertEquals(0, provider.runnerCreations)
  }

  @Test fun `missing Tensor SDK provider is reported without silent fallback`() {
    val selected = FluxBackendSelector(gpuRunner).select(FluxBackendPreference.TENSOR_TPU)

    assertEquals(
      FluxBackendSelection.Unavailable(
        FluxBackendPreference.TENSOR_TPU,
        FluxBackendAvailability.UnavailableSdk,
      ),
      selected,
    )
  }

  private fun assertSelected(
    selection: FluxBackendSelection,
    backend: FluxBackendPreference,
    runner: FluxGraphRunner,
  ) {
    assertTrue(selection is FluxBackendSelection.Selected)
    selection as FluxBackendSelection.Selected
    assertEquals(backend, selection.backend)
    assertSame(runner, selection.runner)
  }

  private fun capabilities(
    sdk: Boolean = true,
    device: Boolean = true,
    artifacts: Boolean = true,
    model: Boolean = true,
  ) = FluxTensorTpuCapabilities(sdk, device, artifacts, model)
}

private class StubRunner(val identity: String) : FluxGraphRunner {
  override suspend fun run(modelFile: File, inputs: List<FloatArray>) =
    FluxGraphResult(identity, emptyList())
}

private class StubTensorTpuProvider(
  private val value: FluxTensorTpuCapabilities,
  private val runner: FluxGraphRunner,
) : FluxTensorTpuProvider {
  var runnerCreations = 0
  override fun capabilities() = value
  override fun createGraphRunner(): FluxGraphRunner {
    runnerCreations++
    return runner
  }
}
