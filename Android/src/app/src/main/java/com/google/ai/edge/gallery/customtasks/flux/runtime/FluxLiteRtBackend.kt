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

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable

internal enum class FluxGpuPrecision { FP32 }
internal data class FluxCompilationOptions(
  val accelerator: Accelerator = Accelerator.GPU,
  val precision: FluxGpuPrecision = FluxGpuPrecision.FP32,
)

internal interface FluxEnvironmentHandle : Closeable
internal interface FluxTensorBuffer : Closeable {
  fun writeFloat(values: FloatArray)
  fun readFloat(): FloatArray
}
internal interface FluxCompiledGraph : Closeable {
  fun createInputBuffers(): List<FluxTensorBuffer>
  fun createOutputBuffers(): List<FluxTensorBuffer>
  fun run(inputs: List<FluxTensorBuffer>, outputs: List<FluxTensorBuffer>)
}
internal interface FluxLiteRtBackendFactory {
  fun createEnvironment(): FluxEnvironmentHandle
  fun createCompiledModel(
    environment: FluxEnvironmentHandle,
    modelPath: String,
    options: FluxCompilationOptions,
  ): FluxCompiledGraph
}

internal object RealFluxLiteRtBackendFactory : FluxLiteRtBackendFactory {
  override fun createEnvironment(): FluxEnvironmentHandle = RealEnvironment(Environment.create())

  override fun createCompiledModel(
    environment: FluxEnvironmentHandle,
    modelPath: String,
    options: FluxCompilationOptions,
  ): FluxCompiledGraph {
    require(options.accelerator == Accelerator.GPU)
    require(options.precision == FluxGpuPrecision.FP32)
    val compiledOptions = CompiledModel.Options(Accelerator.GPU).apply {
      gpuOptions = CompiledModel.GpuOptions(
        precision = CompiledModel.GpuOptions.Precision.FP32,
      )
    }
    val nativeEnvironment = (environment as RealEnvironment).delegate
    return RealCompiledGraph(CompiledModel.create(modelPath, compiledOptions, nativeEnvironment))
  }
}

private class RealEnvironment(val delegate: Environment) : FluxEnvironmentHandle {
  override fun close() = delegate.close()
}

private class RealTensorBuffer(val delegate: TensorBuffer) : FluxTensorBuffer {
  override fun writeFloat(values: FloatArray) = delegate.writeFloat(values)
  override fun readFloat(): FloatArray = delegate.readFloat()
  override fun close() = delegate.close()
}

private class RealCompiledGraph(private val delegate: CompiledModel) : FluxCompiledGraph {
  override fun createInputBuffers() = delegate.createInputBuffers().map(::RealTensorBuffer)
  override fun createOutputBuffers() = delegate.createOutputBuffers().map(::RealTensorBuffer)
  override fun run(inputs: List<FluxTensorBuffer>, outputs: List<FluxTensorBuffer>) =
    delegate.run(inputs.map { (it as RealTensorBuffer).delegate }, outputs.map { (it as RealTensorBuffer).delegate })
  override fun close() = delegate.close()
}
