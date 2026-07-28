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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlin.coroutines.coroutineContext

/** Runs one downloaded graph at a time and never retains a compiled graph between calls. */
class FluxGpuGraphRunner internal constructor(private val runtime: FluxLiteRtEnvironment) :
  FluxGraphRunner {
  override suspend fun run(modelFile: File, inputs: List<FloatArray>): FluxGraphResult =
    withContext(Dispatchers.Default) {
      runtime.gpuMutex.withLock {
        coroutineContext.ensureActive()
        val environment = runtime.acquire()
        try {
          execute(modelFile, inputs, environment)
        } finally {
          runtime.release()
        }
      }
    }

  private suspend fun execute(
    modelFile: File,
    inputs: List<FloatArray>,
    environment: FluxEnvironmentHandle,
  ): FluxGraphResult {
    val graphName = modelFile.name
    var compiled: FluxCompiledGraph? = null
    val inputBuffers = mutableListOf<FluxTensorBuffer>()
    val outputBuffers = mutableListOf<FluxTensorBuffer>()
    try {
      coroutineContext.ensureActive()
      val activeGraph = runtime.factory.createCompiledModel(
        environment,
        modelFile.absolutePath,
        FluxCompilationOptions(),
      )
      compiled = activeGraph
      inputBuffers += activeGraph.createInputBuffers()
      if (inputs.size != inputBuffers.size) {
        throw FluxRuntimeException(
          "Graph $graphName expected ${inputBuffers.size} inputs but received ${inputs.size}.",
        )
      }
      outputBuffers += activeGraph.createOutputBuffers()
      inputs.forEachIndexed { index, values -> inputBuffers[index].writeFloat(values) }
      coroutineContext.ensureActive()
      activeGraph.run(inputBuffers, outputBuffers)
      val outputs = outputBuffers.map { it.readFloat() }
      coroutineContext.ensureActive()
      outputs.forEachIndexed { outputIndex, values ->
        if (values.any { !it.isFinite() }) {
          throw FluxRuntimeException("Graph $graphName output $outputIndex contains a non-finite value.")
        }
      }
      return FluxGraphResult(graphName, outputs)
    } catch (error: kotlinx.coroutines.CancellationException) {
      throw error
    } catch (error: FluxRuntimeException) {
      throw error
    } catch (error: Exception) {
      throw FluxRuntimeException("Graph $graphName execution failed.", error)
    } finally {
      inputBuffers.forEach { runCatching { it.close() } }
      outputBuffers.forEach { runCatching { it.close() } }
      runCatching { compiled?.close() }
    }
  }
}
