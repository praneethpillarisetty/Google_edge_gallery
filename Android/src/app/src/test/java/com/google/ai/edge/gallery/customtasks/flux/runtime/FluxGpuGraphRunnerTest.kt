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
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class FluxGpuGraphRunnerTest {
  @Test fun `GPU runner implements hardware neutral graph runner`() {
    val runtime = FluxLiteRtEnvironment(FakeFactory())
    val runner: FluxGraphRunner = runtime.createGpuGraphRunner()
    assertTrue(runner is FluxGpuGraphRunner)
    runtime.close()
  }

  @Test fun `one shared environment is reused and FP32 GPU is requested`() = runBlocking {
    val factory = FakeFactory()
    FluxLiteRtEnvironment(factory).use { runtime ->
      runtime.createGpuGraphRunner().run(File("first.tflite"), listOf(floatArrayOf(1f)))
      runtime.createGpuGraphRunner().run(File("second.tflite"), listOf(floatArrayOf(2f)))
    }
    assertEquals(1, factory.environmentCreations)
    assertEquals(2, factory.options.size)
    factory.options.forEach {
      assertEquals(Accelerator.GPU, it.accelerator)
      assertEquals(FluxGpuPrecision.FP32, it.precision)
    }
  }

  @Test fun `inputs and outputs preserve model order and all resources close`() = runBlocking {
    val factory = FakeFactory(inputCount = 2, outputs = listOf(floatArrayOf(3f), floatArrayOf(4f)))
    val runtime = FluxLiteRtEnvironment(factory)
    val result = runtime.createGpuGraphRunner().run(
      File("ordered.tflite"),
      listOf(floatArrayOf(1f), floatArrayOf(2f)),
    )
    assertArrayEquals(floatArrayOf(1f), factory.graphs.single().inputs[0].written, 0f)
    assertArrayEquals(floatArrayOf(2f), factory.graphs.single().inputs[1].written, 0f)
    assertArrayEquals(floatArrayOf(3f), result.outputs[0], 0f)
    assertArrayEquals(floatArrayOf(4f), result.outputs[1], 0f)
    assertAllGraphResourcesClosed(factory.graphs.single())
    runtime.close()
    assertTrue(factory.environment.closed)
  }

  @Test fun `cleanup occurs when writing fails`() = runBlocking {
    assertFailureCleans(Failure.WRITE)
  }

  @Test fun `cleanup occurs when run fails`() = runBlocking {
    assertFailureCleans(Failure.RUN)
  }

  @Test fun `cleanup occurs when reading fails`() = runBlocking {
    assertFailureCleans(Failure.READ)
  }

  @Test fun `non-finite output is rejected with graph identity`() = runBlocking {
    val factory = FakeFactory(outputs = listOf(floatArrayOf(Float.NaN)))
    val runtime = FluxLiteRtEnvironment(factory)
    val error = try {
      runtime.createGpuGraphRunner().run(File("bad.tflite"), listOf(floatArrayOf(1f)))
      throw AssertionError("Expected failure")
    } catch (error: FluxRuntimeException) { error }
    assertTrue(error.message!!.contains("bad.tflite"))
    assertAllGraphResourcesClosed(factory.graphs.single())
    runtime.close()
  }

  @Test fun `incorrect input count is rejected and allocated resources close`() = runBlocking {
    val factory = FakeFactory(inputCount = 2)
    val runtime = FluxLiteRtEnvironment(factory)
    val error = try {
      runtime.createGpuGraphRunner().run(File("count.tflite"), listOf(floatArrayOf(1f)))
      throw AssertionError("Expected failure")
    } catch (error: FluxRuntimeException) { error }
    assertTrue(error.message!!.contains("expected 2 inputs"))
    assertTrue(factory.graphs.single().inputs.all { it.closed })
    assertTrue(factory.graphs.single().closed)
    runtime.close()
  }

  @Test fun `use after runtime close is rejected and repeated close is safe`() {
    val factory = FakeFactory()
    val runtime = FluxLiteRtEnvironment(factory)
    val runner = runtime.createGpuGraphRunner()
    runtime.close()
    runtime.close()
    assertEquals(1, factory.environment.closeCalls)
    val error = assertThrows(IllegalStateException::class.java) {
      runBlocking { runner.run(File("closed.tflite"), listOf(floatArrayOf(1f))) }
    }
    assertEquals("FLUX LiteRT environment is closed.", error.message)
  }

  @Test fun `concurrent runs are serialized`() = runBlocking {
    val active = AtomicInteger()
    val maximum = AtomicInteger()
    val factory = FakeFactory(onRun = {
      maximum.updateAndGet { maxOf(it, active.incrementAndGet()) }
      Thread.sleep(75)
      active.decrementAndGet()
    })
    val runtime = FluxLiteRtEnvironment(factory)
    val runner = runtime.createGpuGraphRunner()
    val first = async { runner.run(File("one.tflite"), listOf(floatArrayOf(1f))) }
    val second = async { runner.run(File("two.tflite"), listOf(floatArrayOf(2f))) }
    first.await(); second.await()
    assertEquals(1, maximum.get())
    runtime.close()
  }

  @Test fun `cancellation after reading still cleans native resources`() = runBlocking {
    val readStarted = CountDownLatch(1)
    val factory = FakeFactory(onRead = {
      readStarted.countDown()
      Thread.sleep(75)
    })
    val runtime = FluxLiteRtEnvironment(factory)
    val job = async { runtime.createGpuGraphRunner().run(File("cancel.tflite"), listOf(floatArrayOf(1f))) }
    assertTrue(readStarted.await(2, TimeUnit.SECONDS))
    job.cancel()
    try { job.await() } catch (_: CancellationException) {}
    assertAllGraphResourcesClosed(factory.graphs.single())
    runtime.close()
  }

  private suspend fun assertFailureCleans(failure: Failure) {
    val factory = FakeFactory(failure = failure)
    val runtime = FluxLiteRtEnvironment(factory)
    try {
      runtime.createGpuGraphRunner().run(File("failure.tflite"), listOf(floatArrayOf(1f)))
      throw AssertionError("Expected failure")
    } catch (_: FluxRuntimeException) {}
    assertAllGraphResourcesClosed(factory.graphs.single())
    runtime.close()
  }

  private fun assertAllGraphResourcesClosed(graph: FakeGraph) {
    assertTrue(graph.inputs.all { it.closed })
    assertTrue(graph.outputs.all { it.closed })
    assertTrue(graph.closed)
  }
}

private enum class Failure { NONE, WRITE, RUN, READ }
private class FakeEnvironment : FluxEnvironmentHandle {
  var closeCalls = 0
  val closed get() = closeCalls > 0
  override fun close() { closeCalls++ }
}
private class FakeBuffer(
  private val result: FloatArray = floatArrayOf(),
  private val failWrite: Boolean = false,
  private val failRead: Boolean = false,
  private val onRead: () -> Unit = {},
) : FluxTensorBuffer {
  var written: FloatArray? = null
  var closed = false
  override fun writeFloat(values: FloatArray) {
    if (failWrite) error("write failed")
    written = values
  }
  override fun readFloat(): FloatArray {
    onRead()
    if (failRead) error("read failed")
    return result
  }
  override fun close() { closed = true }
}
private class FakeGraph(
  inputCount: Int,
  outputValues: List<FloatArray>,
  private val failure: Failure,
  private val onRun: () -> Unit,
  onRead: () -> Unit,
) : FluxCompiledGraph {
  val inputs = List(inputCount) { FakeBuffer(failWrite = failure == Failure.WRITE && it == 0) }
  val outputs = outputValues.mapIndexed { index, value ->
    FakeBuffer(value, failRead = failure == Failure.READ && index == 0, onRead = onRead)
  }
  var closed = false
  override fun createInputBuffers(): List<FluxTensorBuffer> = inputs
  override fun createOutputBuffers(): List<FluxTensorBuffer> = outputs
  override fun run(inputs: List<FluxTensorBuffer>, outputs: List<FluxTensorBuffer>) {
    onRun()
    if (failure == Failure.RUN) error("run failed")
  }
  override fun close() { closed = true }
}
private class FakeFactory(
  private val inputCount: Int = 1,
  private val outputs: List<FloatArray> = listOf(floatArrayOf(9f)),
  private val failure: Failure = Failure.NONE,
  private val onRun: () -> Unit = {},
  private val onRead: () -> Unit = {},
) : FluxLiteRtBackendFactory {
  var environmentCreations = 0
  val environment = FakeEnvironment()
  val options = mutableListOf<FluxCompilationOptions>()
  val graphs = mutableListOf<FakeGraph>()
  override fun createEnvironment(): FluxEnvironmentHandle {
    environmentCreations++
    return environment
  }
  override fun createCompiledModel(
    environment: FluxEnvironmentHandle,
    modelPath: String,
    options: FluxCompilationOptions,
  ): FluxCompiledGraph {
    this.options += options
    return FakeGraph(inputCount, outputs, failure, onRun, onRead).also { graphs += it }
  }
}
