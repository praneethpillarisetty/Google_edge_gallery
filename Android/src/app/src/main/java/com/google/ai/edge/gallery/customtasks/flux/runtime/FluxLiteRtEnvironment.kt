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

import java.io.Closeable
import kotlinx.coroutines.sync.Mutex

/** Owns the one LiteRT Environment shared by every graph in a future FLUX pipeline. */
class FluxLiteRtEnvironment internal constructor(
  internal val factory: FluxLiteRtBackendFactory,
) : Closeable {
  constructor() : this(RealFluxLiteRtBackendFactory)

  private val environment = factory.createEnvironment()
  internal val gpuMutex = Mutex()
  private var closed = false
  private var activeUsers = 0

  internal fun acquire(): FluxEnvironmentHandle = synchronized(this) {
    check(!closed) { "FLUX LiteRT environment is closed." }
    activeUsers++
    environment
  }

  internal fun release() {
    val closeNow = synchronized(this) {
      check(activeUsers > 0)
      activeUsers--
      closed && activeUsers == 0
    }
    if (closeNow) environment.close()
  }

  fun createGpuGraphRunner(): FluxGpuGraphRunner = FluxGpuGraphRunner(this)

  override fun close() {
    val closeNow = synchronized(this) {
      if (closed) return
      closed = true
      activeUsers == 0
    }
    if (closeNow) environment.close()
  }
}
