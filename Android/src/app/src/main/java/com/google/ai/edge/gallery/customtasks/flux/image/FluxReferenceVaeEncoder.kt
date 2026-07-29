package com.google.ai.edge.gallery.customtasks.flux.image

import com.google.ai.edge.gallery.customtasks.flux.FLUX_MODEL_REVISION
import com.google.ai.edge.gallery.customtasks.flux.FluxModelManifest
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxGraphRunner
import java.io.File
import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext

class FluxReferenceVaeEncoder(private val runner: FluxGraphRunner) {
  fun resolve(root: File, manifest: FluxModelManifest): File {
    require(manifest.revision == FLUX_MODEL_REVISION)
    require(FluxReferenceImageContracts.GRAPH in manifest.files)
    val canonicalRoot = root.canonicalFile
    val graph = File(canonicalRoot, FluxReferenceImageContracts.GRAPH).canonicalFile
    require(graph.parentFile == canonicalRoot && graph.isFile) { "Required VAE graph is missing or unsafe: ${FluxReferenceImageContracts.GRAPH}" }
    return graph
  }
  suspend fun encode(graph: File, input: FluxReferenceImageTensor): FluxReferenceLatent {
    require(input.shape == FluxReferenceImageContracts.INPUT_SHAPE)
    coroutineContext.ensureActive()
    val result = runner.run(graph, listOf(input.values))
    coroutineContext.ensureActive()
    require(result.graph == FluxReferenceImageContracts.GRAPH) { "Unexpected VAE graph result: ${result.graph}" }
    require(result.outputs.size == 1) { "${FluxReferenceImageContracts.GRAPH} returned ${result.outputs.size} outputs; expected one." }
    return FluxReferenceLatent.checked(result.outputs.single())
  }
}
