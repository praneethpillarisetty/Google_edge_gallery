package com.google.ai.edge.gallery.customtasks.flux

import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceImageContracts
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceImagePreprocessor
import com.google.ai.edge.gallery.customtasks.flux.image.FluxReferenceVaeEncoder
import com.google.ai.edge.gallery.customtasks.flux.runtime.FluxLiteRtEnvironment
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import java.security.MessageDigest
import javax.inject.Inject
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class FluxReferenceVaeVerificationState(val running: Boolean = false, val stage: String = "IDLE", val summary: String = "", val error: String? = null)

@HiltViewModel
class FluxReferenceVaeVerificationViewModel @Inject constructor(
  @ApplicationContext private val context: Context,
  private val repository: FluxDownloadRepository,
) : ViewModel() {
  private val mutableState = MutableStateFlow(FluxReferenceVaeVerificationState())
  val state = mutableState.asStateFlow()
  private var job: Job? = null

  fun run(uri: Uri, ready: Boolean) {
    if (job?.isActive == true) return
    if (!ready) { mutableState.value = FluxReferenceVaeVerificationState(error = "Model repository must be Ready.", stage = "ERROR"); return }
    job = viewModelScope.launch {
      val start = System.nanoTime(); val heapBefore = heap(); val pssBefore = Debug.getPss(); val thermalBefore = thermal()
      mutableState.value = FluxReferenceVaeVerificationState(true, "VALIDATING_FILES")
      try {
        val inputStart = System.nanoTime()
        val tensor = FluxReferenceImagePreprocessor(context.contentResolver).preprocess(uri) { stage -> mutableState.value = mutableState.value.copy(stage = stage) }
        val preprocessing = millis(inputStart)
        val summary = repository.withModelFilesLocked { root, manifest, metadata ->
          mutableState.value = mutableState.value.copy(stage = "HASHING_VAE_GRAPH")
          val environment = FluxLiteRtEnvironment()
          try {
            val encoder = FluxReferenceVaeEncoder(environment.createGpuGraphRunner())
            val graph = encoder.resolve(root, manifest)
            require(metadata.single { it.path == graph.name }.size == graph.length()) { "VAE graph size does not match authoritative metadata." }
            val hash = graph.inputStream().use { stream ->
              val digest = MessageDigest.getInstance("SHA-256"); val buffer = ByteArray(1024 * 1024)
              while (true) { val count = stream.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
              digest.digest().joinToString("") { "%02x".format(it) }
            }
            mutableState.value = mutableState.value.copy(stage = "COMPILING_VAE")
            val graphStart = System.nanoTime()
            mutableState.value = mutableState.value.copy(stage = "RUNNING_VAE")
            val latent = encoder.encode(graph, tensor)
            val graphMs = millis(graphStart)
            mutableState.value = mutableState.value.copy(stage = "VALIDATING_OUTPUT")
            """backend: GPU FP32
graph: ${graph.name}
locally observed SHA-256 (not publisher-verified): $hash
input shape: ${tensor.shape}
output shape: ${latent.shape}
output elements: ${latent.values.size}
all values finite: true
preprocessing: $preprocessing ms
graph: $graphMs ms
total: ${millis(start)} ms
Java heap: $heapBefore -> ${heap()} bytes
process PSS: $pssBefore -> ${Debug.getPss()} kB
thermal: $thermalBefore -> ${thermal()}
device: ${Build.MANUFACTURER} ${Build.MODEL}; Android API ${Build.VERSION.SDK_INT}
cancellation available: yes"""
          } finally { environment.close() }
        }
        mutableState.value = FluxReferenceVaeVerificationState(stage = "COMPLETE", summary = summary)
      } catch (e: CancellationException) { mutableState.value = FluxReferenceVaeVerificationState(stage = "CANCELLED", error = "Reference VAE verification cancelled.") }
      catch (oom: OutOfMemoryError) { mutableState.value = FluxReferenceVaeVerificationState(stage = "ERROR", error = "Not enough memory for reference VAE verification.") }
      catch (e: Exception) { mutableState.value = FluxReferenceVaeVerificationState(stage = "ERROR", error = e.message ?: "Reference VAE verification failed.") }
    }
  }
  fun cancel() { job?.cancel() }
  private fun millis(start: Long) = (System.nanoTime() - start) / 1_000_000
  private fun heap() = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
  private fun thermal() = context.getSystemService(PowerManager::class.java).currentThermalStatus
}
