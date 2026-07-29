/* Copyright 2025 Google LLC. Licensed under the Apache License, Version 2.0. */
package com.google.ai.edge.gallery.customtasks.flux

import android.content.Context
import android.os.StatFs
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.io.FileOutputStream
import java.io.Closeable
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext

data class FluxFileMetadata(val path: String, val size: Long, val sha256: String?)
data class FluxFileProgress(val path: String, val received: Long, val total: Long)

sealed interface FluxDownloadEvent {
  data object Checking : FluxDownloadEvent
  data class NotInstalled(val totalBytes: Long?) : FluxDownloadEvent
  data class Downloading(val files: List<FluxFileProgress>) : FluxDownloadEvent
  data object Paused : FluxDownloadEvent
  data class Ready(val totalBytes: Long) : FluxDownloadEvent
  data class Error(val message: String) : FluxDownloadEvent
}

interface FluxDownloadRepository {
  val events: Flow<FluxDownloadEvent>
  suspend fun check()
  suspend fun download()
  fun pause()
  suspend fun cancel()
  suspend fun <T> withModelFilesLocked(block: suspend (File, FluxModelManifest, List<FluxFileMetadata>) -> T): T
}

@Singleton
class DefaultFluxDownloadRepository
@Inject
constructor(@ApplicationContext private val context: Context) : FluxDownloadRepository {
  private val state = MutableStateFlow<FluxDownloadEvent>(FluxDownloadEvent.Checking)
  override val events = state.asStateFlow()
  private val ownership = FluxDownloadOwnership()
  private val cancellation get() = ownership.cancellation
  private var lastMetadata: List<FluxFileMetadata>? = null
  private val root: File
    get() = File(requireNotNull(context.getExternalFilesDir(null)), FLUX_MODEL_DIRECTORY)

  private fun manifest(): FluxModelManifest =
    context.assets.open("flux/manifest.json").bufferedReader().use {
      FluxModelManifest.parse(it.readText())
    }

  override suspend fun check() = ownership.exclusive { checkUnlocked() }

  private suspend fun checkUnlocked() = withContext(Dispatchers.IO) {
      state.value = FluxDownloadEvent.Checking
      try {
        val metadata = resolveMetadata(manifest()).also { lastMetadata = it }
        state.value =
          if (metadata.all(::isValid)) FluxDownloadEvent.Ready(metadata.sumOf { it.size })
          else FluxDownloadEvent.NotInstalled(metadata.sumOf { it.size })
      } catch (e: Exception) {
        state.value =
          FluxDownloadEvent.Error(
            "Could not retrieve authoritative model metadata. Downloads are disabled: ${e.message}"
          )
      }
    }

  override suspend fun download() = ownership.exclusive {
    withContext(Dispatchers.IO) {
      cancellation.reset()
      try {
        val source = manifest()
        val metadata = resolveMetadata(source).also { lastMetadata = it }
        val invalid = metadata.filterNot(::isValid)
        val partialBytes = invalid.sumOf { partialFor(it).length().coerceAtMost(it.size) }
        val needed = requiredStorageBytes(invalid.sumOf { it.size }, partialBytes, SAFETY_MARGIN)
        val available = StatFs(root.parentFile!!.absolutePath).availableBytes
        require(available >= needed) {
          "Not enough storage. ${formatBytes(needed)} required; ${formatBytes(available)} available."
        }
        val progress = metadata.map { FluxFileProgress(it.path, completedBytes(it), it.size) }.toMutableList()
        state.value = FluxDownloadEvent.Downloading(progress.toList())
        invalid.forEach { item ->
          downloadOne(source, item) { received ->
            val index = progress.indexOfFirst { it.path == item.path }
            progress[index] = FluxFileProgress(item.path, received, item.size)
            state.value = FluxDownloadEvent.Downloading(progress.toList())
          }
          if (cancellation.isRequested) {
            state.value = FluxDownloadEvent.Paused
            return@withContext
          }
        }
        state.value =
          if (metadata.all(::isValid)) FluxDownloadEvent.Ready(metadata.sumOf { it.size })
          else FluxDownloadEvent.Error("Model validation failed")
      } catch (e: Exception) {
        state.value = if (cancellation.isRequested) FluxDownloadEvent.Paused else FluxDownloadEvent.Error(e.message ?: "Download failed")
      }
    }
  }

  override fun pause() { ownership.pause() }

  override suspend fun cancel() {
    // Signal and close current I/O first. Only its owner may then finish/close its streams.
    ownership.cancel(
      cleanup = { withContext(Dispatchers.IO) {
        root.walkTopDown().filter { it.isFile && it.name.endsWith(".partial") }.forEach(File::delete)
      } },
      recheck = { checkUnlocked() },
    )
  }

  override suspend fun <T> withModelFilesLocked(
    block: suspend (File, FluxModelManifest, List<FluxFileMetadata>) -> T
  ): T = ownership.exclusive {
    val source = manifest()
    val metadata = lastMetadata ?: resolveMetadata(source).also { lastMetadata = it }
    block(root, source, metadata)
  }

  private fun resolveMetadata(manifest: FluxModelManifest): List<FluxFileMetadata> =
    manifest.files.map { path ->
      val connection = connection(manifest, path).apply { requestMethod = "HEAD" }
      connection.connect()
      require(connection.responseCode in 200..299) { "Metadata request failed for $path (${connection.responseCode})" }
      val size = connection.getHeaderFieldLong("x-linked-size", connection.contentLengthLong)
      require(size > 0) { "Authoritative size is unavailable for $path" }
      val etag = connection.getHeaderField("x-linked-etag")?.trim('"')
      val sha = etag?.takeIf { it.matches(Regex("[a-fA-F0-9]{64}")) }?.lowercase()
      connection.disconnect()
      FluxFileMetadata(path, size, sha)
    }

  private fun downloadOne(
    manifest: FluxModelManifest,
    item: FluxFileMetadata,
    update: (Long) -> Unit,
  ): Unit {
    val output = fileFor(item)
    output.parentFile?.mkdirs()
    if (output.exists() && !isValid(item)) output.delete()
    val partial = partialFor(item)
    if (partial.length() > item.size) partial.delete()
    var offset = partial.length()
    val connection = connection(manifest, item.path)
    if (offset > 0) {
      connection.setRequestProperty("Range", "bytes=$offset-")
      connection.setRequestProperty("Accept-Encoding", "identity")
    }
    val connectionHandle = Closeable { connection.disconnect() }
    cancellation.attach(connectionHandle)
    try {
      connection.connect()
      if (offset > 0 && connection.responseCode != HttpURLConnection.HTTP_PARTIAL) {
        partial.delete()
        offset = 0
        cancellation.detach(connectionHandle)
        connection.disconnect()
        return downloadOne(manifest, item, update)
      }
      require(connection.responseCode in 200..299) { "Download failed for ${item.path} (${connection.responseCode})" }
      connection.inputStream.use { input ->
        cancellation.attach(input)
        try {
          FileOutputStream(partial, offset > 0).use { outputStream ->
            cancellation.attach(outputStream)
            try {
              val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
              var received = offset
              while (!cancellation.isRequested) {
                val count = input.read(buffer)
                if (count < 0) break
                outputStream.write(buffer, 0, count)
                received += count
                update(received)
              }
              if (!cancellation.isRequested) outputStream.fd.sync()
            } finally {
              cancellation.detach(outputStream)
            }
          }
        } finally {
          cancellation.detach(input)
        }
      }
    } finally {
      cancellation.detach(connectionHandle)
      connection.disconnect()
    }
    if (cancellation.isRequested) return
    require(partial.length() == item.size) { "Size validation failed for ${item.path}" }
    item.sha256?.let { require(sha256(partial) == it) { "SHA-256 validation failed for ${item.path}" } }
    require(partial.renameTo(output)) { "Could not atomically install ${item.path}" }
  }

  private fun connection(manifest: FluxModelManifest, path: String) =
    (URL("https://huggingface.co/${manifest.repository}/resolve/${manifest.revision}/${requireSafeRelativePath(path)}").openConnection() as HttpURLConnection).apply {
      connectTimeout = 30_000
      readTimeout = 30_000
      instanceFollowRedirects = true
      setRequestProperty("Accept-Encoding", "identity")
    }

  private fun fileFor(item: FluxFileMetadata) = safeChild(root, item.path)
  private fun partialFor(item: FluxFileMetadata) = File(fileFor(item).path + ".partial")
  private fun completedBytes(item: FluxFileMetadata) = if (isValid(item)) item.size else partialFor(item).length()
  private fun isValid(item: FluxFileMetadata): Boolean {
    val file = fileFor(item)
    return file.isFile && file.length() == item.size && (item.sha256 == null || sha256(file) == item.sha256)
  }

  companion object {
    const val SAFETY_MARGIN = 512L * 1024 * 1024
    fun safeChild(root: File, path: String): File {
      requireSafeRelativePath(path)
      val result = File(root, path).canonicalFile
      require(result.path.startsWith(root.canonicalPath + File.separator)) { "Path escapes model directory" }
      return result
    }
    fun sha256(file: File): String {
      val digest = MessageDigest.getInstance("SHA-256")
      file.inputStream().use { input ->
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
          val count = input.read(buffer)
          if (count < 0) break
          digest.update(buffer, 0, count)
        }
      }
      return digest.digest().joinToString("") { "%02x".format(it) }
    }
    fun formatBytes(bytes: Long) = "%.2f GB".format(bytes / 1_000_000_000.0)
  }
}
