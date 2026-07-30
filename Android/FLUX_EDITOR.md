# FLUX.2 Klein LiteRT image editor

## Phase 2F: editing transformer preparation (debug only)

Phase 2F is based only on `google-ai-edge/litert-samples` immutable commit
`f48a89e4f29a74ab51f29c311ac7a0e5e479d225`. The exact inspected files are
`compiled_model_api/text_to_image/flux2_klein_kotlin_gpu/android/app/src/main/java/com/google/ai/edge/examples/flux2_klein/Flux2KleinGenerator.kt`,
`compiled_model_api/text_to_image/flux2_klein_kotlin_gpu/conversion/chunked_export_klein.py`,
`compiled_model_api/text_to_image/flux2_klein_kotlin_gpu/conversion/build_klein_dit.py`, and
`compiled_model_api/text_to_image/flux2_klein_kotlin_gpu/conversion/gen_prep_klein.py`. Mutable
upstream `main` and generic FLUX implementations are not evidence for this contract.

The editing sequence is exactly `cat([noiseTokens, referenceTokens], image-token axis)`: noise
occupies tokens 0–255 and reference tokens occupy 256–511. Both inputs are `[1,256,128]` FP32
(32,768 elements), and the owned, finite result is `[1,512,128]` FP32 (65,536 elements). Assembly
uses checked arithmetic, defensive ownership, bounded cancellation checks, and never mutates either
typed input.

The sole Phase 2F graph call is `kce_prep.tflite(editingImageTokens,
promptConditioning, timestepEmbedding)`, in that exact order. Shapes are `[1,512,128]`,
`[1,512,7680]`, and `[1,3072]`. The graph must return exactly five ordered finite FP32 outputs:
image hidden `[1,512,3072]` (1,572,864), text hidden `[1,512,3072]` (1,572,864), image modulation
`[1,1,18432]` (18,432), text modulation `[1,1,18432]` (18,432), and single-stream modulation
`[1,1,9216]` (9,216). Runtime resolution retains the app-owned model-file lock, checks the
canonical path and authoritative manifest size, and shared GPU FP32 execution remains sequential
with at most one compiled graph resident. Inputs, outputs, compiled graph, and environment are
released in native cleanup order on success, failure, or cancellation; no later graph is run.

The developer diagnostic alone creates `syntheticZeroNoiseTokens` (32,768 zeros) and
`syntheticZeroTimestepEmbedding` (3,072 zeros). These establish only structural device verification.
Production random noise, distributions, scheduler sigmas/deltas, timestep values, and learned
production timestep embeddings remain unimplemented. A successful Pixel run would prove only that
the pinned prep graph compiles/runs on that device with GPU FP32 and returns five correctly shaped,
finite buffers. It would not prove semantic output, scheduler/timestep/noise parity, diffusion,
transformer-block execution, decoding, generation, or end-to-end memory safety.

Major host arrays are: prompt conditioning 3,932,160 floats / 15,728,640 bytes; reference tokens
32,768 / 131,072; diagnostic noise 32,768 / 131,072; editing tokens 65,536 / 262,144; synthetic
timestep 3,072 / 12,288; each image/text hidden output 1,572,864 / 6,291,456; each image/text
modulation 18,432 / 73,728; and single modulation 9,216 / 36,864. These are host-array sizes, not
GPU/native peak-memory claims. Outputs are validated and released immediately and no tensor array
is stored in Compose state. Double-stream, single-stream, final, scheduler/update, and VAE decoder
graphs remain deferred. **Generate remains disabled.** Earlier Phase 2E and physical Pixel results
below remain unchanged.

Phase 1 adds the editor input UI and secure, resumable model management. It does **not** run FLUX
inference, produce an output image, or use a cloud fallback.

The model is several gigabytes. The app obtains every exact file size from authoritative Hugging
Face response metadata before enabling download, sums those values for display, and additionally
requires enough free app-storage space for remaining `.partial` data and a 512 MB safety margin.
Downloads remain disabled with an explicit error if authoritative sizes cannot be obtained. A
64-character `x-linked-etag` is treated as the Hugging Face LFS SHA-256; otherwise hash validation
is explicitly unavailable and exact-size validation remains mandatory.

The current mobile export targets LiteRT GPU, exactly one reference image, and 256×256 resolution.
Files are downloaded only to app-owned external storage. Completed files survive later failures;
retry processes only missing or invalid files. Temporary `.partial` files permit byte-range resume
and are renamed only after validation.

## Phase 2A: prompt tokens and embedding rows

Phase 2A adds offline prompt tokenization and an embedding-table reader. Inspection of the three
authoritative export files established these serialization formats:

* `qwen_vocab.txt` is UTF-8, one token string per line, with no header or delimiter. The zero-based
  line number is the token ID. It has 151,669 records (IDs 0 through 151,668). Token strings use the
  GPT-2/Qwen byte-to-Unicode alphabet (for example, `Ġ` represents byte `0x20`); they are not escaped
  or JSON encoded.
* `qwen_merges.txt` is UTF-8, has no header, and contains exactly two non-empty token strings joined
  by one ASCII space per line. Its zero-based line order is the BPE rank (151,387 ranks); earlier
  lines have higher merge priority.
* `qwen_special.txt` is UTF-8 and contains `token<TAB>decimal-id`, one record per line. All 26
  records were inspected: `<|endoftext|>`=151643, `<|im_start|>`=151644,
  `<|im_end|>`=151645, `<|object_ref_start|>`=151646, `<|object_ref_end|>`=151647,
  `<|box_start|>`=151648, `<|box_end|>`=151649, `<|quad_start|>`=151650,
  `<|quad_end|>`=151651, `<|vision_start|>`=151652, `<|vision_end|>`=151653,
  `<|vision_pad|>`=151654, `<|image_pad|>`=151655, `<|video_pad|>`=151656,
  `<tool_call>`=151657, `</tool_call>`=151658, `<|fim_prefix|>`=151659,
  `<|fim_middle|>`=151660, `<|fim_suffix|>`=151661, `<|fim_pad|>`=151662,
  `<|repo_name|>`=151663, `<|file_sep|>`=151664, `<tool_response>`=151665,
  `</tool_response>`=151666, `<think>`=151667, and `</think>`=151668. Each mapping also agrees
  with the same token's implicit ID in the vocabulary file.

The implementation explicitly reads UTF-8, validates duplicates and cross-file references, applies
the official Qwen2 pre-tokenization expression, maps the literal prompt's original UTF-8 bytes
through the byte-to-Unicode alphabet, and performs deterministic ranked BPE. It does not apply
Unicode normalization, so canonically equivalent composed and decomposed text remains byte-distinct.
Authoritative special strings are isolated before ordinary pre-tokenization. Like the official Qwen2 BPE model,
there is no byte fallback or BPE unknown token: an unrepresentable piece is reported as a prompt
preparation error rather than silently substituted. Parsed immutable tables are cached by file
identity, length, and modification time.

The reviewed official Hugging Face `Qwen2Tokenizer` defaults establish `<|endoftext|>` as both EOS
and padding, no BOS token, no automatically added BOS/EOS, right padding, and right truncation.
Accordingly, Phase 2A tokenizes the literal user prompt without a wrapper or Unicode normalization,
retains the first 512 token IDs, and pads on the right to exactly 512 with ID 151643 while returning a
parallel validity representation. The export files themselves specify neither semantic token roles,
the 512-token model limit, nor a prompt template. The literal-prompt decision and absence of a
pipeline prompt wrapper therefore remain pending verification against an authoritative FLUX mobile
pipeline or tokenizer fixture; complete end-to-end tokenizer parity is not claimed without that
fixture.

`tokenizer/qwen_embed_fp16.bin` is contracted as 151,936 rows × 2,560 values × 2 bytes, exactly
777,912,320 bytes, row-major IEEE-754 binary16 in little-endian order. The reader checks that exact
length, opens a `FileChannel` with `READ` only, creates a read-only memory mapping, and converts only
requested rows into a flat primitive `FloatArray`. It never copies the complete table to the heap,
rejects IDs outside `0 until 151936`, and explicitly owns and closes its channel.

Mask tensor materialization, LiteRT environment/model setup and inference, GPU execution, image/VAE
processing, scheduling, decoding, and cloud fallback remain unimplemented. Generate remains
unconditionally disabled; Phase 2A does not produce images or placeholder results.

## Phase 2B: LiteRT GPU runner infrastructure

Phase 2B adds generic graph-runner infrastructure using the LiteRT `CompiledModel` Kotlin API from
`com.google.ai.edge.litert:litert:2.1.0`. A future complete FLUX pipeline owns one non-null shared
`Environment`; individual graphs reuse it rather than creating an environment per graph. The owner
has an explicit, idempotent `Closeable` lifecycle and rejects new work after it is closed.

GPU execution is mandatory and compilation explicitly requests `Accelerator.GPU` with
`CompiledModel.GpuOptions(precision = Precision.FP32)`. FP16 fallback is intentionally unavailable:
the model's modulated blocks can overflow and produce NaNs at FP16 precision. Calls are dispatched
off the main thread and serialized by a coroutine `Mutex`, so two generations cannot compile or run
large GPU graphs concurrently.

Only one `CompiledModel` is resident at a time. Each call compiles one downloaded `.tflite` file,
creates ordered input and output buffers, writes and reads FP32 arrays in model-reported order, then
deterministically closes every input buffer, every output buffer, and finally the compiled model.
Cleanup also runs for validation errors, LiteRT failures, and coroutine cancellation. The graph is
released before another graph can load; compiled models are not cached in this checkpoint.

Phase 2B unit tests use fake runtime handles to verify ordering, FP32 option selection, serialization,
failure paths, cancellation, and cleanup. They do not demonstrate physical-device GPU execution.
No FLUX graph—including the text encoder, conditioning, or VAE graphs—is executed yet, no rotary
tables or masks are synthesized, and prompt wrapping remains unspecified. **Generate remains
disabled**, and this phase does not produce an image.

### Backend boundary

The only executable Phase 2B backend infrastructure is the verified LiteRT GPU implementation
described above. The current authoritative FLUX artifact manifest is GPU-targeted and is represented
as a distinct GPU artifact set; it is never treated as a Tensor TPU artifact set or sent to
`Accelerator.NPU`.

`TENSOR_TPU` is an architectural extension point, not an operational backend in this checkpoint.
Working Tensor TPU support requires access to the Google Tensor SDK beta, a supported device,
separately published and installed TPU-compiled FLUX artifacts, and validated model compatibility.
No Google Tensor SDK dependency, TPU runner, TPU artifact filenames, hashes, sizes, URLs, or model
metadata are included. `AUTO` may select a future Tensor TPU provider only after all four conditions
are verified; otherwise it selects GPU. An explicit unavailable `TENSOR_TPU` request reports why it
is unavailable and never silently falls back to GPU.

No physical-device GPU or Tensor TPU execution is claimed. No FLUX graph is invoked by the app in
Phase 2B, and **Generate remains disabled**.

## Phase 2C: Qwen text conditioning

Phase 2C pins the model manifest to immutable Hugging Face revision
`f9b9171c841790a39147903febe73a85e9eaf42e`. The 17 manifest names were compared with the download
inventory in the companion implementation. Hugging Face file metadata remained inaccessible from
the development proxy, so the three full-file graph SHA-256 values were not fabricated; verifying
the downloaded LFS objects remains part of physical-device validation.

The authoritative host contract comes from the open (not merged) `google-ai-edge/litert-samples`
PR 227, pinned locally at commit `f48a89e4f29a74ab51f29c311ac7a0e5e479d225`. In particular,
`conversion/build_klein_enc.py`, `conversion/gen_prep_klein.py`,
`conversion/gen_verify_klein.py`, `conversion/export_tokenizer_klein.py`, and the Kotlin
`PromptEncoder`, `QwenTokenizer`, and `Flux2KleinGenerator` sources were reviewed in full. The
implementation is pinned as evidence rather than described as merged upstream.

### Authoritative text contract

The Qwen chat text is exactly
`<|im_start|>user\n{literal prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`.
Its prefix IDs are `[151644, 872, 198]`, suffix IDs are
`[151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]`, and only the literal prompt body is
truncated to make the wrapped sequence fit 512 positions. The original body is still converted
directly to UTF-8 without normalization. Padding is on the right with ID 151643 and a parallel
validity array distinguishes wrapper/body tokens from padding.

Each of `ke_enc0.tflite`, `ke_enc1.tflite`, and `ke_enc2.tflite` has four ordered, row-major FP32
inputs: hidden state `[1,512,2560]`, additive causal-plus-padding mask `[1,32,512,512]`, Qwen
rotary cosine `[1,512,128]`, and Qwen rotary sine `[1,512,128]`. Each has one FP32 output
`[1,512,2560]`. The mask uses 0 for allowed entries, adds `-1e9` when a key is in the future, and
adds `-1e9` when a key is padded. Consequently an entry that is both future and padded is `-2e9`.
Padded query rows are not blocked from attending to earlier valid keys. All 32 head planes are
materialized: `[1,1,512,512]` broadcasting is prohibited because the target GPU delegate can
silently miscompile that attention add.

Encoder positions are `0..511`. Qwen3 rotary values use base 1,000,000 and head width 128: the 64
inverse frequencies are `1 / base^(2*i/128)`, their position products are cosine/sine transformed,
and each 64-value half is concatenated with itself to form `[1,512,128]`. Every generated input is
checked for its exact element count and finite FP32 values.

Execution is strictly `ke_enc0` (layers 1–9, tap h9), `ke_enc1` (layers 10–18, tap h18), then
`ke_enc2` (layers 19–27, tap h27). The previous output occupies input slot zero of the next graph;
mask, cosine, and sine remain input slots one through three. Phase 2B compiles each call with GPU
FP32 and closes its buffers and model before returning, so only one `CompiledModel` is resident.
The three taps are written directly into a destination in token-major order equivalent to
`stack(taps, 1).transpose(0, 2, 1, 3).reshape(1, 512, 7680)`. The typed result is FP32 with shape
`[1,512,7680]` and contains per-stage durations, never tensor values or the prompt.

### Memory and lifecycle

The checked host-array estimate is 70,778,880 bytes: embeddings 5,242,880 bytes; expanded mask
33,554,432 bytes; cosine plus sine 524,288 bytes; three retained taps 15,728,640 bytes; and the
final conditioning destination 15,728,640 bytes. This excludes LiteRT-owned tensor buffers,
compiled weights, runtime overhead, and short-lived tokenizer structures, so it is not a claim of
Pixel 10 Pro XL memory safety. FP16 embedding rows unavoidably expand to 5,242,880 bytes of FP32 for
the graph input. The 777,912,320-byte table remains read-only memory mapped and is deterministically
closed after preprocessing. Checked multiplication rejects overflow; preprocessing and every graph
boundary check coroutine cancellation; failure prevents later graphs from running. Phase 2B owns
and closes all native tensor/model resources on success, error, and cancellation.

JVM tests use small contract-shaped fixtures and fake `FluxGraphRunner` instances. They verify graph
and input order, stage wiring, sizes, non-finite rejection, representative exact mask and rotary
values, tap interleaving, cancellation, failure short-circuiting, asset completeness/canonical path
checks, cleanup, memory accounting, tokenizer UTF-8 behavior, and the immutable manifest. They do
not execute LiteRT or establish reference numeric parity for the 912 MB graphs. No official numeric
tensor fixture was committed by the pinned PR, so no expected tensor values were invented.

A later Pixel run must verify all three exact downloaded graph hashes, GPU FP32 compilation and
sequential execution, output shape and finiteness, parity tolerances against the official pipeline,
stage durations, native/Java peak memory, cancellation, and thermal behavior. Phase 2C does not run
image conditioning, diffusion, denoising, VAE code, or bitmap creation. **Generate remains
disabled.**

## Phase 2C.1: physical-device text-encoder verification (debug only)

Build and install the diagnostic APK from `Android/src` with
`./gradlew --no-daemon app:assembleDebug --console=plain`, followed by
`adb install -r app/build/outputs/apk/debug/app-debug.apk`. Open **FLUX Image Editor**, complete the
model download until its state is **Ready**, then use **Developer verification — Text encoder only**.
This section is supplied only by the debug source set; release builds contain an empty source-set
implementation and expose no verification action or navigation route. Generate remains disabled in
both variants.

Use a non-sensitive prompt. The run validates every tokenizer, embedding, and encoder asset against
the authoritative resolved size metadata and its canonical app-owned location. It then streams each
encoder graph through SHA-256 with cancellation checks and caches the observation by filename, size,
and modification time. Because publisher graph digests are not present in authoritative metadata,
the summary deliberately labels these values **observed**, not publisher-verified. A changed file
identity invalidates the cached observation.

The expected graph order is `ke_enc0.tflite` → `ke_enc1.tflite` → `ke_enc2.tflite`, with one GPU
FP32 compiled model resident at a time. Success requires a finite `[1,512,7680]` result. Expect a
multi-minute run and significant device heating; emergency/shutdown thermal states are rejected and
severe states remain visible as a warning in the diagnostic context. The UI reports approximate Java
heap and process PSS—not exact native or GPU allocation—and thermal status before and after.

Cancel is checked while hashing and preprocessing and between graph stages. Leaving the editor
cancels verification when its navigation-scoped ViewModel is permanently cleared; configuration
recreation retains that ViewModel and therefore cannot start a duplicate run. Downloads, retries,
deletion of partial files, and verification share the same model-file operation lock.

To capture a sanitized report, tap **Copy diagnostic summary** or run:

```shell
adb logcat -c
adb logcat -v threadtime | sed -E 's#(/storage/[^ ]+|prompt=[^ ]+)#[REDACTED]#g' > flux-device.txt
```

Review the file before sharing it. The built-in summary contains graph filenames, locally observed
hashes, timings, shape, finiteness, approximate memory, device/API and thermal information, but no
prompt, token IDs, tensor values, credentials, or filesystem paths.

A successful run proves that this device can resolve the pinned files, preprocess the prompt, compile
and execute all three text graphs sequentially on LiteRT GPU FP32, and produce a correctly shaped,
finite conditioning tensor. It does **not** prove reference numeric parity, diffusion, image
conditioning, denoising, VAE execution, image generation, or end-to-end memory safety. No placeholder
image is produced, and **Generate remains disabled**.

## Phase 2C.1 locally observed Pixel result

A **local physical-device observation**, not publisher reference-parity proof, was recorded on a Google Pixel 10 Pro XL (Android API 37), using GPU FP32 in the order `ke_enc0.tflite` → `ke_enc1.tflite` → `ke_enc2.tflite`. The observed hashes were `ce32c38c94df957a83c2ae805c2f04380e8429a2b0114b89c73bbceb425cccaf`, `18d1ffb8d4ab945c93a719fbd99315fce877e684bfe7eef1714a30f1a8768ce0`, and `d48816e2d55c2e52cf6bd4c8c988736ea824065b185fb3922b727056d45cd801`, respectively. The final tensor was `[1,512,7680]` (3,932,160 finite elements). Preprocessing took 1,231 ms; the graphs took 11,450, 10,185, and 11,377 ms; total time was 38,190 ms. Approximate process PSS changed from 396,555 to 442,092 kB and thermal status remained 0 → 0.

## Phase 2D reference VAE checkpoint

Phase 2D uses the `litert-community/FLUX.2-klein-4B-LiteRT` model at immutable revision `f9b9171c841790a39147903febe73a85e9eaf42e` and the companion `google-ai-edge/litert-samples` PR 227 commit `f48a89e4f29a74ab51f29c311ac7a0e5e479d225`. The latter establishes the graph contract: exactly one row-major FP32 NCHW RGB `[1,3,256,256]` input and exactly one FP32 `[1,32,32,32]` output. `kv_vae_enc.tflite` returns the distribution mode/mean directly.

### Deliberate Android application preprocessing contract

This application owns the deterministic preprocessing policy. It is derived from the companion Android execution path but is **not claimed to be pixel-identical** to the Python preparation script's PIL bicubic resize; upstream provided no official preprocessing/VAE numeric parity tolerance.

1. A single selected `content://` image is opened only through `ContentResolver`. Bounds are decoded first and a power-of-two sample is chosen while retaining at least 256 pixels on each side. Estimated sampled dimensions and ARGB bytes are computed with checked `Long` arithmetic; inputs whose estimated sampled decode exceeds the conservative 64 MiB Java bitmap limit are rejected.
2. AndroidX ExifInterface 1.4.1 reads orientation from a separately opened stream. All eight EXIF orientations (including mirrored forms) are applied exactly once before cropping; absent/undefined orientation is normal.
3. Pixels are straight-alpha composited onto opaque black with `round(channel × alpha / 255)`, implemented as `(channel × alpha + 127) / 255` integer arithmetic.
4. After orientation, `side = min(width,height)`, `left = (width-side)/2`, and `top = (height-side)/2`. Odd excess pixels remain at the right or bottom.
5. The square is resized by `Bitmap.createScaledBitmap(square, 256, 256, true)`. This Android filtered scaling—not PIL bicubic—is canonical for the app.
6. Opaque pixels are emitted as red plane, green plane, then blue plane, row-major within each plane, and normalized as `channel / 127.5f - 1.0f`. Exactly 196,608 finite values in `[-1,1]` are required.

The Java-side VAE input is 786,432 bytes and the raw output is 131,072 bytes. Each 256×256 ARGB bitmap and the temporary pixel array are 262,144 bytes. The sampled bitmap varies with aspect ratio and is accepted only when its checked estimate is at most 64 MiB; therefore Phase 2D makes no fixed Java peak claim. The diagnostic reports sanitized sampled dimensions and estimated sampled ARGB bytes. These calculations do not claim native or GPU peak memory.

The debug build exposes **Developer verification — Reference VAE only** and reuses the editor selection. It validates files under the shared model-file lock, hashes the graph locally, preprocesses off the main thread, runs only `kv_vae_enc.tflite` through GPU FP32, and runtime-validates exactly 32,768 finite output elements. The current `FluxGraphResult` exposes arrays but no tensor metadata, so `[1,3,256,256]`/`[1,32,32,32]` and FP32 are pinned export contracts rather than runtime-observed metadata. It reports sanitized timing, heap/PSS, thermal, device/API, shapes, and the locally observed hash. Cancel uses coroutine cancellation. EXIF/bounds/decode streams, decoded/oriented/cropped/opaque/scaled bitmaps, LiteRT input/output buffers, the compiled model, and the environment are deterministically released on success, error, OOM, and cancellation. Release source code renders no verification control.

Physical Pixel verification remains required for all eight orientations with known images, crop inspection, GPU compilation, `[1,32,32,32]` finite output, runtime, memory, thermal behavior, cancellation, and the locally observed graph hash. Structural validation does not establish semantic editing quality.

Patchification, latent batch normalization, `[1,256,128]` tokens, image IDs, reference/noise concatenation, `kce_*`, scheduling, denoising, VAE decoding, and image generation are deferred to Phase 2E or later. **Generate remains unconditionally disabled.**

## Phase 2E reference-token preparation

Phase 2E is based on evidence commit `94a4d9d537876150ec7e7e4f16a4c9a696b8ed1d` and the
immutable base-model revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`. The authoritative
companion is `google-ai-edge/litert-samples` commit
`f48a89e4f29a74ab51f29c311ac7a0e5e479d225`, specifically
`compiled_model_api/text_to_image/flux2_klein_kotlin_gpu/conversion/gen_prep_klein.py` (SHA-256
`1f2b3d902d4f36281e61447d86331e08bd1c61f20f9034896817c086ae1ab61d`) and
`compiled_model_api/text_to_image/flux2_klein_kotlin_gpu/android/app/src/main/java/com/google/ai/edge/examples/flux2_klein/Flux2KleinGenerator.kt`.
The pinned generator obtains patchification by probing `_patchify_latents` with a flat arange tensor
and saving the result as a gather map. The pinned Android example gathers with that map, normalizes
packed channel planes, and transposes channel-major packed storage to token-major storage. No
behavior was inferred from mutable upstream or a generic FLUX implementation.

The committed evidence manifest and binaries are staged unchanged by Gradle into generated build assets under `flux/reference/`, then parsed and validated before use. No runtime constant is duplicated in the source asset tree. The manifest pins epsilon
`0.0001`, source `[1,32,32,32]`, packed `[1,128,16,16]`, and final `[1,256,128]` shapes. Its exact
formula is `(packed[channel] - running_mean[channel]) / sqrt(running_variance[channel] +
batch_norm_eps)`. `bn_std.bin` already stores that square root, so Android applies exactly
`(packedValue - mean[packedChannel]) / std[packedChannel]`: it does not apply epsilon or square root
a second time, gamma/weight is not applied, and beta/bias is not applied.

* `bn_mean.bin`: 512 bytes, 128 little-endian FP32 values, SHA-256
  `9027fac5727854f779ebbeae3032cfce0d47a11bc85f4329eb0a317c9ad90217`.
* `bn_std.bin`: 512 bytes, 128 finite positive little-endian FP32 values, SHA-256
  `e89b48bf701b864cc6cad73070e0e49052ee673d2c34ec2084ecf6386284d199`.
* `patch_perm.bin`: 131,072 bytes, 32,768 little-endian signed int32 values forming the exact
  bijection `0..32767`, SHA-256
  `90c531082fddef4309f5ba43b8c898c823f049d2aa8951fb84e9cc4395942c3d`.

For each packed output index Android performs the authoritative gather
`packed[outputIndex] = source[patchPerm[outputIndex]]`; it does not replace the evidence with a
shape-derived formula. It normalizes each `[128,16,16]` channel plane and writes final storage as
`tokens[spatialIndex * 128 + packedChannel]`, producing 256 tokens of width 128. The typed latent and
token boundaries copy caller arrays and reject wrong counts or non-finite values. Preparation runs
off the UI thread and checks cooperative cancellation before each phase, every 1,024 elements while
gathering and normalizing, before final validation, and before return. Full-size arrays remain local
to verification and never enter Compose state; graph/environment/staged-image ownership continues
to use deterministic `finally`/scoped cleanup.

Debug builds expose **Developer verification — Reference VAE and tokens only**. It stages and
preprocesses one image, executes `kv_vae_enc.tflite` exactly once on GPU FP32, validates the typed
latent, strictly loads the constants, builds and validates tokens, emits only sanitized shapes,
counts, timings, memory/thermal/device data and locally observed graph identity, then releases local
resources. Release source exposes no verification action. Evidence consistency proves the shipped
files and JVM transformation agree with the pinned record; physical-device compatibility remains a
separate Pixel verification requirement.

### Phase 2D locally observed Pixel result

A successful local observation—not publisher verification or Phase 2E device proof—was recorded on
a Pixel 10 Pro XL running Android API 37. `kv_vae_enc.tflite` ran with GPU FP32; its locally observed
SHA-256 was `a9d7435e01f1266d8024a887c7bfbe0f58736436e9c10f8566ee2561c5db9df6`. The FP32 input was
`[1,3,256,256]`; the FP32 output was `[1,32,32,32]` with 32,768 finite values. The sampled decode was
640×360 (921,600 bytes), preprocessing took 139 ms, graph execution took 3,486 ms, and total time was
3,680 ms. Approximate process PSS was 1,029,919 kB and thermal status remained 0 → 0. Approximately
1.03 GB PSS is substantial and must be monitored before later transformer graphs are loaded.

Image position IDs, noise creation, reference/noise concatenation, timestep/guidance tensors,
`kce_*`, scheduling, diffusion/denoising, `kv_vae.tflite` decoding, output bitmaps, CPU/FP16/cloud/NPU/
Tensor TPU fallback, and image generation remain explicitly deferred. **Generate remains
unconditionally disabled in debug and release builds.**
