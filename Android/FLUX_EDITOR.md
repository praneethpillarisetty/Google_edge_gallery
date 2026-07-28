# FLUX.2 Klein LiteRT image editor

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
