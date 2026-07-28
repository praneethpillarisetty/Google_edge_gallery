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

Phase 2 will implement tokenizer execution, image preprocessing, the LiteRT GPU inference pipeline,
and output decoding. Until then the Generate control remains disabled and is accompanied by a clear
explanation that generation will be available in Phase 2.
