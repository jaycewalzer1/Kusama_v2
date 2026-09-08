# First milestone verification

Verification date: 7 September 2026. This file separates software checks from actual image-model execution and visual assessment.

## Software and environment

- Apple M4 Max, 32 GPU cores, 36 GiB unified RAM, macOS 26.5.2 / Metal 4. No NVIDIA GPU or CUDA driver is available in this session.
- uv 0.11.15; managed CPython 3.11.15. The initial system Python 3.11.4 had an incomplete standard-library TLS certificate store; the project was rebuilt with uv's managed runtime.
- Locked PyTorch 2.14.0, Diffusers 0.40.0, Transformers 5.16.1, Accelerate 1.14.0 and matching transitive dependencies installed successfully.
- Required pipeline, Qwen text encoder and tokenizer imports passed. No torchvision or sentencepiece is required by this pipeline's selected components.
- Actual MPS BF16 arithmetic passed. Native MPS recommended maximum working set was 30,150,672,384 bytes, which is a hardware/driver limit rather than an inference-memory measurement.
- Accelerate model and sequential CPU-offload hooks passed small MPS operation probes. These probes do not establish full-model memory fit or image quality.
- The exact documented Python install, locked dependency sync, `.env` copy, environment check, help and test commands passed in a separate temporary fresh project copy.
- **54 focused tests passed** in that fresh environment. Tests cover real PNG persistence with explicitly synthetic fixtures, saved-parent input equality, lineage and hashes, identical edit settings, seed handling, bounded retries, partial failures, prompt preflight, immutable revision validation and environment precedence. They never substitute for real FLUX image results.
- A real CLI request for unavailable CUDA exited with code 1 and an actionable error, preserved a failure manifest, and made zero model image calls.

## Execution ledger

| Run | Purpose and status | Actual image calls |
| --- | --- | --- |
| `20260907T190519.609865Z_a50a4268` | Initial setup interrupted during download so provenance fixes and managed Python could be used. Manifest retained. | 0 |
| `20260907T190926.622194Z_de125bc5` | Completed full development experiment with MPS BF16, model CPU offload and VAE tiling. | 3, all returned images; no retries |
| `20260907T191303.223786Z_3be50abf` | Expected unavailable-CUDA error check; failed before model loading/inference. | 0 |

Manifests are located under `artifacts/<run>/manifest.json`. The first interrupted setup sampled CPU memory during loading because its load-device provenance bug was one of the fixes being verified; no GPU load peak is claimed for that attempt. Its model revision remains available in `backend.model_revision`; later runs also populate `model.resolved_revision`.

The full experiment retains the requested 1024 × 1024 resolution, 50 steps, guidance 4.0, parent seed 17 and independent edit seeds 23/41. Offloading and VAE tiling are explicit memory settings recorded in the manifest. Model revision: `a3b4f4849157f664bdbc776fd7453c2783562f4d`.

## Image evidence and visual assessment

All three actual model PNGs were opened and visually inspected. They decode at 1024 × 1024, and their SHA-256 hashes match the manifest. Both edits have identical effective settings, both input records match the saved parent hash, and exactly three real pipeline calls occurred with seeds 17, 23 and 41. The comparison's three panels match the original image pixels at 1:1 scale. Its hash was checked after correcting a missing em-dash glyph in the labels; the initial comparison is also preserved and that formatting correction is recorded under `postprocessing`.

| Call | Synchronized runtime | Sampled maximum MPS driver allocation |
| --- | --- | --- |
| Parent, seed 17 | 504.23 s (8.40 min) | 9.175 GiB |
| Edit, seed 23 | 1169.52 s (19.49 min) | 13.083 GiB |
| Edit, seed 41 | 1261.26 s (21.02 min) | 13.120 GiB |

Loading, including the initial download in this run, took 1191.80 s (19.86 min). Total run time was 4130.26 s (68.84 min), including loading, inference and artifact writing. GPU timings include synchronization after asynchronous Metal work; the progress-bar time alone undercounts execution. MPS values are sampled maxima at 0.25-second intervals, not native exact peaks. The manifest also retains MPS current allocation, process RSS and host-memory measurements.

**Visual outcome:** seed 23 makes the person roughly two-thirds as tall but shifts them right and changes unrelated content. Seed 41 leaves person height essentially unchanged while approximately preserving the foot position. Both redraw detail and alter colors/lighting. Neither satisfies the full instruction. These outputs are retained with no quality-based retry. See [the separate visual review](visual_review.md) for each constraint and the diagnosis.

Visual findings are stored in `observations.model_observations`, with image hashes, method, timestamp and uncertainty. `observations.human_labels` remains empty. The mountain feels somewhat larger in seed 23 to the assistant reviewers; this is a visual assessment requiring the user's judgment. Seed 41 provides no clear scale effect from person reduction.

Artifacts: `artifacts/20260907T190926.622194Z_de125bc5/{parent.png,edit_seed_23.png,edit_seed_41.png,comparison.png,manifest.json}`. `execution.log` and the initial comparison are retained in the same run. These local generated artifacts are intentionally excluded from Git.

## Compatibility limits

CUDA inference and NVIDIA driver compatibility could not be tested on this Mac. The lock's Linux dependency set uses CUDA 13.0; verify the actual target with `--check-environment` and the [NVIDIA compatibility documentation](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html). CPU and full-model sequential-offload inference are unverified. Reproducibility means retaining inputs and configuration; identical seeds do not guarantee byte-identical outputs across platforms.
