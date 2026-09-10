# Kusama v2: frozen creative studio

Start with [project direction, current evidence and next milestones](docs/research/project_direction.md) for how this implementation relates to the ambition of superhuman artistic abilities. [AGENTS.md](AGENTS.md) gives coding agents the working rules; the [version 2.1 source specification](docs/research/architecture_spec_v2.1_source.md) preserves the research design. The current system runs artistic experiments with frozen models; learned artistic improvement and superiority to human workflows remain unproven.

This Python 3.11 project connects frozen `Qwen/Qwen3-VL-8B-Instruct` direction with frozen `black-forest-labs/FLUX.2-klein-base-4B` generation and editing. It saves reference inputs, proposals, images, lineage, resource usage and monetary accounting. Humans separately review finished quality and development potential and explicitly choose what advances. No model was trained in the recorded runs.

**A full artistic run begins with a sufficiently complex artistic problem statement:** a grounded question with consequential stakes, unresolved tensions and multiple possible artistic responses. The [problem-statement guide](docs/research/artistic_problem_statements.md) establishes the user-supplied Wangechi Mutu primary case study and the future goal of a model that can construct adequate statements. Operators supply this starting point today; automatic adequacy assessment and a dedicated formulation model remain future work. Image generation and edit diagnostics are component experiments within the broader research effort.

The [source-connected studio](docs/workflows/source_connected_studio.md) adds bounded official collection ingestion, whole/detail retrieval, source-grounded combinations, realized branching, local revisions and follow-up review. Use the `kusama` command for this workflow. The earlier [reference-informed workflow](docs/workflows/reference_loop.md) still accepts curated packets and generates sibling realizations. The [reference library](references/README.md) preserves its initial Mutu process and Maya memory packets. Automated collection retrieval remains bounded; it is not comprehensive cultural research or learned reference selection.

The [independent research workflow](docs/workflows/independent_research.md) lets frozen Qwen choose searches, inspect images and contextual text, revise questions, reject leads and commit hypotheses with variable reference sets. It includes a reusable metadata catalog, bounded source acquisition, an explicit FLUX rendering bridge, research during development and an inspectable dashboard history. No trained evaluator is needed to make and review studies. These are implemented capabilities verified with substitutes; their artistic usefulness still needs real runs and human inspection.

New full panel runs now [make provisional studies before artistic commitment](docs/operations/2026-09-09_studies_before_commitment.md): at least three distinct studies, inspection of actual images, a model-attributed choice, and one optional follow-up with another inspection. These workflow requirements do not establish artistic progress.

The main panel now supports [image-led development cycles](docs/operations/2026-09-09_development_cycle.md): after inspecting three provisional studies, Qwen can reopen source research, revise or combine any saved branches, reformulate, or pause for human review within displayed limits. Every new image is inspected before the next decision. This workflow flexibility has not yet demonstrated an artistic advantage.

New research and bounded-development runs also offer Qwen [image prompts](docs/workflows/image_prompts.md): it can attach inspected reference images directly to FLUX, alongside its text instruction and the existing artwork for edits. This is optional and preserves exact source hashes and input order; artistic benefit remains unverified.

The original three-call image/edit experiment remains available below. Saved technical success is separate from artistic success, which requires looking at the images and reviewing them.

New panel research also offers [up to two chance source leads](docs/operations/2026-09-09_random_exposure.md), sampled with a saved seed alongside the model's directed inquiry. The model can discard them. Exposure adds no novelty reward, and its artistic usefulness remains unverified.

The [final overnight handoff](docs/operations/2026-09-08_overnight_handoff.md) records 38 new rendered images, three complete E0 development pools, the expired inference window and zero human votes. The final software suite passed 706 tests; [implementation status](docs/operations/implementation_status.md) separates those checks from real inference and artistic evaluation.

## Documentation

The [documentation index](docs/README.md) organizes the lab's records into `research/`, `models/`, `data/`, `evaluation/`, `experiments/`, `workflows/` and `operations/`. Use it to find design proposals, model and data status, evaluation protocols, recorded results, executable guides and dated handoffs.


The [bounded development workflow](docs/workflows/bounded_development.md) adds `kusama edit-benchmark` and `kusama develop`: saved-failure measurements, explicit edit controls and a bounded frozen-model proposal/observation/selection loop. Model choices remain separate from human judgment and training. Planning never creates an inference resource session.

## Setup from a fresh checkout

Run these commands in the repository directory. If `uv` is missing, install it using the [official uv installer](https://docs.astral.sh/uv/getting-started/installation/):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the terminal if the installer requests it, then:

```sh
uv python install 3.11.15
uv sync --locked --dev
cp .env.example .env
uv run --locked python scripts/first_edit.py --check-environment
uv run --locked python scripts/first_edit.py --help
```

`uv.lock` records the dependency resolution; use `--locked` to avoid changing it. Model weights download separately on the first inference run. The current FLUX cache occupies approximately 15 GiB and the separate Qwen director cache approximately 16 GiB. Allow more than their combined 31 GiB for the complete workflow, plus the Python environment, temporary downloads and saved images. The earlier 20 GB allowance covered only FLUX. A Hugging Face token is optional for public model access; configure `HF_TOKEN` locally if your access requires one.

## Run the experiment

```sh
uv run --locked python scripts/first_edit.py
```

The baseline is 1024 × 1024, 50 inference steps, guidance 4.0, and BF16 where the selected device supports it. The parent uses seed 17; edits use seeds 23 and 41 with identical settings. Each edit loads the same saved parent, so neither edit is an input to the other. The default plan is exactly three image calls and no retries.

The parent prompt requests a collage-style mountain landscape with a visible solitary person near the mountain's base. The edit instruction requests reducing the person to about one-third of their height with their feet anchored, while preserving mountain shape, framing, colors, and lighting. The full prompt and instruction are recorded in each manifest.

Each run creates a unique directory beneath `artifacts/` containing:

```text
parent.png
edit_seed_23.png
edit_seed_41.png
comparison.png
manifest.json
```

The manifest records configuration, resolved checkpoint revision, seeds, parent relationships, file hashes, dependency versions, timings, available memory measurements, and failures. Partial runs retain completed images and the manifest. The comparison includes all available images; a partial comparison is explicitly marked in the manifest. There is no automatic human judgment or claim that a successful image call followed the visual instruction.

Repeat the edits with an existing parent, preserving that starting point:

```sh
uv run --locked python scripts/first_edit.py \
  --parent-image artifacts/REPLACE_WITH_RUN_ID/parent.png
```

Replace `REPLACE_WITH_RUN_ID` with an actual completed run directory. This applies EXIF orientation, converts to RGB, and saves a PNG copy in a new run before making only the two edit calls. It records both the source-file hash and saved-parent hash. For a parent with other dimensions, supply matching `--width` and `--height` (multiples of 16); it is never silently resized. The generation prompt and seed of an imported image are unknown and are not claimed as that image’s provenance. Inspect the original and both alternatives for person size, foot position, unrelated changes, and the perceived scale of the mountain.

## Configuration

For the original `kusama-first-edit` CLI, copy `.env.example` to `.env` and edit it locally; command-line arguments override environment configuration. The `kusama-creative` runner uses its explicit arguments and defaults and does not load these `.env` settings. Relative paths resolve from the working directory.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `KUSAMA_CACHE_DIR` | `.cache/huggingface` | Model download/cache directory |
| `KUSAMA_OUTPUT_DIR` | `artifacts` | Parent directory for unique run directories |
| `KUSAMA_DEVICE` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `KUSAMA_PRECISION` | `auto` | `auto`, `bf16`, `fp16`, or `fp32` |
| `KUSAMA_OFFLOAD` | `none` | `none`, `model`, or `sequential`; offloading requires an accelerator |
| `KUSAMA_VAE_TILING` | `false` | Reduce VAE image-processing memory by tiling |
| `KUSAMA_MODEL_REVISION` | `a3b4f4849157f664bdbc776fd7453c2783562f4d` | Immutable model revision |
| `KUSAMA_LOCAL_FILES_ONLY` | `false` | Require already-cached checkpoint files |
| `HF_TOKEN` | empty | Optional Hugging Face access token; never written to provenance |

Run `--help` for the supported values and full CLI. The matching CLI flags are `--cache-dir`, `--output-dir`, `--device`, `--precision`, `--offload`, `--vae-tiling` / `--no-vae-tiling`, `--model-revision`, and `--local-files-only` / `--no-local-files-only`. Additional experiment controls are `--prompt`, `--edit-instruction`, `--parent-seed`, `--edit-seeds 23 41`, `--width`, `--height`, `--steps`, `--guidance`, `--max-sequence-length` (default 512), `--parent-image`, `--max-retries` (0–2, default 0), and `--note`.

For example, a deliberately reduced-resolution CUDA run with model CPU offload and VAE tiling is:

```sh
uv run --locked python scripts/first_edit.py \
  --device cuda --offload model --vae-tiling \
  --width 512 --height 512 \
  --note "Reduced resolution and CUDA CPU offload to fit available memory"
```

This changes the proposal's resolution, and the manifest records that configuration. Diffusers supports `--offload model` and `--offload sequential` on CUDA and MPS, with a potentially large runtime cost for sequential offload. MPS uses Apple unified memory; offloading still reduces the set of tensors allocated to the GPU. Sequential offloading has not yet been validated with the complete model here. CPU execution is supported for an explicitly chosen CPU device but may be very slow and require substantial RAM.

## Verification and compatibility

The local target is macOS 26.5.2 on an Apple M4 Max with 36 GB unified memory. Installation with `uv sync --locked --dev` succeeded using uv 0.11.15 and managed Python 3.11.15 (pinned in `.python-version`). Initial import checks also passed on system Python 3.11.4, but its standard-library TLS certificate store was incomplete; the project was rebuilt with the managed runtime before final verification. All direct runtime dependency imports, `Flux2KleinPipeline`, `AutoTokenizer`, and `Qwen3ForCausalLM` originally imported successfully without torchvision or sentencepiece. The frozen Qwen vision-language director now adds pinned torchvision 0.29.0 for its processor. PyTorch 2.14.0 reported MPS available, and an actual MPS BF16 tensor operation passed. Its recommended maximum MPS working set was 30,150,672,384 bytes (about 28.1 GiB); this is not a measurement of model inference memory.

Direct dependencies are pinned in `pyproject.toml`, with transitive dependencies in `uv.lock`. This machine has no NVIDIA CUDA device; CUDA execution and NVIDIA driver compatibility remain unverified. The Linux lock resolves the PyTorch 2.14.0 CUDA 13.0 dependency stack. NVIDIA documents driver branch r580 or newer for CUDA 13.x; see the [CUDA 13.0.3 release notes](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html). A CUDA runtime alone is not a driver, and the script performs a real device arithmetic probe before downloading weights. On Linux with NVIDIA hardware, `--check-environment` reports the installed PyTorch CUDA runtime and available GPU before inference. The complete three-call MPS BF16 experiment also passed execution and artifact verification. Real runtimes were 8.40 minutes for parent generation, 19.49 minutes for seed 23 and 21.02 minutes for seed 41. Peak sampled MPS driver allocation was 13.12 GiB. Both edits failed parts of the visual instruction; see [verification evidence](docs/experiments/verification.md) and [the visual review](docs/experiments/visual_review.md). Full-model execution with other devices, precisions, or offloading settings remains unverified.

For this 36 GB Mac, use the recorded memory settings for the complete experiment:

```sh
uv run --locked python scripts/first_edit.py --device mps --offload model --vae-tiling
```

Run the focused software checks with:

```sh
uv run --locked pytest
```

These checks exercise software behavior with small local fixtures and controlled pipeline substitutes. They do not establish model access, real inference, or visual instruction-following. The historical 04:17 UTC audit on 8 September 2026 passed **402 tests**, including the frozen director, reference loop, recovery, cost accounting and live human-review interface (two dependency deprecation warnings). Real inference and saved artifacts are checked separately; see [the original execution evidence](docs/experiments/verification.md) and [the creative experiment](docs/experiments/creative_verification.md).

The normal pytest run also checks that dashboard TypeScript declarations match the Python contracts. After intentionally changing a server schema, regenerate the declarations with `uv run --locked python scripts/generate_contract_types.py`, then rerun the relevant tests. To check only declaration drift, use the same script with `--check`; it does not rewrite files.

## Troubleshooting

- **Python or imports missing:** run `uv python install 3.11.15`, then `uv sync --locked --dev` from the repository directory. Use `uv run --locked` rather than an unrelated system Python.
- **Model download or access fails:** check connectivity, disk space, the pinned revision, and any needed `HF_TOKEN`. With `KUSAMA_LOCAL_FILES_ONLY=true`, the complete checkpoint must already exist in the configured cache. Failure information is preserved in the run manifest.
- **Device unavailable:** inspect `--check-environment`, then choose a device actually reported as available. NVIDIA execution needs a working driver compatible with the installed PyTorch CUDA runtime. Apple Silicon uses `--device mps`; a CUDA option will not enable NVIDIA support there.
- **Out of memory:** completed artifacts remain available. Close competing applications; try `--vae-tiling`, `--offload model` on CUDA or MPS, or an explicitly recorded smaller resolution. Lowering step count primarily affects runtime, and is not a dependable fix for model memory use. Start a new run, optionally using `--parent-image`, rather than overwriting prior evidence.
- **Precision rejected:** use `--precision auto`, or explicitly choose a precision supported by the selected device. The effective precision is recorded in provenance.
- **Images do not follow the instruction:** retain both outputs. Compare the person's size and foot position, and inspect changes in the mountain, framing, colors, and lighting. A model can execute correctly while producing a poor edit; the comparison and manifest are the evidence for diagnosing that outcome.

## Project layout and sources

`scripts/first_edit.py` is the executable entry point. `src/kusama/` contains configuration, inference orchestration, and artifact/provenance helpers. `tests/` contains focused integration checks. `artifacts/` and model caches are untracked. The installed console command `uv run --locked kusama-first-edit` invokes the same experiment. `docs/experiments/verification.md` records the measured integration result, and `docs/experiments/visual_review.md` separates visual instruction-following from software success.

The FLUX implementation follows the [official model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) and [Diffusers 0.40.0 FLUX.2 documentation](https://huggingface.co/docs/diffusers/v0.40.0/en/api/pipelines/flux2). The original `kusama-first-edit` experiment remains limited to parent generation, two independent image-conditioned edits, and inspectable provenance. The controlled executor, studio, and frozen director extend that foundation; no model training is performed.

## This session's saved run

The comparison and manifest are in `artifacts/20260907T190926.622194Z_de125bc5/`. The parent and both actual model edits are retained there. Seed 23 reduced height to roughly two-thirds and moved the person; seed 41 preserved height and approximate foot position. Neither achieved the full requested intervention.

Repeat only the edits from this exact saved parent on the tested Mac configuration:

```sh
uv run --locked python scripts/first_edit.py \
  --device mps --offload model --vae-tiling \
  --parent-image artifacts/20260907T190926.622194Z_de125bc5/parent.png
```

The run directories are local generated artifacts and are not included in Git. Keep each run directory intact when copying or sharing it; image paths in the manifest are relative to that directory.

## Controlled editing milestone

Three editing methods, explicit masks and plans, bounded fallbacks, and independent preservation checks are available through `kusama-edit`. See [the controlled editing guide](docs/workflows/controlled_editing.md) for the plan contract, example command, artifacts and review limitations.

## Independent checks and benchmark

The benchmark includes measured output geometry, duplicate/remnant checks, tighter repair masks, and seven calls covering resizing, removal and recoloring. See [verification and benchmark instructions](docs/evaluation/verification_benchmark.md). Run all software checks with `uv run --locked pytest -q`.

## Development dashboard

```sh
uv sync --locked --dev
uv run --locked kusama-dashboard
```

Open [the local dev panel](http://127.0.0.1:8765/). The home screen centers on writing your own open-ended question and pressing **Start run**, with explicit research, image, time and online-access limits. Follow each run in one newest-first update stream combining research, images, development, execution, files and logs; inspect saved prompts, source evidence, lineage, costs and failures, or stop before its next action. Development, Sources, Research, and Tools & system keep the existing workflows accessible: switch artifact stores, prepare research/development plans, control selected-parent continuations, and inspect resource sessions, memory and API schemas. CLI workflows have copyable commands and local guides. Run/workflow comparators and the A/B review UI have been removed; historical artifacts remain preserved. See [the panel guide](docs/workflows/dev_panel.md).

Use `--artifacts path/to/runs` or `--port 8877` to change the source directory or port. Reviewer names identify local records; they are not authentication accounts.

## Reference-informed experiment

```sh
uv run --locked kusama-creative --packet references/maya_memory/packet-v3.json \
  --directions 12 --realizations 2 --observe \
  --width 512 --height 640 --steps 50 \
  --device mps --precision bf16 --offload model --local-files-only
```

This explicitly uses smaller study images for comparing ideas, with fifty diffusion steps. Higher-resolution development follows human selection. Qwen sees the packet's images and sourced text; FLUX receives the exact standalone prompts Qwen writes. The models load sequentially to fit local memory. All outputs and source relationships remain inspectable. See [resume and stage controls](docs/workflows/reference_loop.md) and [cost accounting](docs/operations/cost_accounting.md). Local API fees are zero; electricity and hardware totals remain unknown until sourced rates are supplied.

## Resumable jobs

Controlled edits and benchmarks support stable `--job-id` values, frozen inputs, SQLite budgets, and `--resume`. Saved model outputs are reused; a reserved inference with no surviving output remains uncertain. See [the recovery guide](docs/operations/recovery.md) for setup and interruption behavior.

```sh
uv run --locked kusama-edit --resume artifacts/recovery-integration-20260907
```

The directory must be an existing registered job. Legacy first-edit runs and older runs without a job ledger are not resumable.

## Studio projects and bounded iteration

`uv run --locked kusama-studio --help` exposes persistent inquiry/reference state, action and branch history, and bounded proposal → execute → inspect → select → continue/stop controls. The basic dashboard shows project status and branches. See [the studio workflow](docs/workflows/studio.md) for the included three-direction example and recovery behavior. The separate `kusama-direct` command brings frozen Qwen into a bounded, single-edit experiment; studio selection remains explicit.

## Frozen Qwen director

`Qwen/Qwen3-VL-8B-Instruct` receives the saved image and artistic intention, proposes one concrete edit and its expected effect, and supplies the instruction to one FLUX edit. Both models use their existing weights without training. This vision-language director is separate from FLUX's existing Qwen text encoder.

```sh
uv run --locked kusama-direct \
  --parent-image artifacts/20260907T190926.622194Z_de125bc5/parent.png \
  --intention 'Make this landscape feel overwhelmingly vast' \
  --device mps --offload model --vae-tiling
```

Qwen is released before FLUX loads. The run saves the original image, Qwen's proposal and provenance, the exact edit instruction, and FLUX's result and comparison for human judgment. See [the frozen director workflow](docs/workflows/frozen_director.md) for staged execution, verification boundaries, and the path from one reviewed edit to 12 directions × 2 realizations. The first real Qwen-directed result is saved in `artifacts/20260908T000653.098912Z_qwen_719e9b03/`; one Qwen call and one FLUX call completed, and all hashes and recovery were verified. The user's first judgment is uncertain: the images feel somewhat similar in vastness. Finished quality and development potential remain unassessed. See [the measured result and human feedback](docs/experiments/director_verification.md).

The second experiment tests a one-third-size figure using a guided Qwen instruction, deterministic geometry, and one FLUX background repair. It preserved all 1,031,297 protected pixels and completed in about 98 seconds for the FLUX job. The user confirmed that the smaller figure changes the feeling of vastness; finished quality and development potential remain unassessed. See [the exact process, saved result, and human feedback](docs/experiments/scale_experiment.md).
