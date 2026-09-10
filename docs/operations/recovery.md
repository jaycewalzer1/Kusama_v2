# Resumable controlled edits and benchmarks

New `kusama-edit` and `kusama-benchmark` jobs persist their execution state and model-call budget in SQLite. A stable job identity, frozen input snapshots, registered artifact hashes, and saved attempt checkpoints allow interrupted work to continue without repeating completed image generation.

The original `kusama-first-edit` / `scripts/first_edit.py` workflow has **no resume support**. Older controlled runs without `job.json` and a corresponding ledger record remain inspectable in the development dashboard, but cannot be retroactively resumed. No historical run or benchmark result is changed by installing these features.

## Commands

Run commands from the repository directory after `uv sync --locked --dev`. Validate a plan without model inference:

```sh
uv run --locked kusama-edit --plan examples/controlled/mountain.json --validate-only
uv run --locked kusama-benchmark --suite examples/benchmark/suite.json --validate-only
```

Start a controlled edit with an explicit job ID and output directory:

```sh
uv run --locked kusama-edit \
  --plan examples/controlled/mountain.json \
  --job-id mountain-recovery-01 --output-dir artifacts \
  --device mps --offload model --vae-tiling --local-files-only
```

That command performs real inference if the job has unfinished, unreserved work. It uses the existing approved mountain parent referenced by the example plan, so that source artifact and the pinned local model cache must exist. The job directory is `artifacts/mountain-recovery-01/`; its database is `artifacts/jobs.sqlite3`.

Resume using the saved directory:

```sh
uv run --locked kusama-edit --resume artifacts/mountain-recovery-01
```

Start and resume a benchmark with the same pattern:

```sh
uv run --locked kusama-benchmark \
  --suite examples/benchmark/suite.json \
  --job-id repair-pilot-01 --output-dir artifacts \
  --device mps --offload model --vae-tiling --local-files-only
uv run --locked kusama-benchmark --resume artifacts/repair-pilot-01
```

The benchmark start command performs real inference, subject to the suite's seven-call limit and each plan's own limit. These command examples describe execution; they are not evidence that the new recovery implementation has completed a GPU run.

Use `--resume` on its own, without a new job ID, suite, or backend settings. It restores the recorded configuration, including device, model revision, paths, and generation settings. Controlled-edit resume also accepts `--plan` as an optional consistency check; any supplied plan and its input hashes must exactly match the frozen source. A benchmark resume rejects a supplied `--suite`.

Job IDs accept 1–128 letters, digits, underscores, or hyphens. Omitting `--job-id` generates a new timestamped identity. Repeating a start command with the same ID is idempotent only when its immutable plan/suite, input hashes, source paths, configuration, output directory, budget, and parent linkage still match. A mismatch is rejected. Once snapshots are complete, `--resume` can work without the original source files; repeating the original start command still reads those source files during validation.

## Recovery decisions

| Persisted state at interruption | Resume behavior |
| --- | --- |
| Inputs are not fully snapshotted | Recheck the original sources against their frozen hashes and finish copying them. Changed or unavailable sources block recovery. |
| Attempt interrupted before inference reservation | Continue the unfinished attempt; it may invoke the model once after reserving budget. |
| Inference reserved, but no saved raw output exists | Retain the charge, mark the interrupted attempt and job `uncertain`, and do not automatically invoke that attempt again. |
| Raw output saved before interruption | Register/verify the retained output, finish deterministic composition and remaining verification, and reuse the model-call charge. No editor backend is constructed for this recovery. |
| Cleanup, composition, or other postprocessing failed after raw output was saved | Keep `postprocessing_failed` state and the raw output. Resume retries downstream work from the retained bytes. |
| Final `needs_review`, `rejected`, `budget_exhausted`, or `uncertain` job | Verify stored artifacts and regenerate derived metadata as needed; do not generate another image. |
| Benchmark summary or contact-sheet rendering failed | Keep terminal execution metadata and reporting errors. Resume rebuilds reports and reuses completed child outputs. |
| Registered artifact is missing or changed | Refuse recovery; restore the original evidence bytes. |

Ordinary caught preparation/inference failures remain recorded as failed attempts; the executor may advance only to an already-declared fallback with remaining budget. It does not retry the same failed invocation automatically. A crash can leave uncertainty about whether a reserved inference actually reached the model, so a reservation counts as a possible call, even when no output survives. There is no automatic refund, reset, or override for that uncertainty. A deliberately new run requires a new job identity and budget.

Verification can execute during recovery when it was not previously completed and registered. Saved verification is reused only when it is registered and bound to the retained result's hash. It remains model or program evidence; it does not become a human acceptance judgment.

## Frozen inputs and shared budgets

Each controlled job stores the original `plan.json`, a self-contained `execution_plan.json`, its approved parent image, and masks. The parent snapshot preserves the exact approved source bytes, including encoding and metadata. Mask normalization and derived regions are saved separately. Configuration and original source hashes are included in the immutable job specification.

A benchmark validates and snapshots **all** case inputs before its first child can spend budget. It creates a stable child job and records its directory in the suite report before executing that child. Children are sibling run directories under the output directory, linked through the ledger; they are not nested beneath the suite directory.

An inference reservation atomically charges both the child job and every ancestor budget. Concurrent children cannot overspend the shared suite allowance. Repeating the same reservation does not create another charge and does not authorize another inference. A nonblocking filesystem lock prevents two executors from running the same job concurrently; it is released by the operating system after a crash. Lock files remain on disk intentionally.

The ledger schema is version 1; new controlled manifests use version 3 and benchmark reports use version 2. Jobs, attempts, artifacts, and events carry schema versions. The implementation rejects an unknown ledger schema version. It does not yet provide schema migration, remote-worker coordination, code-environment restoration, or portable relocation of registered jobs. Resume uses the installed code and dependencies with the saved configuration; provenance is recorded, but an old Python environment or source checkout is not reconstructed automatically.

## Durable records and file locations

`jobs.sqlite3` is authoritative for job status, attempts/checkpoints, budget charges, artifact hashes, and append-only events. Mutable `manifest.json` and `benchmark.json` files are inspection/reporting projections of the recorded state. A rendering failure cannot turn incomplete verification into a pass or erase the persisted terminal execution status.

Each run's `events.jsonl` is a derived export of committed SQLite events. A normal export appends to an exact complete prefix. If a process leaves a torn, truncated, or mismatched JSONL file, the resume path explicitly rebuilds it from SQLite using an fsynced temporary file and atomic replacement. Do not edit JSONL to change a budget or execution outcome; the database controls recovery. Recovery of a derived log does not recover a missing or corrupt SQLite database.

Keep `jobs.sqlite3`, its linked controlled and benchmark run directories, suite input snapshots, and `.job-locks/` together. SQLite uses WAL mode; use a consistent SQLite backup or stop writers before making a filesystem backup, and do not discard active `jobs.sqlite3-wal` data. The `job.json` locator records an **absolute database path**, the database stores absolute run directories, and frozen configuration/source paths are also absolute. Copying only a run directory or moving the tree to different paths does not create a supported resumable copy. Preserve the registered paths until relocation support exists. The development dashboard reads run files and the ledger without modifying them.

## Software verification

Run the focused recovery, reporting, geometry, and dashboard checks with:

```sh
uv run --locked pytest \
  tests/test_ledger.py tests/test_execution.py tests/test_suite_execution.py \
  tests/test_benchmark.py tests/test_verification.py tests/test_dashboard.py -q
```

Run the complete software suite with:

```sh
uv run --locked pytest -q
```

These are CPU software checks using small generated fixture images and substitute backends. Recovery tests terminate separate processes before reservation, after reservation, after raw-image saving, and after result-image saving. They check preserved hashes, unchanged call counts, explicit uncertainty, source-snapshot reuse, child registration before interruption, parent budget enforcement, and reporting recovery without another inference. They also exercise cross-process locks, concurrent reservations, and torn-event export recovery.

The software checks do not establish GPU performance or artistic success. A separate real integration check is recorded below; historical measurements remain in [verification evidence](../experiments/verification.md) and [the benchmark evidence](../evaluation/verification_benchmark.md).

## Actual integration check, September 7, 2026

At this milestone the software suite passed **203 tests**, including the subsequently removed reviewer workflow. Dependency compatibility and wheel building passed. Human-review UI was deferred after this check; use the current test command above to validate the development dashboard.

One new real FLUX edit completed in `artifacts/recovery-integration-20260907/`, using the existing `mountain_tight.json` plan, seed 23, pinned base 4B weights, MPS BF16, model CPU offload, VAE tiling, 50 steps, guidance 4 and a 256 × 320 crop. End-to-end execution took 105.59 seconds, including observation. The sampled maximum MPS driver allocation during inference was 9,860,497,408 bytes (about 9.18 GiB); this is a sampled stage maximum, not a hardware peak.

Both new geometry dimensions passed: measured height ratio 0.3370, width ratio 0.3489, and bottom-center error 0.893 pixels. Zero pixels changed outside either the outer allowed mask or the effective edit region. Codex inspected the actual output: the person is smaller and anchored, but a narrow pale repair patch remains. This observation is model evidence; human review remains pending.

The exact CLI `uv run --locked kusama-edit --resume artifacts/recovery-integration-20260907` then completed with **zero additional model calls**. All registered artifact hashes and the raw PNG modification time stayed unchanged; the ledger and JSONL contain one inference reservation. `integration_audit.json` in that run records the checks and image-bound observation. This validates completed-run reuse with the real backend; abrupt crash recovery was tested separately with subprocess fixtures.

The earlier human-review UI has been removed from the application; existing data files are retained. The basic development dashboard requires no review database or setup.
