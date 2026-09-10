# Phase 1 evaluation handoff — 8 September 2026

Snapshot: **2026-09-08T15:10:50.685902+00:00**. This milestone performed no new Qwen/FLUX generation, no weight updates and no human annotations. It ran deterministic image analysis and the already cached frozen CLIP encoder on CPU.

Implemented `kusama eval` with a versioned 24-brief development suite, six-brief smoke schedule, current/baseline compact-workflow capture, immutable attempt/image records, descriptor and CLIP caches, reference proximity, local blinded packets, validated checklist/pair/image-first import, brief-level comparison and a future twelve-task release registration schema. The [evaluation guide](../operations/2026-09-08_dev_panel.md) contains exact commands, metric definitions, scope and limits; [future work](../operations/2026-09-08_dev_panel.md) remains explicitly planned.

## Executed evidence

| Flow | Images | Attempt records | Review pairs | Genuine annotations |
| --- | ---: | ---: | ---: | ---: |
| Fixture demonstration | 24 | 25, including one deliberate failure | 12 | 0 |
| Cached E0 development diagnostic | 12 | 12 native revision attempts | 6 available, 2 missing | 0 |

The cached diagnostic compares the native **ordinary_critique** and **inquiry_relationship** conditions on three completed development briefs, retaining the fourth brief's two missing pairs. It is not the new compact-workflow benchmark. All 12 real images decoded, matched source hashes and received pinned 512-dimensional CLIP descriptors and top-five reference neighbors. Legacy hard requirements were not frozen, so none were retroactively invented. All inference intervals are withheld for this tiny historical diagnostic; the reported differences are descriptive. Higher embedding spread is not a quality or preference result.

Native condition renderer durations are retained; shared parent/planning, model-load and other-condition work appears in a separate native cost context. Actual total monetary costs and matched-resource claims remain unavailable. Historical E0 profile identity remains `flux_base4b_e0_384x512_v1`; prospective capture remains `flux_base4b_study_512_v1`.

- [Real cached comparison report](../../artifacts/phase1/delivery/cached/reports/954c83ba367f34fb78a29280/report.md) and [machine-readable delivery record](../../artifacts/phase1/delivery/delivery.json).
- [Fixture comparison report](../../artifacts/phase1/delivery/fixture/reports/ff7a6e1ed654ac0a334733b1/report.md).
- [Real image-first review packet](../../artifacts/phase1/delivery/cached/review/reviewer/stage1/README.md).
- [Editable stage 1 responses](../../artifacts/phase1/delivery/cached/responses/stage1.csv) and [stage 2 responses](../../artifacts/phase1/delivery/cached/responses/stage2.csv). Both remain blank templates.
- [Prospective smoke manifest](../../artifacts/phase1/delivery/smoke-plan/manifest.json) and [full development plan](../../artifacts/phase1/delivery/full-development-plan/manifest.json): no generation dispatched; the frozen resource session has expired.

## Verification

The full shared repository suite passed **725 tests** with two dependency deprecation warnings. The final focused Phase 1 suite passed **17 tests** after the last changes, including packet identity bound to actual stimuli. Eight checked-in schema snapshots match the Pydantic contracts. Own documentation links, preserved image hashes, contact-sheet layout, zero-annotation template import and descriptive-only intervals were checked. Native generation dispatch was exercised with controlled substitutes and existing ledgers were read; no new live generation is claimed.

A concurrent documentation reorganization and content diagnostic were preserved. This milestone's new files are under `src/kusama/phase1/`, `configs/evaluation/`, `docs/evaluation/`, this handoff and `tests/test_phase1.py`; the shared CLI and documentation index/readme have small integration additions.

## Next evidence gate

Complete image-first responses before reading the intended effects in stage 2, then import genuine human responses:

```sh
.venv/bin/python -m kusama.cli eval review-import \
  --packet artifacts/phase1/delivery/cached/review \
  --annotations artifacts/phase1/delivery/cached/responses/stage1.csv
.venv/bin/python -m kusama.cli eval review-import \
  --packet artifacts/phase1/delivery/cached/review \
  --annotations artifacts/phase1/delivery/cached/responses/stage2.csv
.venv/bin/python -m kusama.cli eval compare \
  --baseline artifacts/phase1/delivery/cached/cached-e0 \
  --candidate artifacts/phase1/delivery/cached/cached-e0 \
  --packet artifacts/phase1/delivery/cached/review \
  --output artifacts/phase1/delivery/cached/reports
```

Blank imports collect zero judgments. Human preference, the benefit of branching/revision, and improved artistic outcomes remain unmeasured. The next scientific gate is genuine blinded review followed by an explicitly selected-parent revision and follow-up review, alongside a separately authorized prospective comparison. Local artifacts are ignored by Git and require a separately preserved artifact tree; links will not resolve in a fresh checkout.
