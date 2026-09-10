# Independent verification and repair benchmark

This milestone adds output measurements, smaller repair regions, and a seven-call development benchmark. It does not add an autonomous director or automatic artistic acceptance.

## Run

```sh
uv sync --locked
uv run --locked python scripts/build_benchmark.py
uv run --locked kusama-benchmark --suite examples/benchmark/suite.json --validate-only
uv run --locked kusama-benchmark --suite examples/benchmark/suite.json --vae-tiling
```

Use `--local-files-only` once **both** FLUX and the detector weights are cached. The detector is `IDEA-Research/grounding-dino-tiny`, pinned to `a2bb814dd30d776dcf7e30523b00659f4f141c71`. It runs frozen in FP32 on CPU, after the editing backend releases its weights. Transformers and NumPy are locked in `uv.lock`; no additional computer-vision runtime is needed.

The mountain case references the existing seed-17 parent in `artifacts/20260907T190926.622194Z_de125bc5/parent.png`. On a fresh checkout supply that original or update the parent path/hash and its annotation. The two synthetic calibration images and masks are rebuildable from the script. They deliberately have known colors and geometry, and do not substitute for testing a broad natural-image distribution.

The CLI accepts the same backend flags as `kusama-edit`. `--validate-only` checks every plan and source before loading weights. The suite has a global budget, separate from each plan's bound, and reserves calls at the actual inference boundary. Each row starts from its own original source. Failures and budget skips remain visible. A completed benchmark means the suite ran; individual edits may fail.

## Smaller repair regions

A plan may specify:

```json
"repair_policy": {"padding": 4, "feather": 2},
"secondary_mask": "optional_shadow_mask.png"
```

`secondary_mask` is optional and must remain inside the original allowed mask. It explicitly permits repair of effects such as a shadow. The outer `allowed_mask` remains the maximum scope. Padding/feather are integer pixels, with `0 <= feather <= padding <= 16`.

For geometric edits, the background repair region is the padded original object silhouette plus any secondary-effect mask, minus pixels that the new cutout will cover completely. Partially transparent new edges still get repaired underneath. The model may inspect the larger crop, but only repair pixels are composited back. Blending remains inside the repair region, with full strength beneath exposed original-object pixels to reduce ghosts. The cutout is pasted afterward.

For masked edits, the active region also includes the original and destination object supports, so the model can draw the replacement. Masked generation uses hard compositing over this smaller active region. The feather setting applies to geometric background repair, not masked target generation. Without `repair_policy`, legacy broad-mask behavior remains available for comparison.

Every run saves separate object, secondary-effect, repair, blend, effective and local masks. The effective preservation check measures whether nearby pixels excluded by the smaller scope stayed identical, in addition to checking the outer allowed mask. Boundary gradient change and repair-area pixel error are diagnostic measurements, not universal seam-quality scores.

## Independent output measurements

A resize plan can request:

```json
"verification": {
  "observer": "grounding_dino",
  "label": "person",
  "operation": "resize",
  "height_tolerance": 0.2,
  "width_tolerance": 0.2,
  "position_tolerance": 6
}
```

The observer detects the target in both the original and completed candidate crop. It does not receive the transformed cutout or construction bounds. Measured height and width must both match the requested uniform scale; the measured box's bottom center is compared with its expected position under the requested anchored transform. This is a ground-contact proxy, not pose estimation. Height and width tolerances are relative to their respective requested scaled dimensions; position tolerance is in original-image pixels. `width_tolerance` defaults to the configured `height_tolerance` (0.2 if neither is supplied), and can be specified independently between 0 and 1. Width ratio and relative width error are recorded alongside the height measurements. A stretched or narrowed target cannot pass merely by preserving its height and bottom center.

Multiple strong target detections flag possible duplicates/remnants. Missing or weak detections yield `uncertain`, rather than proof of absence. The frozen thresholds are 0.30 for strong detections, 0.15 for weak detections, and 0.5 IoU for duplicate-box suppression. These are operational thresholds, not calibrated probabilities. Tiny fragments and atypical objects can be missed. Detector nondetection alone never passes removal.

Malformed, degenerate, or nonfinite observed boxes and confidence values yield `uncertain` before geometry arithmetic or overlay drawing. Invalid observations are retained as string evidence so the verification report remains valid JSON. A valid observation is never substituted to turn an invalid or ambiguous observation into a pass.

The synthetic cases use a separate color observer based on HSV thresholds and connected components. Recoloring checks target-color coverage, remaining source color and measured box overlap. Removal requires both absence of the source-color components and agreement with the explicitly annotated background color. These checks apply to the calibration scenes' stated properties; they do not prove semantic removal in arbitrary photographs.

Independent verification is saved in `attempt_N_verification.json`, bound to the result's SHA-256, with detections, measurements, settings and limitations. Detection overlays make it inspectable. A failed independent check rejects the attempt and may trigger a predeclared fallback. An uncertain check remains visible and requires review; it is never silently treated as passing. Verification errors, including unavailable detector weights, likewise remain uncertain with the recorded error.

## Benchmark and review

The suite covers three source images and three edit types:

- Mountain/person resize: broad geometric repair, tight geometric repair, tight masked generation.
- Red-circle removal: broad and tight masked generation.
- Green-rectangle-to-blue recoloring: broad and tight masked generation.

Each comparison uses seed 23, 50 FLUX steps and guidance 4 by default. Mountain inference uses a 256 × 320 crop; synthetic images use their full 256 × 256 canvas. There are no hidden retries. The seven-call cap covers image generation, while detector evaluations are recorded separately in verification runtime.

The benchmark directory contains the suite snapshot, durable `benchmark.json`, `results.csv`, `report.md`, and a contact sheet with original/result/detection crops. Per-case run directories contain full-resolution images and provenance. Reports count pass/fail/uncertain separately and retain cases with unavailable measurements. Visual observations and human labels remain separate; an independent-check pass does not accept the artwork.

The terminal suite status, elapsed time, consumed calls, and case errors are saved before summary or contact-sheet generation. `reporting_status` and `reporting_errors` record derived-report failures independently of execution, including the stage and affected case when available. Failure in one report does not prevent the other from being attempted or replace an original interruption. The CLI exits unsuccessfully when reporting errors remain. A corrupt case preview is recorded and omitted while readable cases still produce a contact sheet.

Summary rows describe the latest attempt and include its index, status, and error. Partial attempts retain whatever preservation metrics were completed; absent verification is `unavailable`. A prior attempt's passing verification is never carried forward to a later failed attempt. Contact sheets label the attempt whose saved result is actually shown. Cleanup and preview failures therefore preserve the saved result evidence even when verification did not run.

## Sources

The detector integration follows the [official Transformers Grounding DINO documentation](https://huggingface.co/docs/transformers/model_doc/grounding-dino). It supplies open-vocabulary object boxes; the geometry checks, uncertainty rules, mask compositing and benchmark accounting are implemented locally.

An additional diagnostic deliberately pastes a second original-sized person beside the edited person. It is stored separately from benchmark candidates and does not consume an image-generation call. The real frozen detector found both people and rejected the image. This exercises actual detection integration, while unit tests cover weak/missing detections, geometry error, color failures and budget boundaries.

Color-observer tolerances are deliberately explicit: target recolor coverage at least 80%, remaining source color at most 5%, and box IoU at least 0.8; known-background removal MAE at most 15 RGB levels with no remaining source-color component of eight pixels or more. These are approximate calibration checks, not exact shape preservation. Exact equality is enforced separately outside the editable region.

## Completed pilot, September 7, 2026

Results are in `artifacts/20260907T212453_benchmark_d44a1b5e/` (`report.md`, `results.csv`, `comparison.png`, `benchmark.json` and `artifact_audit.json`). The suite completed seven real FLUX calls in 780.11 seconds. Five independent checks passed; two failed. Every saved candidate changed zero pixels outside both its outer permission mask and its more restrictive effective region.

| Case | Measured result | Independent outcome | Codex visual observation |
| --- | --- | --- | --- |
| Mountain, broad geometric | Height ratio 0.335; position error 0.49 px | Pass | Smoother background patch |
| Mountain, tight geometric | Height ratio 0.337; position error 0.89 px | Pass | Narrow pale repair patch |
| Mountain, tight masked | Height ratio 0.878; position error 8.31 px | Fail | Person remains too large |
| Circle, broad removal | Background MAE 0.67; no red components | Pass | No obvious repair defect at inspected scale |
| Circle, tight removal | Background MAE 93.84; residual color | Fail | Circle became orange instead of disappearing |
| Poster, broad recolor | Blue coverage 100%; box IoU 0.977 | Pass | Large light background patch |
| Poster, tight recolor | Blue coverage 98.9%; box IoU 0.990 | Pass | Thin pale outline remains |

The tight mountain method reduced the region that could change from 17,279 to 9,988 pixels (about 42%). It preserved more nearby background but did not establish better visual repair quality. The removal results likewise show that smaller masks are not always more successful. No method ranking or general success rate is inferred from this one-seed pilot.

**92 software tests passed.** All seven runs' PNG hashes/readability, plan hashes, result-to-verification bindings, exact preservation and total call count were audited. Codex inspected each actual result; observations are image-bound and remain separate from human labels. The final code additionally enforces the effective region for whole-image attempts; all saved benchmark cases already satisfy that gate. Original inference code hashes are retained, and report postprocessing is recorded separately.

The pilot outcomes above used the original height-and-position resize check. They have not been reclassified under the subsequent width requirement. Follow-up regression tests cover width expansion and collapse, invalid detector observations, incomplete attempt manifests, cleanup failure, independent report failures, and preservation of an original interruption; these checks use temporary fixtures and do not alter the saved pilot artifacts.
