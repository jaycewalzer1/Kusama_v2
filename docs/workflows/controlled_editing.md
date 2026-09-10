# Controlled editing

`kusama-edit` adds three execution methods while preserving the original `kusama-first-edit` experiment:

| Method | Operation | Preservation |
| --- | --- | --- |
| `whole` | FLUX regenerates the complete image | Measured against the mask; rejected if any protected pixel changes |
| `masked` | FLUX Klein inpainting, followed by hard compositing | Original RGB pixels copied outside the allowed mask |
| `geometric` | Inpaint the allowed region to repair the background, then paste a scaled original cutout about a fixed anchor | Same hard preservation, with deterministic cutout geometry |

A model can supply a JSON plan selecting a method and fallback order. `auto` is a deterministic capability policy: geometric when scale/anchor/object mask are supplied, otherwise masked for a local mask, otherwise whole for an entirely white mask. This milestone does not load a Qwen director, infer masks, or train a routing model.

## Run the included mountain example

From the repository root:

```sh
uv sync --locked
uv run --locked kusama-edit --plan examples/controlled/mountain.json --validate-only
uv run --locked kusama-edit --plan examples/controlled/mountain.json --local-files-only --vae-tiling
```

The example references the existing seed-17 parent in `artifacts/20260907T190926.622194Z_de125bc5/parent.png`. Generated artifacts are not tracked, so on a fresh checkout provide your own RGB PNG, mask files, parent path and SHA-256. To obtain the hash:

```sh
shasum -a 256 path/to/parent.png
```

The example mask was manually traced by Codex and is approximate. The editable region includes the original person and immediate surroundings; the crop is explicitly 256 × 320 pixels. Output remains 1024 × 1024. This is a different inference configuration from the initial full-canvas experiment, recorded in each attempt. Base 4B weights, pinned revision, BF16 on MPS, 50 steps and guidance 4 remain the defaults. Remove `--local-files-only` if weights have not been downloaded.

## Plan contract

All file paths are relative to the plan file. Required fields:

- `parent`, `parent_sha256`: RGB PNG and its immutable file hash.
- `instruction`, `target`: requested intervention and named object.
- `allowed_mask`: same-size grayscale PNG, exactly 0 for protected pixels and 255 for editable pixels. Must be nonempty.
- `method`: `auto`, `whole`, `masked`, or `geometric`.
- `seed`: integer 0 through 2^63−1.
- `max_calls`: maximum actual model invocations, 1–6.
- `visual_checks`: nonempty list of independent review questions, including intended artistic effect.

Optional fields:

- `fallbacks`: ordered list of at most five explicit methods. Every attempt starts from the same original; a rejected candidate never becomes the next parent. A repeated method uses the same seed, so it is useful for transient failures rather than sampling diversity.
- `object_mask`, `scale`, `anchor`, `repair_prompt`: required together for geometric editing. Anchor is `[x, y]` in full-image coordinates; scale is uniform and greater than zero, at most 4. Both the original and transformed cutout, including interpolation edges, must fit within the allowed mask. Background repair uses `repair_prompt`, not the resize instruction.
- `crop`: `[left, top, right, bottom]`, containing the entire allowed mask. Width and height must be at least 64 and multiples of 16. No implicit resize. Whole-image execution uses the full parent regardless of crop.
- `author`: provenance describing the plan/mask author.

Unknown fields are rejected. Scope is frozen before execution; fallbacks cannot widen masks. A completely white mask explicitly permits the whole image to change and protects zero pixels.

## Checks and outcomes

Each run writes the plan, parent and mask snapshots; raw model outputs; composited candidates; difference images; side-by-side comparisons; and an atomic manifest containing settings, hashes, source code/dependency provenance, attempts, failures and actual call count. Raw outputs are retained even if dimensions are wrong. Model invocation failures consume budget; prompt/preparation failures do not. The finite fallback list also bounds failures before inference.

Pixel checks compare decoded RGB values, not compressed PNG bytes. Masked methods guarantee identical pixels outside the mask through compositing; inside the mask, unrelated details may still change. No check silently expands the allowed region. Hard mask edges can produce seams.

Geometry records the requested affine transform, supplied mask bounds and transformed alpha/core bounds. Antialiasing may extend fractional edge coverage by a pixel. This establishes cutout construction, not correct segmentation, natural contact with the ground, successful removal of the original person, or semantic geometry in a generated alternative. Whole/masked candidates with geometric requirements explicitly need independent measurement.

A candidate passing preservation stops at **`needs_review`**, never automatic acceptance. `review.json` binds questions to the candidate SHA-256 and separates model observations from human labels. Inspect the original, result, masks and difference image before recording observations. Review remains pending until that inspection; no artistic scoring model is included. Failed deterministic checks or execution errors can trigger only the predeclared fallbacks. A visual rejection requires a new recorded plan/run; it cannot silently trigger broader edits.

`--validate-only` loads no model. The normal CLI returns 0 for a candidate ready for review, 1 for rejection/exhausted budget, and 2 for invalid input. It accepts the original runner's backend flags (`--device`, `--precision`, `--offload`, `--cache-dir`, `--steps`, `--guidance`, etc.). The plan controls source, instruction, seed, budget and dimensions; corresponding first-experiment flags do not override those plan fields. `--max-retries` is not used here: retries must appear in `fallbacks`.

## Verification

```sh
uv run --locked pytest -q
```

Focused tests exercise unauthorized drift, successful compositing, geometry, wrong-size output retention, mask validation, immutable parent hashes, fallback behavior, and budget accounting. These software tests do not establish artistic quality; real-run observations are recorded separately.

## Actual integration results, September 7, 2026

Two new real inpainting calls used the included plans, pinned base 4B, cached weights, MPS BF16, 50 steps, guidance 4, VAE tiling and a 256 × 320 crop:

| Run | Wall time including loading | Protected pixels changed | Finding |
| --- | --- | --- | --- |
| `artifacts/20260907T210122_controlled_ea475800` — geometric | 107.60 s | 0 of 1,031,297 | Supplied cutout height reduced from 193 to 64 pixels about anchor (351, 871) |
| `artifacts/20260907T210342_controlled_956dca9c` — masked | 130.29 s | 0 of 1,031,297 | Visibly smaller person; apparent slight rightward displacement, exact geometry unmeasured |

Codex inspected both outputs. Both background repairs show a smoother patch than adjacent original textures. They suggest the intended increase in perceived mountain scale, but remain partial successes pending human judgment. Observations are bound to image hashes in each run's `review.json` and manifest; no human labels were fabricated.

The masked run's `methods_comparison.png` shows original, geometric and masked outputs at identical scale. The first real run preceded the addition of per-attempt comparison/memory capture; its code hashes identify that version. The second records sampled memory (about 15.82 GiB MPS driver allocation at maximum observed sample). Neither run is a controlled ranking of all methods or seeds. Whole-image execution was exercised in the original milestone; this milestone's tests additionally cover its preservation rejection/fallback path.

Verification: **72 tests passed**; both new real runs used exactly one model call; saved PNG readability and manifest PNG hashes were independently checked. CLI plan validation and locked installation passed. No Qwen direction, automatic segmentation, independent visual judge, or artistic acceptance is claimed.

The subsequent verification milestone adds optional `repair_policy`, `secondary_mask`, and `verification` plan fields. See [independent verification and benchmark](../evaluation/verification_benchmark.md) for the extended contract and actual comparative results.
