# One frozen Qwen-directed edit

`kusama-direct` connects `Qwen/Qwen3-VL-8B-Instruct` to the existing FLUX editor. Qwen sees a saved image and an artistic intention, proposes one specific change and its expected effect, and supplies a concrete instruction for one FLUX call. “Frozen” means using existing weights without training. Qwen's visual director is separate from the Qwen text encoder already inside FLUX.

The immediate milestone ends with a human assessment: did the visible change produce the intended artistic effect? A valid proposal, completed inference, or preservation check cannot answer that question on the user's behalf.

## Run from the saved mountain parent

```sh
uv sync --locked --dev
uv run --locked kusama-direct \
  --parent-image artifacts/20260907T190926.622194Z_de125bc5/parent.png \
  --intention 'Make this landscape feel overwhelmingly vast' \
  --device mps --offload model --vae-tiling
```

The parent is the existing 1024 × 1024 FLUX generation, with SHA-256 `f23302736be704e039b860f2ba717a9951816ba07b024e60dfbcbb233b6912e8`. Qwen receives that image and decides the intervention. Its expected effect is a hypothesis for review, not a finding about the output.

Qwen closes before FLUX loads to avoid holding both full models in the Mac's unified memory. The first Qwen run needs its separate model download. `--local-files-only` controls the FLUX cache; `--director-local-files-only` separately requires already-cached director weights. These flags do not substitute one model for another.

To save the proposal before rendering:

```sh
uv run --locked kusama-direct \
  --parent-image artifacts/20260907T190926.622194Z_de125bc5/parent.png \
  --intention 'Make this landscape feel overwhelmingly vast' \
  --device mps --propose-only
```

Then execute the saved packet, replacing the example directory with the path returned above:

```sh
uv run --locked kusama-direct \
  --execute artifacts/REPLACE_WITH_QWEN_RUN_DIRECTORY \
  --device mps --offload model --vae-tiling --local-files-only
```

`--execute` uses the saved, validated instruction and does not ask Qwen for a replacement proposal. Backend settings are recorded with the FLUX execution.

For recovery after execution has started, use `--execute <packet-directory>` alone. The existing job restores its saved FLUX settings and reuses surviving outputs; a charged call with no surviving output remains uncertain and is not rerun automatically.

## Saved evidence and review

Each new packet is stored beneath `artifacts/<timestamp>_qwen_<id>/`:

```text
parent.png
direction.json
plan.json
allowed_mask.png
manifest.json
edit.png
comparison.png
edit/
```

`direction.json` retains Qwen's raw response, structured proposal, and model/configuration metadata. The plan links the parent and exact instruction to a single whole-image FLUX edit. The nested `edit/` job uses the existing executor to save raw and final images, a comparison, review questions, and execution provenance. The top-level `edit.png` and `comparison.png` are verified copies for the existing dashboard. A proposal-only packet has no completed FLUX image.

The director pins the [official Qwen checkpoint](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) to revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` and uses the installed Transformers processor and model class. `torchvision==0.29.0` is included because that processor initializes video support even for this image-only task. Greedy decoding, image preprocessing, frozen parameter counts, and dependency versions are saved with the response.

Open the saved comparison and judge the change against Qwen's expected effect. Record whether the intended change happened, whether it improved the image's finished quality, and whether it opened a promising direction for further work. Keep observations of what changed separate from preference. A/B/tie labels must come from the human reviewer; assistant or Qwen assessments remain model observations.

Whole-image editing supplies the image and instruction to FLUX without enforcing local pixel preservation. The original FLUX experiment already showed that this can alter unrelated details or miss a spatial constraint. Retain those outcomes as evidence of the director/editor combination's weaknesses. Existing geometric and masked methods remain available for later explicitly planned interventions.

## Readiness and progression

The original FLUX generation → edit → save → compare loop completed real inference on this Mac, with the evidence in [verification](../experiments/verification.md) and [visual review](../experiments/visual_review.md). Human preference labels are still absent from that run. Tests of the new adapter establish software behavior; they do not establish actual Qwen inference or artistic success. A completed director run needs its own saved response and FLUX result, followed by the user's judgment.

The first real Qwen-directed edit has now completed in `artifacts/20260908T000653.098912Z_qwen_719e9b03/`, with one director call and one FLUX call. The user reviewed the intended effect as uncertain, saying the images feel somewhat similar in vastness. Its images, exact instruction, hashes, timings, and human feedback are recorded in [the director verification](../experiments/director_verification.md). Finished quality and development potential remain unassessed.

| Stage | Qwen's role | Gate |
| --- | --- | --- |
| Frozen director | Propose directions and concrete FLUX instructions from the brief and current image. | Start now with one complete edit and human review. |
| Frozen observer and judge | Describe visible changes and recommend candidates, assessing finished quality separately from development potential. | Director-generated edits run reliably; compare its judgments with human reviews. |
| Train the judge | Learn preferences from human A/B/tie reviews. | Reviews are consistent and sufficiently varied to measure improvement on unseen briefs. |
| Train the director | Learn which interventions helped during development. | Reviewed development histories support evaluating useful actions; preference training can follow if justified. |

After the first reviewed Qwen-directed edit, expand to 12 directions × 2 realizations with Qwen already connected and weights still frozen. The first 216 diagnostic judgments should identify weaknesses. That count is not a training threshold: updates must demonstrate improvement on held-out briefs, and no training is part of the current milestone.
