# First real frozen Qwen-directed FLUX edit

The image-to-direction-to-edit-to-save loop completed on 7 September 2026 local time (8 September UTC). The user subsequently reviewed the intended effect and was uncertain, describing the images as feeling somewhat similar in vastness. Both models used their existing pretrained weights with zero trainable parameters; no training was performed.

## Retained run

Local packet: `artifacts/20260908T000653.098912Z_qwen_719e9b03/`.

- Intention: “Make this landscape feel overwhelmingly vast”.
- Parent: the original 1024 × 1024 mountain collage, SHA-256 `f23302736be704e039b860f2ba717a9951816ba07b024e60dfbcbb233b6912e8`.
- Director: `Qwen/Qwen3-VL-8B-Instruct`, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, MPS BF16, greedy decoding, 1,268 input tokens and 183 output tokens. The real processor supplied image tensors; there was no text-only substitute.
- Editor: `black-forest-labs/FLUX.2-klein-base-4B`, revision `a3b4f4849157f664bdbc776fd7453c2783562f4d`, MPS BF16, model CPU offload, VAE tiling, 1024 × 1024, 50 steps, guidance 4.0, seed 23.
- Calls: exactly one Qwen proposal and one FLUX image edit, without retry.
- Edited image: SHA-256 `e0b529d2429f4b27d3f651af48faa258a9bd6e7378bef9bdaaf9ab89bc4705f7`.

Qwen's exact edit instruction, retained in `direction.json`, `plan.json`, and the executed FLUX attempt, was:

> Make the mountain ridgelines more jagged and extend them further into the distance, while keeping the hiker's position fixed.

Qwen predicted greater vastness and isolation, with the hiker appearing smaller against an expanded range. It separately listed seven preservation constraints. Most of those constraints were absent from the executable instruction. The original response was retained unchanged for diagnosis.

## Verification

The Qwen inference measurement was 19.21 seconds; direction preparation, inference, and packet creation took 32.76 seconds. The FLUX job took 1,170.36 seconds (19.51 minutes). Total recorded runtime excluding checkpoint download was 1,203.12 seconds (20.05 minutes). Sampled Qwen MPS driver allocation reached 17.32 GiB; this is an observed sample maximum, not an exact allocator peak. Qwen was released before FLUX loaded.

All packet and image hashes were checked. The exact parent hash and instruction match across the director response, controlled plan, and FLUX manifest. The edited image and comparison decode successfully, and the dashboard lists them. Re-executing the saved packet returned in under a second without loading either model or increasing either call count.

The software suite has 254 passing tests. Tests cover multimodal input, frozen weights, strict proposal validation, raw failure retention, sequential model release, exact handoff, interrupted-call accounting, and recovery without duplicate inference. Two existing dependency deprecation warnings concern the dashboard test client.

## Assistant observations, not human preference labels

The edited mountains have sharper, more repeated angular ridges. The hiker remains in approximately the same location and appears similar in size. The forest pattern, foreground rocks, cloud shapes, and landscape textures also changed. Those visible changes do not establish that the image feels more vast or has better finished quality.

The executor reports `needs_review`; semantic geometry verification was not requested, and a whole-image allowed mask does not enforce the textual preservation constraints. The review now contains one human response on intention alignment. Finished quality and development potential remain unassessed.

## Human review of the intended effect

The user's verbatim response was:

> honestly not sure if its overwhelmingly more vast or not they feel somewhat similar

This is recorded as `uncertain`, with no decisive A/B preference or forced tie. The human observation is that the images feel somewhat similar in vastness; it does not establish equivalence, failed execution, or a preference about finished quality. The response is bound to both image hashes, retained in an immutable human-review artifact, and reflected in the ledger and review records.

The first Qwen-directed loop now includes human feedback. The diagnostic finding is that Qwen's predicted increase in vastness was not clearly established for this viewer, despite visible changes to the mountain ridges. This is one uncertain comparison, not evidence for training an update. The 12 directions × 2 realizations expansion should test varied artistic interventions with both models still frozen.

See [the workflow and training gates](../workflows/frozen_director.md). Keep the entire local packet, including its nested `edit/` job and SQLite ledger, when backing up the evidence.
