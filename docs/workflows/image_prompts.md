# Qwen-selected image prompts

Implemented 9 September 2026 for new independent-research/panel plans and bounded-development plans. Qwen may choose text-only rendering or attach inspected source images. The legacy creative and automatic-pairing paths keep their existing behavior. Both models remain frozen.

## Model choice

An independent-research hypothesis has an optional `render_images` list. Each entry names a selected, opened `source_id`, a zero-based `image_index` within that source's saved views, and Qwen's `purpose`. The model cannot supply filesystem paths. Empty or omitted means text only. Whole/detail views are distinct choices, and draft/commit still requires reinspection.

For example, these fields could form part of a hypothesis after opening a source with the actual ID `doll`:

```json
{
  "render_images": [
    {"source_id": "doll", "image_index": 0, "purpose": "Transform the forehead contour into a new portrait element"}
  ],
  "render_instruction": "Paint a frontal portrait. Give its forehead the broad convex contour visible in image 1, translated into overlapping painted paper. Preserve an asymmetrical mouth and the surrounding flat blue field."
}
```

This example is operator-authored syntax and visual scaffolding, not a historical claim about a doll or evidence of a successful transformation. Source interpretation, intended effect and visible result remain separate.

Initial generation accepts at most two attachments. FLUX receives them as image 1 and image 2 in list order, plus the exact saved text instruction. Source references do not become lineage parents. Other research sources remain context for Qwen but are not attached to FLUX.

Bounded-development proposals expose the same `render_images` choice. Each supplied development image has index 0; text sources have no image index. FLUX image 1 is the artwork being edited, and attachments become image 2 and image 3. Whole edits permit two references; masked edits permit one. For a cropped masked edit, image 1 is the recorded parent crop. Qwen must state the transfer and preservation instructions itself. The masked pipeline receives the parent through `image` and the separate reference through `image_reference`; the ordinary pipeline receives an ordered `image` list.

The existing continuation API also accepts `revision.reference_images`: resolved entries containing `source_id`, `image_index`, `purpose`, `path` and `sha256`. Each ID must resolve in the supplied reference list. Preparation copies the bytes into the continuation and controlled-execution snapshots. Model choices use this API only after resolving them against the actual supplied source images.

## Preserved evidence and limits

Selections, purposes, source provenance, original bytes, copied image bytes, SHA-256 hashes, input order and model instructions remain saved. Initial render actions and output records have `reference_images`/`conditioning` evidence; controlled-edit attempts record the exact parent/crop and attached references. The panel's image transcript displays the FLUX inputs and exposes their hashes. Copied sources, rather than mutable original files, drive execution. Altered or missing frozen references fail validation; completed renders resume without a new call. Existing source restrictions on derivatives are respected.

Each reference is decoded as RGB. The pinned Diffusers 0.40.0 processor bounds a reference above one megapixel and crops dimensions to multiples of 16, as recorded in `conditioning.preprocessing`. Original files remain untouched. References can have different dimensions from the output. Output resolution, seeds, steps, model revision, token limits, charged-call limits and retry policy are unchanged; more reference pixels can increase inference time and memory. The implementation uses the installed [FLUX pipelines](https://huggingface.co/docs/diffusers/v0.40.0/en/api/pipelines/flux2).

New versions are `independent-research-v12-image-prompts`, `research-render-v2-image-prompts`, and `bounded-development-v7-image-prompts` / `bounded-development-v7-research-image-prompts`. Existing frozen plans are not upgraded or retried. Start a new run with the updated application process to expose the option. The choice is available automatically; Qwen is not required to use it. No extra inference calls or resource sessions are authorized by planning.

## Verification and next evidence

Fixture tests cover model-selected ordering through research, rendering and whole/masked continuation; explicit detail selection; text-only defaults; invalid or invented selections; source restrictions; changed/missing bytes; parent/reference separation; and replay without duplicate inference. Backend tests inspect actual pipeline arguments with substitutes. These tests do not establish real multi-image inference quality or Qwen's tendency to choose useful attachments.

The 9 September verification passed 966 default software tests (28 opt-in cases skipped), followed by 23 passing cached-processor acceptance cases, including image-prompt selection through the panel and renderer substitute. Four archive-dependent processor cases and the unmocked live-process case were not run for this change. Generated TypeScript contracts, JavaScript syntax, local documentation links and diff whitespace checks passed. This is not the combined live-process release gate used for the earlier v11 run.

```sh
.venv/bin/python -m pytest -q tests/test_image_prompt.py tests/test_backend.py \
  tests/test_research_render.py tests/test_research_argument_recovery.py tests/test_autonomous.py
.venv/bin/python scripts/generate_contract_types.py --check
```

The next evidence gate is a separately admitted real generation and parent revision using an explicitly selected reference, followed by inspection of feature transfer, unintended changes and human assessment of the intended effect. No new model inference, human labels or training updates were produced while implementing this capability.
