# Creative loop verification

The implemented workflow accepts a saved brief and a manually researched reference packet, asks frozen Qwen for directions, renders two FLUX realizations per direction, optionally asks Qwen to observe the results, and presents independent human comparisons and advancement choices. It does **not** include an automated reference-finding service.

## Maya experiment and V3 scaffolding

The first Maya-memory experiment is ready for human review: **12 directions, 24 paintings and all 12 planned observer attempts are saved** in [`20260908T022716_creative_ee890a8f`](../artifacts/20260908T022716_creative_ee890a8f/manifest.json). Technical integrity verification passed at **2026-09-08T04:17:27Z**. The observer remains partial: **7 valid responses and 5 unresolved pairs**. The active packet is [`packet-v3.json`](../references/maya_memory/packet-v3.json). The [dated project assessment](project_direction.md) incorporates this final execution evidence and distinguishes it from the remaining research questions.

The brief concerns the user's feeling that Maya history is lost: grief, interrupted memory and ancestral absence. Wangechi Mutu supplies process and formal inspiration—collage, bodily transformation, material texture and charged composition. Maya-specific sources supply cultural context; the packet does not claim a historical relationship between Mutu's culture and Maya cultures.

V3 retains the same 12 reference records as V2, including four images and eight text-only sources. Its sources include Mutu's *Family Tree* and studio process, living Maya voices, the Dresden Codex, the *Popol Vuh*, weaving, testimony and documented destruction and access. Source URLs, contextual paraphrases, proposed relevance and permitted uses are preserved in [the V3 packet snapshot](../artifacts/20260908T022716_creative_ee890a8f/packet.json). These materials condition inference; they are not a training corpus.

## What each stage does

1. **References → Qwen:** The runner snapshots the brief, source text and image bytes with SHA-256 hashes. Qwen receives the four reference images and their paired context. The experiment requests 12 directions, one per call, with a specific pictorial intervention, expected effect, source relationships and exact FLUX prompt. V2 supplied prior direction summaries. The V3 adapter supplies the assigned composition challenge and avoids repeating prior core interventions as visual anchors. Proposed analogies remain distinct from documented cultural facts.
2. **Qwen → FLUX:** Each saved direction gets two initial generations, differing in recorded seed. FLUX receives Qwen's exact text prompt; the reference images are inputs to Qwen and are not passed directly to FLUX in this experiment. Initial candidates have no image parent. The runner saves PNGs, prompts, seeds, source links, model provenance, call reservations and cost evidence.
3. **FLUX → observer:** With observation enabled, frozen Qwen examines each direction's two generated images. It records visible observations, separate finished-quality and development-potential preferences, uncertainty and a possible revision. These outputs are model judgments, never human labels or advancement decisions.
4. **Observer/results → human:** The dashboard displays all candidates and pair comparisons. Each pair has independent A/B/tie/unsure answers for finished quality and development potential, optional confidence and notes. Human advancement is a separate candidate/reviewer/reason record. Neither a vote nor an observer recommendation starts another generation. Executing a revision from a selected parent remains a separate explicit action.

The 12 × 2 design plans 24 images and **108 unique comparisons**, each with two questions: **216 planned dimension questions**. These are not completed votes. Each candidate appears nine times; all 12 sibling pairs and all 66 direction relationships are represented. See [comparison design](comparison_design.md) and [human review](human_review.md).

The observer attempted twelve within-direction comparisons. Seven passed the response contract, yielding fourteen usable model preference fields across the two dimensions; five pairs have rejected responses only. These are not predictions for the full 216-question human schedule. Agreement can be checked on the seven overlapping pairs once human reviews exist. The remaining human comparisons diagnose differences across directions.

These first reviews are diagnostic data, not a sufficient training milestone by themselves. Judge training should wait for consistent reviews across varied briefs and a held-out set of new briefs on which improvement can be measured. Director training additionally needs reviewed development histories linking interventions to their outcomes. This experiment performs neither update.

## Operator scaffold versus model invention

The V2 outputs repeated a fragmented body, woven veil, bark/codex and whistle structure. The operator therefore authored 12 distinct **composition challenges** for V3: portrait, interrupted weaving, landscape, hands, empty interior, disrupted map, speech/listening close-up, roots, diptych, nonfigurative erasure, river and broken circular gathering.

These structures were supplied by Codex, not independently discovered by Qwen and not selected by a human quality vote. Qwen must formulate the specific pictorial treatment, source transformations and executable painting prompt within each challenge. The intervention changes inference instructions; it does not train weights. Provenance and limitations are recorded in [`diversity_scaffold.json`](../references/maya_memory/diversity_scaffold.json).

Read-only comparison confirmed unchanged reference records. The new challenges describe proposed visual metaphors, not asserted Maya ceremonies, authentic maps, historical events or sacred symbols. The packet retains distinctions among Maya sources, abstract-only invented marks, and the separation of Mutu's process inspiration from Maya cultural context. Adherence and artistic diversity still require inspecting Qwen's actual outputs and generated images.

## Frozen models and study settings

| Role | Existing pretrained weights | Pinned revision |
|---|---|---|
| Generator | `black-forest-labs/FLUX.2-klein-base-4B` | `a3b4f4849157f664bdbc776fd7453c2783562f4d` |
| Director / observer | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |

Both loaders set evaluation mode and disable parameter gradients; no training occurs. The run records `models_frozen: true` and `training_enabled: false`. Qwen is released before FLUX loads, and FLUX is released before observer loading.

V3 executed **24 RGB PNG studies at 512 × 640**, 50 FLUX steps, guidance 4.0 and BF16 on MPS with model CPU offload. Seeds run from 11001 through 11024; both realizations within each direction use the exact same Qwen prompt. These settings support comparing ideas; they do not establish finished-work quality or caliber comparable to the artist's work. Artistic success and cultural accuracy still require human judgment. Unknown electricity/hardware rates remain unknown monetary totals; see [cost accounting](cost_accounting.md).

## Retained V2 diagnostic evidence

The manifest last updated **2026-09-08T02:22:43.811117+00:00** records `failed`: the seventh Qwen response was rejected for a duplicate direction title. It retains **six validated direction records, seven model invocations, zero FLUX images and zero observer responses**. The six accepted directions are semantically close despite passing the earlier text-identity checks; the duplicate-title rejection does not establish that the accepted batch was creatively diverse.

No candidate review collection was completed by this run. Its `human_review: pending` field is not evidence of human votes. Preserve this run and its raw responses as a failed diagnostic; do not rewrite it as a successful 12-direction experiment.

## Final V3 execution evidence

The [technical verification report](../artifacts/20260908T022716_creative_ee890a8f/verification.json) checks source snapshots, exact model IDs and revisions, frozen runtime flags, image bytes and dimensions, unique seeds, exact Qwen-to-FLUX prompts, observer input bindings, every accepted/rejected response, the immutable review design and complete recorded cost coverage. It verifies technical integrity, not artistic success or the truth of model judgments.

| Evidence | Verified result |
|---|---|
| Frozen director | 12 validated directions within the disclosed operator challenges |
| Frozen FLUX | 24 RGB PNGs, 512 × 640, 50 steps, guidance 4.0, seeds 11001–11024 |
| Observer coverage | All 12 original pairs attempted; 7 accepted responses, 5 unresolved pairs |
| Invocation accounting | 48 original calls plus 2 separately recorded retries = **50 actual model calls** |
| Retained outcomes | 43 usable work outputs and 7 failed attempts; failed costs and raw responses preserved |
| Human review collection | 24 candidates, 108 comparisons, 216 planned dimension questions |
| Human history at verification | **0 votes and 0 advancement records**; later records may increase these counts |
| Software checks | **402 tests passed**, with two dependency deprecation warnings |

The [contact sheet](../artifacts/20260908T022716_creative_ee890a8f/contact-sheet-24.png) presents every output in direction/seed order with equal thumbnails. Source paintings are unchanged. Open the [local review studio](http://127.0.0.1:8765/) to compare full-size images and record independent finished-quality and development-potential answers. The live interface was inspected with all 24 images, separate observer status, combined recovery accounting and an empty reviewer field. No assistant-authored preference was submitted.

### What worked and what did not

The generation and reference handoff worked: Qwen received four saved reference images with sourced text, wrote specific treatments for twelve operator-supplied structures, and FLUX generated both realizations of each exact prompt. The [direction audit](../artifacts/20260908T022716_creative_ee890a8f/assistant-direction-audit.json) records remaining repetition and ambiguous invented-mark instructions. This is evidence of a functioning experiment, not proof of artist-level work.

The observer is not yet reliable. The unresolved pairs are **d02, d04, d05, d06 and d10**. Their outputs failed JSON or content-shape validation, including omitted candidate descriptions. [Accepted observations](../artifacts/20260908T022716_creative_ee890a8f/observations.json) and [rejected responses](../artifacts/20260908T022716_creative_ee890a8f/observer_rejections.json) are separate. Seven rejected response records cover five unresolved pairs plus the retained original d01 failure and the additional d02 retry failure. Rejections are not counted as usable preferences.

The [assistant visual audit](../artifacts/20260908T022716_creative_ee890a8f/assistant-visual-audit.json) also identifies a grounding error in accepted d01: Qwen credits the first portrait with blue drips that appear in the second. There are useful observations too: d07 identifies the absent ear and distinguishes finished quality (second image) from development potential (first idea); d12 correctly notices that the first spiral has a filled center. These are model observations and predictions, not human votes. The observer sees the intended direction along with the images, so its descriptions may be influenced by the intended result.

### Disclosed recovery and costs

Two parsing recoveries retained original response bytes and wrote linked sidecars. The tenth director proposal had a complete object followed by surplus closing delimiters. The d01 observer retry contained both descriptions but lacked one array-closing bracket; the explicit recovery recorded the single inserted bracket at Unicode offset 602, then applied strict parsing and full content-shape validation. Neither operation invented prose or made another model call.

The original d01 observation omitted the second candidate description, so it received a separately reserved retry. The d02 retry repeated the same unusable response as its original. Further retries were stopped. An explicit diagnostic-continuation mode then completed the remaining original observer calls, retaining failures and preserving the distinction between known rejected content and an uncertain missing output. No missing human judgments were filled in.

The original job's immutable budget remains 48. A separate recovery job authorized at most one extra call per pair (12 maximum); only two were used. **60 is the combined ceiling, not the actual call count.** All weights remained frozen throughout prompt, parser and orchestration changes.

The [cost snapshot](../artifacts/20260908T022716_creative_ee890a8f/cost-summary.json) includes both earlier failed pilots and the final experiment with recovery: **59 model invocations plus 11 model-load operations**, 70 recorded operations total, with no open cost records. Their measured operation scopes total **5,633.7 seconds**; this is not total project elapsed time. Local provider API fees are **$0**. Electricity, hardware and total operating cost remain unknown because rates were not supplied. Failed and revalidated work is included.

Human selection and subsequent development are still pending. The current workflow is ready to create and compare reference-informed studies; the observer needs better output reliability and evaluation before it can be trusted as a judge. Training remains off.
