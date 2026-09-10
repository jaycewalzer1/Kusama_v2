# Research and online collection audit — 8 September 2026

The user requested an audit after three research runs stopped on action-contract errors. This audit inspected model actions, official collection search, metadata selection, acquisition, source/image hashes, contextual text, resource limits, immutable replay and reporting. New research plans use `independent-research-v5-source-audit`.

The full software suite passed **855 tests**, with two existing dependency deprecation warnings. Small live checks passed search, eligible image acquisition, hash verification and cached reopening for Met, Art Institute of Chicago and Cleveland. No Qwen or FLUX calls, training or human judgments were made by this audit. These checks do not establish model-directed discovery or artistic adequacy.

## Findings and repairs

| Finding | Change and verification |
| --- | --- |
| Independent research used Met's deprecated unpaginated search endpoint. | Migrated to `/public/collection/v1.1/search` with explicit `offset=0` and the requested limit. Object requests retain `/v1/objects/{id}`. Fixture request assertions and live acquisition passed. The separate corpus discovery path was already migrated. |
| A candidate's discovery query counted as searchable evidence about it. | Removed discovery-query text from local relevance scoring. A regression test proves unrelated results no longer match solely because they were found by that query. Original queries remain in provenance. |
| Candidate summaries omitted useful documented metadata and always said unacquired. | Exposed normalized maker, culture/place, date, medium and tags, plus acquisition state and acquired source ID. No inferred cultural identity or relevance judgment is added. |
| Malformed envelopes could appear to be successful empty searches; one malformed record could hide valid neighbors. | Validate envelope/list shape and retain raw responses. Exclude malformed records individually. Met `objectIDs: null` is accepted only with `total: 0`; returned object IDs must match the request. |
| One missing Met object could abort the whole result page. | Preserve individual 404/410 failures and continue within the bounded page. Cached failures are recognized without another request. Rate limits and server errors still surface as failures. |
| Reading an HTTP error body could throw before recording the original failure. | Retain the HTTP status and any body-read error, with conservative charges and no retry. Tests cover 429, 500, 503 and interrupted error bodies. |
| Large descriptions and tags could overflow the next prompt. | Bound search-result and recalled excerpts; mark shortened fields. Full snapshots/raw bytes remain intact. Existing tokenizer admission remains enforced. |
| New source snapshots bypassed the storage-budget check. | Check snapshot bytes before writing; an over-budget fixture writes no snapshot. Existing network and acquisition limits remain. |
| Backend summaries omitted acquisition states shown by read-only accounting. | Report acquisition states consistently, including in complete fixture research runs. |

Earlier fixes remain: bounded integer search limits, optional explanations, `note`/`reject` text in either original field, conditional action requirements and recent search mode/provider details. Operational fields, inspection and draft reinspection, source-set caps and unknown-field rejection remain enforced. No schema-repair calls or automatic inference retries were introduced.

## Live checks and accounting

The baseline checked `portrait`, one result and at most one acquisition per provider; all three passed. After repair, Chicago and Cleveland again passed acquisition and cached reopening. Met's migrated search returned a different top result without an eligible image, correctly excluded. A separately recorded `sunflowers` query then passed Met acquisition and cached reopening.

| Provider after repair | Result | Requests | Charged bytes |
| --- | --- | ---: | ---: |
| Met, `portrait`, limit 1 | Metadata record excluded: no eligible image | 2 | 2,046 |
| Met, `sunflowers`, limit 1 | `met_436524` acquired; hashes/replay verified | 3 | 211,058 |
| Chicago, `portrait`, limit 1 | `artic_11` acquired; hashes/replay verified | 2 | 198,422 |
| Cleveland, `portrait`, limit 1 | `cleveland_124089` acquired; hashes/replay verified | 2 | 164,335 |

Including the baseline: **16 network requests**, **1,028,760 charged bytes** and **six diagnostic acquisitions**, including repeated samples in separate stores. Each provider store had caps of four requests, one acquisition, 12 MiB network bytes, 2 MiB per metadata response, 4 MiB per image and 64 MiB storage. These are operator-authored audit probes, not artistic runs or model-selected references.

Preserved local evidence:

- [Baseline stores/results](../../artifacts/research_audit/20260908_before/)
- [Updated stores, code hashes and results](../../artifacts/research_audit/20260908_after/)
- [Additional Met acquisition](../../artifacts/research_audit/20260908_after_met_sunflowers/probe.json)
- [Aggregate accounting and original-run integrity](../../artifacts/research_audit/20260908_after/integrity.json)

These ignored artifacts require the local artifact tree and will not resolve in a fresh checkout.

## Verification coverage

The [audit tests](../../tests/test_research_source_audit.py) exercise search → acquire → reopen → draft → commit for each provider through the real engines with fixture API bytes and fixture model responses. They verify source IDs, attributed fields, charges and replay without dispatch. Acquired-ID bookkeeping passed and did not require repair.

The full suite additionally covers offline network denial, resource limits, unknown IDs, failed/unsettled work, rights/exclusions, forbidden hosts, original bytes, contextual passages and observed links, catalog imports, frozen indexes, reinspection, image capacity, rendering transfer, development adoption and dashboard integrity. Verification command: `.venv/bin/python -m pytest -q`.

All three previously reported runs still verify as `response_rejected`, respectively with three, one and four charged reasoning calls and no hypotheses. Every research-directory file hash matched before and after final read-only verification. No run was rewritten, resumed or repaired in place.

## Remaining limits and next evidence gate

APIs can return empty, irrelevant, unavailable or ineligible results. Rate limits, outages, schema changes, network failures and malformed model responses remain possible and will be recorded. The system does not silently turn a local search into an online search or retry charged work. Code verification cannot guarantee a future model choice or external service availability.

The [Met migration documentation](https://metmuseum.github.io/#migrating-from-v1search) announces retirement of the old endpoint on 1 October 2026 and warns that result sets/order differ. Its image-search filter does not replace checking an object's returned image and rights fields. Chicago's [API documentation](https://api.artic.edu/docs/) documents IIIF and search fields; Cleveland's [API documentation](https://openaccess-api.clevelandart.org/) distinguishes metadata availability from eligible images.

The online library remains bounded to these three collections plus explicitly configured contextual text. Modern works relevant to the user's question may be absent or lack admissible images. Metadata candidates are not viewed images, and acquisition is not evidence of artistic usefulness.

The next evidence gate is a newly planned, separately admitted real research run demonstrating valid searches, relevant inspected sources and useful committed hypotheses. The audit does not establish that gate or a learned artistic advantage.

## Follow-up: token admission failure

The user subsequently reported `studio-ad9ab0c1bce446549010`, whose fourth decision stopped before generation with 6,811 input tokens. The preceding audit's character-based excerpt bounds did not guarantee tokenizer admission. The error also incorrectly named the 8,192 combined ceiling as the input ceiling: reserving 1,800 output tokens leaves only 6,392 for input.

New plans use `independent-research-v6-token-admission`. A shared CPU input-preparation function performs identical chat templating, image resizing and tokenization for preflight and generation. Research now counts with the pinned cached processor before loading weights or reserving a model call. Compact JSON is tried first without content removal, followed by deterministic disclosed projections of optional evidence/history if needed. Full and admitted contexts are separately frozen. Protected problem text, schema constraints, source IDs, images and pending hypothesis remain intact. Unfittable context is retained and stops without an inference reservation; replay uses the saved projection. The director's error reports the actual input ceiling and output reserve.

A local processor-only check reproduced the exact saved prompt at **6,811 tokens**. The same context serialized compactly measured **5,872**, fitting the **6,392** ceiling with no removed fields. A stress check with six image slots (three existing diagnostic images repeated) measured **6,328**; this was tokenization only, not an artistic run or model inference. The original research directory remained byte-for-byte unchanged. See the [exact prompt check](../../artifacts/research_audit/20260908_token_admission/exact_saved_prompt_check.json) and [regression tests](../../tests/test_research_context.py).

Final verification passed **861 tests**, including admission-before-reservation, exact frozen-context delivery, projection invariants, corrected error messages and UI pipeline tests. The TypeScript contracts and documentation links also passed checks. This closes the reproduced admission failure. It does not establish future model schema compliance, relevant source selection or artistic quality.
