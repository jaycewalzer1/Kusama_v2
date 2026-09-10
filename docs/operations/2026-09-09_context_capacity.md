# Research context capacity — 9 September 2026

## Incident and cause

`studio-b70366976547419586e8` stopped before its third decision with a reported 8,521 input tokens after the old projections, exceeding the 6,392-token input allowance. Its question and operator context were short. The preceding remote search returned **21 candidates** across the three providers. The prompt duplicated candidate metadata and its final projection still included all 21 search cards, provider snapshots and exclusion detail. Bounding individual fields did not bound the combined result set.

Read-only ledger inspection verified **two completed charged Qwen calls**, **8,083 measured tokens**, no third reservation and no images. All **14 registered research artifacts** matched their recorded hashes. The original failed manifest, responses, request, resource session and source snapshots remain unchanged.

The exact saved decision context is retained as the portable [overflow fixture](../../tests/fixtures/research/search_context_overflow.json), SHA-256 `0dd974b07812cd4b1aaba0d9075093950ba9dbb4ca36d7473b25e26d3f5b0d7d`.

## New behavior

The context-capacity change was introduced as v17 and is retained in the combined v18 research implementation, alongside the separately developed optional chance-exposure feature.

- After the existing projections, token-counted pages narrow to 8, 4, 2 or 1 candidate as needed. Pages interleave providers while preserving each provider's encounter order. This is an operator-authored access policy, not a relevance judgment.
- Every visible page carries its total, offset and an exact `recall` cursor. `recall_steps`, `offset` and `limit` retrieve another page of the original saved search without another network request. No source is deleted or treated as rejected because it is absent from the current page.
- Tool transport details and repeated metadata receive disclosed excerpts. Shortened source-text passages retain the correct next character offset. Source and link IDs remain usable. The separate chance-exposure candidates remain unchanged through every projection.
- Tighter stages remove schema display annotations while preserving property names and validation constraints. The operator's problem/context, current question, pending hypothesis, source images and their hashes remain intact.
- If essential inputs still cannot fit, `ContextCapacityPause` saves a registered capacity checkpoint and returns `context_paused`. It does not throw the old fatal context error, reserve a call, invent a model response or add an uncharged research step. Terminal replay reconstructs the checkpoint without repeating inference. The panel preserves the session as partial, and available committed work remains available for rendering. This fallback is an unresolved capacity condition, not a successful artistic run or a model-requested human review.

At this repair milestone, the combined 8,192-token ceiling, 1,800-token output reservation, model revision, renderer profile and invocation budgets were unchanged. The subsequent user-requested [v19 context increase](2026-09-09_expanded_context.md) raises the ceiling for new plans explicitly. The controller does not truncate a hypothesis or quietly raise the budget to fit it.

## Verification

The pinned cached processor admitted the **saved failing context at 5,854 tokens**, with eight visible candidates and access to all 21. A reconstruction using the current system policy admitted at **5,930 tokens**. Both fit below 6,392 without changing the original question. Parsing the saved JSON and serializing it again reproduced the old final projection at 8,523 tokens; that replay count is kept separate from the original recorded 8,521-token failure.

The local diagnostic receipt and admitted contexts are in `artifacts/diagnostics/studio-b70366976547419586e8-context-capacity/`. These generated files are not in Git and will not exist in a fresh checkout. This diagnostic made no model calls, network requests or changes to the original run.

`KUSAMA_PROCESSOR_ACCEPTANCE=1 .venv/bin/python -m pytest -q` completed with **1,063 passed, five skipped** in 101.72 seconds. The skips require a live unmocked run or separately selected historical archives. The portable overflow replay ran. Verification covers actual cached-processor admission and scripted dispatch of the saved overflow context, complete provider-interleaved page traversal, narrowed recalled pages, schema constraints, text offsets, preserved charges, crash reconstruction and a panel capacity pause before any inference. A separate panel case pauses after one committed hypothesis and still renders its two planned images, retaining partial status and the original four research calls. The final integration run's research implementation hashes remained unchanged throughout the run.

Integration also covered the concurrently introduced chance-exposure feature. Its separate two-candidate shelf remains unchanged through all projections; paging does not silently discard that feature. Both dashboard processes were refreshed after confirming that no studio worker was active. Read-only contracts and start-options requests verified the refreshed processes; no start-run request was submitted.

The next artistic evidence gate remains a real run reaching grounded provisional studies, actual-image inspection and human assessment. Context admission and software continuation do not establish better artistic choices.
