# Nested hypothesis rejection — 9 September 2026

The user reported another fatal `response_rejected` after the 940-test release gate. The newest run, `studio-809c0f1646f94714a85f`, started at 16:58:49 UTC and failed at 17:06:46 UTC after 13 Qwen calls. Its final valid JSON draft omitted `hypothesis.sources`. The nested Pydantic contract still rejected that object at the response boundary, although v9 allowed missing top-level action arguments to reach tool feedback.

A second defect made the outcome worse: decision 6 had already committed `hypothesis_001`, but the panel rejected the whole research-to-render handoff when decision 13 failed. The saved commitment and original reference bytes remain intact. No repair of the failed response, old-run resume or fabricated missing source is performed by this correction.

## New-plan behavior

`independent-research-v10-hypothesis-feedback` preserves invalid hypothesis objects as arguments and validates the full strict hypothesis contract before any tool side effect. Missing fields, invalid nested types, empty/duplicate sources, extra nested fields and length violations return concise, recorded validation feedback. The next ordinary decision may correct them within the existing call, token, deadline and stagnation limits. Valid hypotheses retain their existing normalization. The model-facing target schema remains strict; no source, title or artistic instruction is filled in.

Invalid JSON, unknown top-level actions/fields and invalid top-level types remain terminal rejections. If earlier hypotheses were committed, new panel runs render only those validated commitments and finish **partial**, keeping the rejection and exact response filename visible. With no commitment, the run fails and reports its underlying error. The strict live release verifier continues to reject partial runs.

## Regression evidence

Eighteen new nested-argument regressions failed before correction. Two cached-processor panel scenarios also reproduced the failure after a prior commitment: the exact omitted-sources response and truncated JSON. They verify actual tokenization/image inputs, preserved errors, original invocation budgets, unchanged committed render prompts, dashboard image visibility and zero-dispatch terminal replay. The truncated JSON case must remain partial rather than becoming a green success.

A separate replay uses all 13 original model responses and retained source HTTP bytes, remapping only timestamp-dependent acquisition candidate IDs. One explicitly assistant-authored stop action follows the incident; it is not evidence of a future model choice. The replay verifies that the earlier committed prompt reaches rendering and the exact final draft becomes tool feedback. Original JSON hashes are checked before and after.

The release verifier also inspects sibling runs with the same frozen implementation identity. Failed, partial, missing-manifest or still-running cases prevent a green result based on an older success; explicitly stopped runs remain listed but are not counted as failures. Earlier-version failures stay preserved and disclosed separately.

The pre-dispatch software gate passed **960 tests**, with only the live-evidence check pending. A separate unchanged-budget browser run, `studio-51e0958560004701abc6`, was then started using the original question. Four additional release-gate regressions cover failed/partial/in-progress sibling runs.

## Live v10 result and context correction

The v10 live check did **not** pass. It demonstrated the new nested-field behavior twice: drafts missing `hypothesis.relationship` became tool errors and research continued. It then failed before decision 10 because accumulated search/candidate context required 6,661 input tokens against the 6,392 ceiling. Nine Qwen calls were charged, no tenth call was reserved, and no hypothesis or image resulted. The run is retained without resume or replacement of its evidence.

New `independent-research-v11-context-deduplication` plans add two optional disclosed projections. Stage 5 removes search-card transport URLs/repeated annotations and references matching candidate metadata in the existing search cards; all operational IDs stay present. Stage 6, when needed, bounds optional title/maker/date/medium/culture excerpts. Images, problem, current question, prior context, pending hypotheses and committed hypotheses are unchanged. Full source/decision records remain saved. The original token allowance and output reserve remain unchanged, and stages 0–4 retain their historical behavior.

The preserved context replays at 6,652 tokens under stage 4 (JSON key ordering affects token merges), still over the same limit; stage 5 admits it at **6,235 tokens**. A variant with long metadata also passes. These cached-processor regressions preserve all source/candidate IDs and protected inputs. An intermediate test incorrectly asserted equality with the live 6,661 count; that failed result is retained and the regression now checks the actual admission boundary rather than key-order-sensitive equality. No model inference was used to repair or test the context projection.

## Completed v11 live verification

After the final pre-dispatch gate passed **967 tests** with only the live test pending, a fresh browser run, `studio-29fd86d2de4540338c92`, executed the original question under the unchanged 24-action/two-direction/one-image-per-direction limits and study profile. It completed at **17:43:41 UTC**. Qwen used 16 calls and 92,382 actual tokens, acquired one reference through three source requests, and committed “Fragmented Sovereignty: The Crown as Collective Memory” at the final synthesis checkpoint. Research closed as `stagnated` with one commitment; the two-direction setting was a maximum. One actual FLUX image was saved and all 17 invocations settled successfully.

The [live verifier](../../artifacts/research_audit/20260909_nested_hypothesis/live_v11_completed.json) passed, including current-implementation outcomes, source bytes, committed reference pixels, frozen prompt/profile, image hashes, accounting and real dashboard endpoints. The assistant opened the image in the actual dashboard and verified 512 × 640 natural dimensions. The [image](../../artifacts/runs/studio-29fd86d2de4540338c92/render/image-000.png) has SHA-256 `5ee54a9ab1dae438be3bfdaadecd71e756a1d33e6a23c26aa874a25b154f5a8f`, seed 23, and measured render runtime 172.85 seconds. No artistic-quality vote is implied.

The [final evidence](../../artifacts/research_audit/20260909_nested_hypothesis/final_evidence.json) includes the user's original failed run and both new diagnostics. The two diagnostics together used **25 Qwen calls, one FLUX call, 139,836 actual reasoning tokens, seven source requests, two acquisitions and 230,627 charged source bytes**. The original run's 63 JSON evidence files retain their pre-audit hashes; its valid commitment was not lost or rewritten.

The [combined release result](../../artifacts/research_audit/20260909_nested_hypothesis/release_v11/result.json) passed **968 tests, zero failures, zero errors and zero skips**, in 84.35 seconds. It includes 26 cached-processor cases, both full incident traces, the saved overflow and long-metadata variant, and the completed unmocked run. Two dependency deprecation warnings remain. Generated TypeScript contracts, documentation links and diff whitespace checks passed. All eight earlier research archives also remain unchanged and readable. Use the [release command](../../scripts/check_research_release.py) with all three retained incident inputs; it performs verification without starting inference.

Verification outputs are retained in `artifacts/research_audit/20260909_nested_hypothesis/`. These tests establish software behavior and one completed live process, not artistic quality or a guaranteed success rate across briefs. The remaining artistic gate is review of the saved work and its source interpretation, followed by explicitly selected development and follow-up review. No human labels or training updates were created. Local artifact links require this artifact store and will not resolve in a fresh checkout.
