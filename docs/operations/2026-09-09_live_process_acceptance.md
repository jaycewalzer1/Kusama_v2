# Representative research-process acceptance — 9 September 2026

The earlier 922 passing checks establish software behavior, including realistic tokenization and archived transport replay. They are **not a complete live-process pass**: model choices, model loading/generation, museum responses and FLUX pixels were substituted. The user explicitly required tests representative of the entire process, so live process acceptance is now a separate mandatory release gate.

The later [nested-hypothesis incident](2026-09-09_nested_hypothesis.md) exposed a gap after this pass; its repair and verification supersede the current-code claim here. The successful v9 run remains historical evidence.

## Scope and pass condition

For the currently implemented New run panel, the process is browser submission → real worker → frozen authorization → real Qwen inquiry/search/source selection → live official collection acquisition → actual pixels supplied to Qwen → model-authored draft and immediate reinspection/commit → exact saved prompt → real FLUX under the named profile → saved images, ledgers, provenance and working inspection endpoints. No reference IDs, hypotheses or render prompts are supplied to make the live gate succeed. Empty, rejected, interrupted, partial or unrendered outcomes fail this gate, even when the software correctly preserves them.

The wider artistic studio also includes human critique, selection, selected-parent revision and follow-up review. Those remain separately attributed workflows and evidence gates; this panel run does not automatically execute them. A successful technical process is not a certification of artistic quality or coverage of every future problem/model decision.

## Reusable tests

- [Process verifier](../../scripts/verify_research_process.py): reads authoritative research and render evidence, frozen request/profile/implementation identities, source byte hashes, real Qwen runtime metadata, actual token accounting, committed reference pixels, exact prompts, output pixels/hashes, settled invocation budgets and live dashboard inspection/image routes. It never dispatches inference. It writes its report outside the original run and refuses to overwrite a prior result.
- [Process gate tests](../../tests/test_research_process_gate.py): explicitly reject a completed scripted panel run as live evidence and reject an empty terminal run. The live test consumes an actual completed run and production dashboard URL, without model/source substitutes.
- [Release command](../../scripts/check_research_release.py): enables the software, cached-processor, archived-incident and live-process checks together. A missing or skipped case cannot produce a successful release result. It verifies a completed live run; it never starts or retries one.

| Process boundary | Evidence required |
| --- | --- |
| Browser submission and worker admission | Actual form submission; sealed request, current implementation identity and resource configuration; fixture coverage for stop/admission controls |
| Search, acquisition and context | Three-provider transport paths, HTTP error recovery and retained bytes; real cached tokenizers including six images; actual live acquisition and image hashes |
| Model decisions and recovery | Actual Qwen runtime/tokens and raw responses; exact missing-argument regression, repeated-search/stagnation paths, oversized/truncated responses and prompt redrafting |
| Draft and commitment | Model-authored live hypothesis with its selected reference pixels on reinspection; multiple-direction and six-image scripted cases |
| Rendering and inspection | Actual FLUX output at the frozen dimensions/profile and exact prompt; settled charges, image hashes and production dashboard image/file endpoints |
| Interruption and restart | Injected crashes around durable responses, steps, source acquisition, checkpoint and draft; terminal replay must dispatch zero extra calls |

The deterministic cases deliberately exercise branches that a single live model trace may not choose. The live trace supplies the actual loading, generation, network and device behavior those cases substitute. Both layers are required; neither alone covers the process.

```sh
uv run --locked python scripts/check_research_release.py \
  --live-run artifacts/runs/ACTUAL_LIVE_RUN_ID \
  --dashboard http://127.0.0.1:8878 \
  --recorded-incident artifacts/runs/studio-87f61ec22af04bf3978b/research \
  --recorded-hypothesis-incident artifacts/runs/studio-809c0f1646f94714a85f/research \
  --recorded-context artifacts/runs/studio-51e0958560004701abc6/research/decision-009-source.json \
  --output artifacts/research_audit/NEW_UNIQUE_GATE_DIRECTORY
```

For a quick read-only check of just the live process:

```sh
uv run --locked python scripts/verify_research_process.py \
  --run-directory artifacts/runs/ACTUAL_LIVE_RUN_ID \
  --base-url http://127.0.0.1:8878 \
  --output artifacts/research_audit/NEW_UNIQUE_RESULT.json
```

## Live diagnostic

The assistant submitted the original user question from the failed acquisition/stagnation run using the actual browser form, under the user's instruction to test the entire process. The production dashboard and worker have no patched model or source boundaries. The new run is `studio-79ae36ca81194da6b8eb`, titled “Live process acceptance · original research question”. The original question/context and settings are reused; a new title/submission/run identity distinguish the diagnostic.

Frozen limits: 24 Qwen decisions / 196,608 reserved reasoning tokens, at most two directions with one image each, `flux_base4b_study_512_v1` at 512 × 640 and 50 steps, 120 minutes including a five-minute dispatch reserve, 48 source requests and 12 acquisitions. No paid APIs, training, old-run resume, silent retry, or profile substitution is admitted. Each attempt is a separate frozen diagnostic. Failed calls and old runs are never replayed; all attempts and their costs are reported below.

### First live result: failed

`studio-79ae36ca81194da6b8eb` failed after 10 Qwen calls, 28 live source requests (154,720 charged bytes), one acquisition and zero hypotheses/FLUX calls. Its exact last raw response was `{"action":"acquire","source_id":"met_268339"}`. Conditional argument validation rejected it before the tool-feedback path because acquisition requires `candidate_id`. The [live verifier failed](../../artifacts/research_audit/20260909_acceptance/live_failed_before_repair.json); it did not relabel the partial process as success. The original run remains unchanged.

### V9 correction and pre-dispatch gate

Twelve new regressions failed before correction. New `independent-research-v9-tool-argument-feedback` plans separate the typed response envelope from executable tool arguments. Missing conditional arguments become recorded tool errors before side effects. The next decision may correct the action within the existing call/token/time and stagnation caps; no argument, source choice or artistic instruction is supplied. Invalid JSON, unknown fields/actions, field type/range violations and malformed hypotheses still terminate without implicit retry. This new-plan policy does not migrate old rejected responses.

The new exact acquisition-error processor scenario runs against all three actual source backends with controlled HTTP/generation, bringing processor cases to 21. The original recorded 15-call incident remains a separate replay regression.

The assistant started `studio-df9ce5e0aa134d41ac09` before the pending processor result was inspected. That processor run exposed a 6,422-token context, 30 over the 6,392 limit; a later run with different generated identifiers fit, showing insufficient margin. The assistant requested a graceful stop and retained the second live attempt: one completed Qwen call, nine source requests (18,057 charged bytes), no acquisitions or rendering, panel `stopped`. The error and this sequencing mistake are retained. Redundant system instructions were shortened without altering protected problem/source/hypothesis data or increasing any token allowance. A complete software/processor gate must finish successfully before the next live start.

### Completed live result

After the shortened-prompt software gate finished with 939 passed and the sole live-evidence case pending, the assistant submitted a fresh browser run, `studio-8677ed3036854e3084a2`, titled “Live process acceptance · verified v9”. Its [pre-dispatch record](../../artifacts/research_audit/20260909_acceptance/live_protocol_v9.json) preserves the gate result and runtime hashes. No runtime source changed during this run.

The panel completed at **16:05:14 UTC** with 19 Qwen calls, 95,488 actual reasoning tokens, 13 source requests (167,742 charged bytes), one acquired reference, one model-authored committed hypothesis and one FLUX image. All 20 model invocations settled successfully. Qwen selected Met source `met_708024`; draft proposals citing unopened evidence were rejected. A later `draft` missing its hypothesis also became recorded tool feedback, and Qwen subsequently committed its own proposal. Research ended at the fixed synthesis checkpoint with status `stagnated` and one committed direction; two directions was a maximum, not a promise of two results. The completed panel rendered that committed direction.

The [saved image](../../artifacts/runs/studio-8677ed3036854e3084a2/render/image-000.png) is RGB, 512 × 640 under the original study profile, seed 23, SHA-256 `24ee4832ac7ab84908167253e8aed7cfdd3890825087a1d6733cec42cc5d15c1`. Its measured rendering duration was 281.45 seconds. The progress bar reached 50 diffusion steps before GPU completion and image saving; an [in-flight verifier result](../../artifacts/research_audit/20260909_acceptance/live_v9_result.json) correctly failed while the renderer charge was unsettled. The [completed verifier](../../artifacts/research_audit/20260909_acceptance/live_v9_completed.json) subsequently passed all checks without another dispatch. The assistant opened Research, Images and Files & logs in the actual browser, verified the output's natural dimensions, and visually inspected the saved image. That inspection is not a human quality vote.

Across all three attempts, the [authoritative totals](../../artifacts/research_audit/20260909_acceptance/live_attempts_verified.json) are **30 Qwen calls, one FLUX call, 144,733 actual reasoning tokens, 50 source requests, two acquisitions and 340,519 charged source bytes**. One hypothesis and one image resulted. These totals include the failed and stopped attempts. An earlier totals file counted only direct manifest images and is explicitly superseded: panel outputs live in the declared render stage. The [eight prior archives](../../artifacts/research_audit/20260909_acceptance/preservation_v9.json) retain their original plans, raw responses, statuses and counts.

### Combined release result

The [release result](../../artifacts/research_audit/20260909_acceptance/release_v9/result.json) is **passed: 940 tests, zero failures, zero errors, zero skipped**, in 73.98 seconds. It includes 21 cached-processor cases, the preserved 15-call incident replay and the completed unmocked panel process. Two dependency deprecation warnings remain. The gate itself dispatched no generation. Generated TypeScript contracts and diff whitespace checks also passed.

This establishes completion of one actual current-panel process plus the deterministic failure/recovery branches above. It does not establish a success rate across new briefs, future collection responses or arbitrary model choices. The next artistic gate is assessment of the saved work and its source interpretation, followed by an explicitly selected-parent revision and follow-up review. No human votes, advancements or training updates were created. Original evidence remains in each run directory. Local artifact links do not resolve in a fresh checkout.
