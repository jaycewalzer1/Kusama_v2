# Bounded visual development

This workflow implements the September 8 technical sequence: measure saved editing failures, expose the editor's controls through continuation, execute a bounded proposal/observation/selection loop and use measured failures to choose a next experiment. Human judgment is outside this milestone. Qwen and FLUX remain frozen.

New requests can opt into [research during development](independent_research.md#keep-research-open-during-development): Qwen decides whether the actual parent warrants another inquiry, then adopts saved image or text references before choosing an intervention. This has separately frozen controller/research allowances; the default loop and existing plans keep their original budgets.

New v7 plans additionally let Qwen attach selected source images to its edit instruction: up to two for whole-image edits or one for masked edits. The actual parent remains the first FLUX image input. See [image prompts](image_prompts.md) for the model choice, frozen evidence and verification boundary.

## Existing components and new connections

`kusama edit-benchmark` evaluates hash-bound saved parent/child pairs. It distinguishes target existence, intended change, protected pixels, model assertions and unavailable measurements. The [benchmark guide](../evaluation/edit_benchmark.md) describes the historical cases, prospective task set and limitations.

The [continuation bridge](continuation.md) now exposes explicit native-pixel crops and geometric editing alongside whole and masked edits. New preflight checks require complete declared target/support coverage, retain uncertain existence reports, reject known absent existing-object targets and preserve linked protected intentions. Old frozen requests retain their original validation policy. A native crop changes the explicitly recorded inference dimensions without changing the full parent canvas or silently selecting another render profile. There is no high-resolution upscaling or autonomous segmentation.

`kusama develop` implements `autonomous.py`. The default loop used by this milestone contains an exact parent, at most two source images and their context, a fixed external brief, inquiry, external requirements, explicit renderer configuration and named profile, two distinct seeds per round, one through four rounds, a wall-time admission bound, and a separately admitted resource root. Sources are copied byte-for-byte; originals and original paths remain recorded. Each plan freezes implementation/lock hashes, schemas, prompts, the optional external memory snapshot and the admitted text excerpt.

Each round performs:

1. A Qwen proposal using the actual parent, references, preceding round records and permitted external memory. The output supplies an intervention and intention map, or an explicit stop reason.
2. Preflight against the exact parent, model-declared supports, external pixel annotations and any linked protected decisions. A rejected region consumes its proposal call but no renderer call. A remaining round can propose a recorded recovery from the same parent; no missing artistic instruction is synthesized.
3. Exactly two declared seed slots through `ContinuationStore` and the existing controlled editor. Both use the same saved parent. Raw outputs, composed outputs, masks, crops and execution failures remain saved.
4. A separate Qwen observation call on the actual parent and available children, including native detail crops. Mechanical checks remain separate from the model's fallible change assessment.
5. A separate Qwen selection call. It may retain the parent, select a child, continue, recover or stop. Selections are explicitly model-attributed and never become human votes or automatic artistic acceptance.

All unresolved cases remain unresolved. Exact no-change can fail a requested visible-change check; changed pixels cannot establish that the intended semantic relationship appeared. Exact preservation applies only to declared pixels. Observation, selection, quality, development potential and readiness remain distinct.

## Planning and execution

For a standalone loop, supply a validated `LoopRequest` JSON file. Its `render_config` is the existing full `Config.to_dict()` contract, with `output_dir` equal to the new loop directory and `local_files_only=true`. `resource_root` must point to an independently admitted session at execution time. Planning never creates one.

```sh
.venv/bin/python -m kusama.cli develop plan \
  --output artifacts/development/example --request /absolute/path/loop-request.json
.venv/bin/python -m kusama.cli develop preflight \
  --output artifacts/development/example
.venv/bin/python -m kusama.cli develop run \
  --output artifacts/development/example
```

The plan seal and ledger reject changed executable inputs. Execution reads the existing resource session before opening any writer, checks its named profile and device, then binds that session snapshot. Neither a missing session nor an empty SQLite file authorizes a default session. The old overnight session remains expired. Each new live experiment needs a new explicit resource contract; preparation and an available image are not dispatch.

In the default loop used here, renderer attempts are bounded by `2 * max_rounds`; Qwen calls are bounded by `3 * max_rounds`, each reserving at most 8,192 total tokens. A new Qwen call checks the local deadline both before and after loading. Each new renderer seed checks the local and shared-session admission boundaries before reservation. Already saved charged responses can reconcile without being invoked again. A call admitted before a deadline may finish after it; the boundary prevents further dispatch, not destruction of in-flight evidence.

No automatic schema repair or renderer retries are added here. A malformed proposal or observation is retained and stops that path. A failed preflight can inform the next bounded artistic action. Uncertain charged inference must reconcile its original evidence before further work. Policy/code changes require a new plan; they cannot be applied to an old frozen request.

## External controls and memory

An external requirement may include `regions`, each with an `id`, integer parent-pixel `box`, and role `required_edit` or `protected`, plus attributed origin/reason. Unknown roles fail validation. These are operator-supplied constraints, not model-discovered geometry. The model cannot omit them by leaving them out of its own map.

New `bounded-development-v2-coordinate-contract` plans advertise every region and element coordinate as a number in 0..1, and distinguish optional integer-pixel crops. Every proposal receives the actual parent dimensions and exact normalized equivalents of the original external pixel annotations. This deterministic unit conversion supplies no artistic instruction. It was added after a live v1 baseline emitted pixel coordinates in normalized fields; original responses and charges remain retained. Invalid baseline proposals now stop immediately instead of repeating an unchanged request. New coordinate contracts apply only to newly frozen plans.

`bounded-development-v3-preservation-contract` retains those controls and admits at most eight preservation constraints. A live v2 response correctly retained five constraints from its external brief and was rejected by the earlier four-item bound. V3 preserves the complete list; it does not truncate constraints or revalidate the old charged response under a changed frozen plan.

V4 additionally makes membership of target/support IDs in the proposed element map explicit. `bounded-development-v5-no-change-observation` permits an empty `visible_changes` list: an observer that sees no change must be able to say so. A live V4 observer accurately reported uncertainty with empty lists and was rejected by the former one-item minimum. Its original failed charge remains preserved; a disclosed, hash-bound V5 revalidation sidecar adds no model call and changes no original response.

Memory-on admits at most four native, preceding, non-review development events. The candidate set, selection, cutoff, payload identities and evidence hashes are frozen. The loop validates selected projections against their saved memory authority and never queries again on resume. Within-loop history exists in both memory-on and memory-off; the ablation concerns additional external experiment memory. The primary memory-off result is reused as the ablation control, with correlated analyses and no duplicated computation. An empty query is an empty treatment, not evidence about memory usefulness.

## Outputs and verification

Each standalone loop saves `plan.json`, `plan_identity.json`, byte-preserved `inputs/`, decision-time input/response records, `details/` with crop lineage, immutable `round-NN.json`, continuation manifests and galleries, a projected `report.json`, and authoritative SQLite events. The report includes measured/unknown costs, resource calls, selected image bytes and explicit null artistic judgments.

```sh
.venv/bin/python -m pytest -q tests/test_edit_benchmark.py tests/test_continuation_controls.py \
  tests/test_autonomous.py tests/test_autonomous_safety.py
.venv/bin/python scripts/generate_contract_types.py --check
```

Software substitutes establish bounded execution, preserved lineage and failure accounting. Cached historical diagnostics establish only their reported measurements. Fresh image execution and improved control on new development examples are separate evidence gates. The broader artistic evidence gate remains outside this technical milestone.

The [8 September live handoff](../operations/2026-09-08_bounded_development_handoff.md) records all eight attempted comparison conditions, the one completed proposal/edit/observation/parent-retention cycle, rejected proposals and a concurrent-source-change limitation. A completed bounded cycle does not imply that the requested edit succeeded. The standalone recovery exercises did not complete a multiround development sequence.

Local artifact links require the preserved ignored artifact tree and do not resolve in a fresh checkout.
