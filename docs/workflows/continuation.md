# Explicit selected-parent continuation

`kusama.continuation` connects an existing saved image to a frozen studio revision batch and a follow-up review collection. It uses the current `Studio`, controlled editor and SQLite invocation ledger. Preparing or reviewing work never launches inference.

The Python entry points are `ContinuationStore(root).prepare(id, request, config)` and `execute(id, backend_factory=FluxBackend, resource_root=None)`. The request is validated by `ContinuationRequest`; `ContinuationRequest.model_json_schema()` exposes the exact schema. The renderer configuration uses the existing `Config` contract, with its output directory and dimensions matching the artifact root and parent image.

Selection has two explicit paths:

- `origin="human_advancement"` requires a registered review `collection_id` and `advancement_submission_id`. The bridge loads the original append-only advancement, reviewer, reason, candidate and exact image hash. A preference vote alone cannot supply this event.
- `origin="operator"` or `"model_selection"` requires a path, SHA-256, attributed author and reason. This provenance never creates a human vote or human advancement.

The revision records its author and origin, instruction, intended change, preservation constraints, whole-work plan, inquiry, source IDs and connection context. It requests exactly two distinct seed realizations of the same intervention. Their plans contain the same saved parent hash. `Studio.Direction.realization_group` distinguishes seed repetitions from distinct artistic directions.

The bridge compiles `whole`, `masked` and `geometric` through the existing controlled editor. `whole` uses FLUX on the entire saved parent, so requested preservation is not guaranteed. `masked` uses FLUX followed by the existing hard compositor with an explicit normalized rectangular region. `geometric` uses a supplied, hash-bound object mask plus an explicit scale and anchor to transform actual parent pixels, while FLUX repairs the background. Decoded pixels outside that rectangle are preserved; object identity, local semantics, integration and edge quality remain uncertain. Raw outputs and raw spillover measurements remain in the controlled action artifacts. Unsupported parameters, unknown source IDs, altered bytes and invalid boxes fail validation.

Inputs are immutable in `jobs.sqlite3:continuation_specs`; the studio keeps authoritative project events and per-action ledgers in the same database. Readable projections live at `<root>/continuations/<id>/`: `inputs.json`, seed plans, `manifest.json`, `review_assignments.jsonl`, and `gallery.html`. The original image remains preserved in the studio project. Resume reuses committed raw bytes and does not repeat uncertain charged calls. A custom backend factory is for controlled substitutes or an independently budgeted worker; default real inference always uses the frozen shared `SessionBudget` root and exclusive model-workload lock. Session charges have stable keys `continuation:<id>:seed:<seed>`. Internal bounded runners may pass `before_dispatch` to `execute`; the callback is checked before backend loading and again immediately before each seed reservation. A refused callback authorizes no charge, and committed raw outputs can still be recovered without dispatch. The default callback is absent, preserving existing execution behavior.

Shared-session completion is settled after the controlled action commits hash-verified raw bytes. Known charged failures without raw bytes settle as failed; an interruption without committed evidence remains uncertain. Every status remains charged. `ContinuationStore(root).reconcile_session(id)` performs this accounting check without loading a model or dispatching inference, under the same exclusive locks. `session_accounting.json` records each seed's local evidence and session status. Legacy immutable completed/failed session records are never rewritten: a disagreement with current local evidence is disclosed in the sidecar. Neither a discrepancy nor reconciliation authorizes a replacement call.

The tested module commands are:

```sh
.venv/bin/python -m kusama.continuation --help
.venv/bin/python -m kusama.continuation --artifacts artifacts/overnight prepare --id example --request request.json --config renderer-config.json
.venv/bin/python -m kusama.continuation --artifacts artifacts/overnight execute --id example
.venv/bin/python -m kusama.continuation --artifacts artifacts/overnight show --id example
.venv/bin/python -m kusama.continuation --artifacts artifacts/overnight gallery --id example
```

`example` and the JSON filenames above are illustrative arguments, not claims of saved real model output. Use the actual prepared continuation ID from a run. The help command and Python equivalent preparation/execution paths were checked; rendering tests use explicitly declared substitutes.

The dashboard exposes `/api/continuations`, `/api/continuations/{id}`, and `/api/continuations/{id}/gallery`; source and inspiration browsing use `/api/sources`, `/api/source-regions/{id}` and `/api/inspirations`. Image handlers resolve registered IDs, verify hashes and reject path escapes. Dashboard reads do not create source/review/project databases.

After preparation, both the local gallery export and web gallery show the complete saved parent beneath an outlined preview of the exact saved `allowed_mask.png`. The gallery verifies the mask against both prepared action hashes, reports its pixel bounds (right and bottom exclusive), counts protected pixels, and displays the unchanged edit instruction. A full-image mask explicitly warns that zero pixels are protected. The overlay scales with the whole image and writes no image bytes. It describes the prepared geometry; it does not infer object boundaries or establish that the requested subject lies inside the editable region. Inspect this preview before explicitly executing a prepared continuation.

Local JSON writes require `X-Kusama-Review: 1` and the existing loopback/origin checks. `POST /api/continuations` accepts `ContinuationCreate` (`id`, typed `request`, and `renderer` configuration) and only prepares the action. `POST /api/continuations/{id}/execute` accepts `{"submission_id":"a-stable-request-id"}` and dispatches a separate worker for that registered ID, returning HTTP 202. The worker command uses a fixed executable/module and argument list without a shell. Busy shared model workloads and exhausted budgets return HTTP 409 before dispatch. Completed batches return their existing dispatch state without another model call.

`GET /api/continuations/{id}/status` returns both the existing studio execution state and durable dispatch history. Dispatch requests and append-only events are in the current `jobs.sqlite3`; worker logs are under the continuation directory. A lost worker lease can be reconciled by repeating the same explicit submission ID. The server does not automatically execute stale requests after restart. The controlled ledger remains authoritative about uncertain invocations and existing raw bytes. An accepted HTTP request is not evidence that rendering completed.

`POST /api/continuations/{id}/observations` records `artifact_id`, `note`, `author`, and explicit `origin` (`operator`, `model_observation`, or `development_example`). This route cannot write human preference labels. `/api/contracts` exposes current Pydantic schemas. The dependency-free `scripts/generate_contract_types.py` generates the checked-in `dashboard_static/contracts.d.ts`; `.venv/bin/python scripts/generate_contract_types.py --check` checks schema drift. Runtime bounds, hashes, budgets and provenance remain server validations.

New follow-up collections use stable, balanced A/B presentation and a statement-free condition. Original schema-1 collection assignments remain unchanged. Review records distinguish finished quality, development potential, readiness, willingness to develop, open reading, felt response, prior exposure and elapsed viewing time. `tie`, `unsure`, `incomparable`, `decline`, and null are distinct. Optional per-image answers carry both the presented position and underlying candidate hash. Browser timing is recorded as browser elapsed time, not independently measured attention.

Pool selections use separate append-only `review_pool_votes` records with choices `candidate`, `none`, `unsure`, or `decline`. The complete-pool expectation is frozen during registration with `pool_expected_count`. Optional `pool_candidate_ids` selects a subset: E0 can register a parent plus six children for parent/edit comparisons while offering only the six children in the pool. Incomplete designated pools remain inspectable and pair-reviewable, but reject standard pool-selection submission. Pair votes, pool choices and explicit advancement remain separate events. No model observation or absent human answer becomes a human label.

Verification covers exact parent and output hashes, unsupported controls, source IDs, seed grouping, idempotency, interrupted postprocessing, uncertain invocations, frozen session root, budget exhaustion, exclusive workload locking, balanced image order and situated-label mapping. These checks establish software behavior only. The evidence gate remains actual desired and unrelated visual changes followed by genuine independent human judgment of the whole work.

## Intention maps and decisions at whole/detail scale

Added 8 September 2026. New requests can opt into `intention_map`; old frozen requests keep their original serialized shape. The map describes the exact selected `parent_sha256` and attributes its contents to an `author` and `origin` (`operator`, `model_proposal`, or `development_example`). It records the whole-work intention, named elements with normalized regions, intended roles, subtlety, relationships to other element IDs, and what must survive. These are artistic hypotheses. Neither `resolved` nor an expected reading is a human quality label.

The map separates the next artistic decision (`target_ids`, `operation`, `attention_reason`, `expected_reading`) from the existing renderer instruction and rectangular mask. Operations name artistic intentions such as concealment or simplification; they are not extra renderer parameters. For newly prepared continuations, each target must be unresolved and its entire declared region must fit the edit rectangle; overlap alone is insufficient. Optional `required_support_ids` names other elements whose entire geometry must be editable for this particular intervention, such as both endpoints of a requested connection. Relationships alone do not imply editable supports. Explicit supports outside the mask cause rejection before any job or charge is registered. Non-target elements may be `resolved` or `leave_alone`. Setting `protect_pixels=true` rejects any proposed rectangle that intersects those pixels, including a whole-image edit. Text in `must_survive` alone cannot guarantee semantic preservation. Regions that become empty at the declared image dimensions fail before job registration.

Optional element `existence` records `status` (`present`, `absent`, or `uncertain`), `author`, `origin` (`operator`, `model_observation`, or `development_example`) and `reason`. The map's parent hash binds that observation to the selected image. Existing-object operations reject an explicitly absent target; `operation="add"` may target an empty region. Explicitly absent required supports are rejected. Missing or uncertain evidence remains uncertain; a bounding box never proves an object is present. An attributed model observation remains a model observation even when its status is `present`.

New continuation inputs include an immutable `preflight` report and a matching `preflight.json` sidecar. It records policy `continuation_control_preflight_v2`, actual crop and mask geometry, source-control hashes, protected pixel count and existence evidence/uncertainty. A passed preflight means the declared controls are internally consistent; it does not mean the semantic change happened. Old frozen continuations lacking this policy keep their original overlap validation on resume or identical preparation. No old inputs or policies are rewritten. `request_snapshot(request)` omits new absent/default fields to preserve the old serialized request shape.

For example, add this map to a request whose edit rectangle covers the chair and excludes the sleeve. Replace the placeholder hash with the selected parent's actual SHA-256:

```json
{
  "parent_sha256": "REPLACE_WITH_SELECTED_PARENT_SHA256",
  "author": "operator name",
  "origin": "operator",
  "whole": "An absence suggested through a relationship between figure and chair",
  "elements": [
    {"id": "chair", "region": [0.5, 0.4, 0.8, 0.9],
     "role": "Suggest someone's absence", "subtlety": "Discovered after the figure",
     "relationships": ["sleeve"], "must_survive": "Its relation to the sleeve pattern"},
    {"id": "sleeve", "region": [0.1, 0.2, 0.3, 0.4],
     "role": "Connect the remaining figure to the absent person", "subtlety": "Quiet repetition",
     "must_survive": "The resolved pattern marks", "status": "resolved", "protect_pixels": true}
  ],
  "target_ids": ["chair"], "operation": "revise",
  "attention_reason": "Test the relationship without disturbing the resolved sleeve",
  "expected_reading": "Viewers may experience absence; this requires an unprompted viewer response"
}
```

The existing initial-generation workflow supplies the whole composition. A prepared continuation tests one saved intervention with two seeds from the same parent, then registers statement-free human follow-up comparisons. The development gallery shows the full parent and each child plus matching enlarged views of the edit rectangle. Enlargement is inspection of existing pixels. Optional crop controls condition native-resolution diffusion on a bounded parent crop; they do not implement upscaled diffusion, layers, arbitrary outer authorization masks, or regional attention guidance. Existing Qwen proposal jobs are unchanged. An operator or a separately attributed model proposal must supply the map; the bounded autonomous development runner may supply model proposals through this same contract.

Use `ContinuationStore(root).decide(id, decision)` or `POST /api/continuations/{id}/decisions` to append a development decision. The JSON contract is `DevelopmentDecision` in `/api/contracts`. For example:

```json
{
  "submission_id": "chair-choice-1", "artifact_id": "root", "choice": "select",
  "author": "operator name", "origin": "operator",
  "reason": "Keep the parent; neither alternative strengthened the relationship",
  "local_assessment": "The pattern became more legible but too emphatic",
  "whole_assessment": "The chair drew attention away from the figure",
  "preservation_assessment": "The sleeve pixels survived; integration remains weak"
}
```

`artifact_id` must resolve to the saved parent (`root`) or a realized child from this continuation. `select` waits for the prepared batch to settle; choosing `root` retains the before image. `stop` can also cancel unexecuted work, without rendering. `origin` is explicitly `operator`, `model_selection`, or `development_example`; human advancement still uses the original review submission path. Decisions save the chosen and original parent hashes, the intention snapshot fingerprint, the reasons and local/whole/preservation assessments in the studio's append-only events. They create no human votes and start no additional work. Stable submission IDs are idempotent and reject changed content. Closed continuations require a new continuation to revisit.

```sh
.venv/bin/python -m kusama.continuation --artifacts artifacts/overnight decide --id example --decision decision.json
```

To continue, prepare a **new** continuation using the selected saved image and its exact hash. Set the new map's `previous_continuation_id` to the prior continuation and update `parent_sha256`. This validates that the new parent is an intact prior parent or child. Linked snapshots require an entry in `revisit_reasons`, keyed by element ID, before changing or removing any previously resolved, leave-alone or pixel-protected element. Reopening a protected target requires both `status="unresolved"` and `protect_pixels=false`, with that reason. Old images, maps, decisions and alternatives survive. The linkage is opt-in; legacy/map-free workflows do not acquire implicit protection, and choosing a prior alternative is an explicitly attributed branch choice.

Fixture checks cover protection, same-parent branching, linked reopening, snapshot immutability, stop without execution, idempotent decisions, unchanged review labels and matching crop views. The next evidence gate is an actual reviewed whole → local intervention → whole comparison: do the intended reading and preservation survive, and does the work improve? No inference or human feedback was produced by this implementation.

## Frozen crop and geometric controls

For `masked` or `geometric`, optional `revision.crop` supplies `[left, top, right, bottom]` in integer **parent pixels**, with exclusive right/bottom bounds. It must contain the entire authorized rectangle and have width/height at least 64 and divisible by 16. The executor receives the saved crop at native dimensions and composites into the original full-size image. No crop is silently widened or resized. Frozen renderer `Config` retains the full parent dimensions and other settings; `preflight.inference_dimensions` discloses the explicitly requested crop dimensions for each invocation. This is native crop conditioning, not high-resolution upscaling, a substituted named profile, or a validated throughput/quality gain. Whole-image methods reject crop controls instead of ignoring them.

Geometry requires `revision.method="geometric"` and `revision.geometric` containing `object_mask`, `object_mask_sha256`, `scale`, `anchor` (parent-pixel coordinates) and `repair_prompt`. Optional `repair_policy` contains integer `padding` and `feather`, with `0 <= feather <= padding <= 16`. Optional `secondary_mask` also requires `secondary_mask_sha256`. Masks must be nonempty binary grayscale images matching the parent. Both original and transformed object, including interpolation edges, must fit the authorized rectangle. Any secondary effects must also fit. The bridge snapshots exact control bytes before registering the two seed plans. Execution uses frozen copies if original files disappear; changed frozen masks fail closed. Supplied mask attribution is the revision author/origin, and geometry is not semantic segmentation evidence.

For a 128×128 parent, this revision-field excerpt explicitly requests a 64×64 native crop and cutout transform. Replace the path and hash with actual evidence; the supplied object mask must actually fit the original and transformed bounds. Existing instruction, author, intended change, visual checks, preservation constraints, whole-work plan and two seeds remain required.

```json
{
  "method": "geometric",
  "region": [0.25, 0.25, 0.75, 0.75],
  "crop": [32, 32, 96, 96],
  "geometric": {
    "object_mask": "/absolute/path/to/object_mask.png",
    "object_mask_sha256": "REPLACE_WITH_ACTUAL_MASK_SHA256",
    "scale": 0.5,
    "anchor": [56, 80],
    "repair_prompt": "Continue the empty background behind the transformed shape",
    "repair_policy": {"padding": 4, "feather": 2}
  }
}
```

`tests/test_continuation_controls.py` verifies native crop coordinates, unchanged frozen full-parent configuration, geometric pixel transforms, exact preservation, immutable mask bytes, absent-target handling, retained uncertainty and old frozen-policy replay. These fixtures demonstrate software behavior; edit fulfillment and observation accuracy on real unfamiliar work remain unvalidated.

## Observation-informed final intervention

`kusama.finishing.finish(run_directory, root=Path('artifacts/overnight'), stage='all')` spends the one reserved finishing allocation after the original initial, ordinary and protected allocations have settled and their charged attempt counts match the frozen session configuration. The default allocation is 6 initial + 4 ordinary + 4 protected + 2 final seed attempts. Failed or uncertain charged calls remain in those counts. The continuation ID is fixed to `<run>_finish`; callers cannot choose a second finishing ID to expand this allocation.

The declared operator policy selects the first protected continuation in the frozen proposal order and its first saved child. It is exploratory selection, with no human approval or quality ranking. The exact child must have a saved, attributed model or operator observation matching its ID and image hash. The proposal snapshots all such observations and admits the latest three, the saved whole-work inquiry and the same two connected source images. Later observations cannot enter this frozen input. Qwen has 512 output tokens and at most one explicitly charged schema repair to propose a local test through `episodes.LocalProposal`.

The saved proposal prepares one masked continuation with two seeds. It preserves original inquiry, reference and connection records and hashes the actual developed child as its new parent. Each resulting image can then receive a separate deterministic 256-token Qwen observation of visible changes and preservation failures. These observations never become human quality, potential or readiness labels. All final-stage work uses the shared resource cap and model-workload lock.

```sh
.venv/bin/python -m kusama.finishing --run artifacts/overnight/runs/RUN_ID --artifacts artifacts/overnight --stage all
```

`RUN_ID` is illustrative. Stages `propose`, `execute` and `observe` are independently resumable. Preparation and raw proposal responses live in `<run>/finishing`; the observer uses `<run>/finishing_observation`. Separate authoritative jobs and append-only event exports retain failures. `<run>/finishing.json` links the lineage, action, observations and charged-attempt accounting. The original `continuations.json` and original observation plan remain unchanged. Fixture verification covers all 16 charged renderer attempts, bounded schema repair, late-observation exclusion, altered-parent rejection, and interruption with no repeated charged seed; it does not establish real visual or artistic success.
