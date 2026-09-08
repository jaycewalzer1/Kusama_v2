# Explicit selected-parent continuation

`kusama.continuation` connects an existing saved image to a frozen studio revision batch and a follow-up review collection. It uses the current `Studio`, controlled editor and SQLite invocation ledger. Preparing or reviewing work never launches inference.

The Python entry points are `ContinuationStore(root).prepare(id, request, config)` and `execute(id, backend_factory=FluxBackend, resource_root=None)`. The request is validated by `ContinuationRequest`; `ContinuationRequest.model_json_schema()` exposes the exact schema. The renderer configuration uses the existing `Config` contract, with its output directory and dimensions matching the artifact root and parent image.

Selection has two explicit paths:

- `origin="human_advancement"` requires a registered review `collection_id` and `advancement_submission_id`. The bridge loads the original append-only advancement, reviewer, reason, candidate and exact image hash. A preference vote alone cannot supply this event.
- `origin="operator"` or `"model_selection"` requires a path, SHA-256, attributed author and reason. This provenance never creates a human vote or human advancement.

The revision records its author and origin, instruction, intended change, preservation constraints, whole-work plan, inquiry, source IDs and connection context. It requests exactly two distinct seed realizations of the same intervention. Their plans contain the same saved parent hash. `Studio.Direction.realization_group` distinguishes seed repetitions from distinct artistic directions.

Only `whole` and `masked` compile through this bridge. `whole` uses FLUX on the entire saved parent, so requested preservation is not guaranteed. `masked` uses FLUX followed by the existing hard compositor with an explicit normalized rectangular region. Decoded pixels outside that rectangle are preserved; local semantics and edge quality remain uncertain. Raw outputs and raw spillover measurements remain in the controlled action artifacts. Unsupported parameters, unknown source IDs, altered bytes and invalid boxes fail validation.

Inputs are immutable in `jobs.sqlite3:continuation_specs`; the studio keeps authoritative project events and per-action ledgers in the same database. Readable projections live at `<root>/continuations/<id>/`: `inputs.json`, seed plans, `manifest.json`, `review_assignments.jsonl`, and `gallery.html`. The original image remains preserved in the studio project. Resume reuses committed raw bytes and does not repeat uncertain charged calls. A custom backend factory is for controlled substitutes or an independently budgeted worker; default real inference always uses the frozen shared `SessionBudget` root and exclusive model-workload lock. Session charges have stable keys `continuation:<id>:seed:<seed>`.

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

Local JSON writes require `X-Kusama-Review: 1` and the existing loopback/origin checks. `POST /api/continuations` accepts `ContinuationCreate` (`id`, typed `request`, and `renderer` configuration) and only prepares the action. `POST /api/continuations/{id}/execute` accepts `{"submission_id":"a-stable-request-id"}` and dispatches a separate worker for that registered ID, returning HTTP 202. The worker command uses a fixed executable/module and argument list without a shell. Busy shared model workloads and exhausted budgets return HTTP 409 before dispatch. Completed batches return their existing dispatch state without another model call.

`GET /api/continuations/{id}/status` returns both the existing studio execution state and durable dispatch history. Dispatch requests and append-only events are in the current `jobs.sqlite3`; worker logs are under the continuation directory. A lost worker lease can be reconciled by repeating the same explicit submission ID. The server does not automatically execute stale requests after restart. The controlled ledger remains authoritative about uncertain invocations and existing raw bytes. An accepted HTTP request is not evidence that rendering completed.

`POST /api/continuations/{id}/observations` records `artifact_id`, `note`, `author`, and explicit `origin` (`operator`, `model_observation`, or `development_example`). This route cannot write human preference labels. `/api/contracts` exposes current Pydantic schemas. The dependency-free `scripts/generate_contract_types.py` generates the checked-in `dashboard_static/contracts.d.ts`; `.venv/bin/python scripts/generate_contract_types.py --check` checks schema drift. Runtime bounds, hashes, budgets and provenance remain server validations.

New follow-up collections use stable, balanced A/B presentation and a statement-free condition. Original schema-1 collection assignments remain unchanged. Review records distinguish finished quality, development potential, readiness, willingness to develop, open reading, felt response, prior exposure and elapsed viewing time. `tie`, `unsure`, `incomparable`, `decline`, and null are distinct. Optional per-image answers carry both the presented position and underlying candidate hash. Browser timing is recorded as browser elapsed time, not independently measured attention.

Pool selections use separate append-only `review_pool_votes` records with choices `candidate`, `none`, `unsure`, or `decline`. The complete-pool expectation is frozen during registration with `pool_expected_count`. Optional `pool_candidate_ids` selects a subset: E0 can register a parent plus six children for parent/edit comparisons while offering only the six children in the pool. Incomplete designated pools remain inspectable and pair-reviewable, but reject standard pool-selection submission. Pair votes, pool choices and explicit advancement remain separate events. No model observation or absent human answer becomes a human label.

Verification covers exact parent and output hashes, unsupported controls, source IDs, seed grouping, idempotency, interrupted postprocessing, uncertain invocations, frozen session root, budget exhaustion, exclusive workload locking, balanced image order and situated-label mapping. These checks establish software behavior only. The evidence gate remains actual desired and unrelated visual changes followed by genuine independent human judgment of the whole work.
