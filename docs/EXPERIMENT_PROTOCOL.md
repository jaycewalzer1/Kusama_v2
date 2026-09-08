# E0 protocol and current execution boundary

Final overnight assessment: 8 September 2026, 13:48 UTC. All workers have stopped. The session deadline was 12:40:47 UTC, with new dispatch blocked after 11:55:47 UTC. E0 human evaluation has not begun; unused image/token capacity does not reopen the expired window. [Project direction](project_direction.md) remains authoritative about the research claims.

The [configuration](../configs/e0.yaml) declares four user-supplied development examples and eight assistant-authored provisional holdouts. The preparation command requires an external holdout file or a pre-existing sealed manifest; no holdout text is embedded in source code. It freezes separate sealed brief data, a public manifest containing only holdout IDs/hashes, and 252 review assignments. Holdouts are constructed by this assistant; independent authorship and strong generalization remain unestablished. Development retrieval must never index the sealed directory. The episode accessor exposes a sealed brief only for its own `heldout_execution` episode ID.

The planned E0 contains 12 parents and 72 edits: three alternative actions per parent, each realized at seeds 23 and 41. The conditions are inquiry/relationship, ordinary critique-and-edit, and exploration. Three reviewer slots receive 216 parent/edit assignments and 36 six-candidate-or-none pool assignments. Each pair assignment supports separate quality, development-potential and readiness questions. Multiple questions and image seeds remain correlated observations within a brief; assignment counts do not count independent trials.

The local [prepared manifest](../artifacts/overnight/evaluation_v2/e0/manifest.json) and [assignments](../artifacts/overnight/evaluation_v2/e0/review_assignments.jsonl) are immutable plans. Their preparation status and the original [empty status snapshot](../artifacts/overnight/evaluation_v2/e0_status.json) are historical, not the current execution authority. The real runner completed d01, d03 and d04: each has one parent and six edits, a live profile gate and registered review assignments. d02 has one saved parent and three original Qwen responses; one response supplied `preserve` as a string. Its six edits remain unexecuted. Thus **22 actual E0 images exist: four parents and 18 edits**. The protocol status reports three frozen parents and 18 edits, excluding d02's not-yet-frozen parent; its 63 missing protocol images differ from the **62 additional renders** needed to complete all 84 outputs. All eight heldout briefs remain unexecuted and unadmitted. Current status is in [the final protocol snapshot](../artifacts/overnight/evaluation_v2/e0_status_final.json) and the current manifests/ledgers.

The nine real review collections contain zero human pairwise votes, zero pool choices and zero advancements. d01/d03/d04's completed job-status projections were reconciled from hash-verified charged evidence in explicit `audits/job_status_reconciliation_v1.json` sidecars, with no model calls. The original failures, requests, attempts and charges remain. [Queued work](../artifacts/overnight/remaining_work_final.json) records the resource and migration gates. Existing exploratory paintings are not relabeled as E0. Local artifact links require the separately preserved artifact root.

A runner must first freeze references and prior memory, generate/hash the parent, validate all three action requests against the actual operation compiler, and call `freeze_episode` with the matched render profile and shared seeds. Heldout episodes additionally require a recorded development configuration freeze. The parent bytes are verified; changing a frozen record is rejected. The execution adapter must reserve measured wall/GPU time, reasoning tokens and image attempts through the existing execution/resource ledger before dispatch. The evaluation module records already-authorized attempts; it is not a bypass around those contracts.

The missing-output rule permits one identical-request retry after technical failure, with uncertain execution reconciled first. Both executions count. A valid image cannot be replaced because it appears weak. An unresolved pair is excluded from completed-pair analysis and its six-candidate pool remains incomplete. Inspecting that pool separately cannot count as a standard complete-pool selection. Frozen requests and attempts are immutable. The separate E0 runner now registers the exact available parent/edit pairs and a six-child-or-none pool in the real review ledger. A missing output leaves its pair excluded and its pool ineligible for standard selection. Real-label statistical analysis is still pending; the protocol planner’s `protocol_status` identifies its own store scope, while actual human responses live in `ReviewStore`.

Tested commands from the repository root:

```sh
.venv/bin/python -m kusama.evaluation prepare --config configs/e0.yaml --output artifacts/overnight/evaluation_v2/e0
.venv/bin/python -m kusama.evaluation status --protocol artifacts/overnight/evaluation_v2/e0
.venv/bin/python -m pytest -q tests/test_evaluation.py tests/test_datasets.py tests/test_learning.py
```

Preparation is idempotent when bytes agree. The source is [evaluation.py](../src/kusama/evaluation.py). Its offline tests verify exact counts, deterministic image-order assignment, sealed access, frozen parent integrity, failure accounting and retry limits. They do not exercise image generation or human evaluation.

The next execution gate requires an explicitly authorized follow-up resource contract and an audited bridge from the expired session. Within that contract, resolve the malformed development response without replacing valid actions, complete all four development pools, and freeze technical/reference/memory inputs before testing full heldout resource admission. The existing CLI does not automatically migrate an expired budget or extend its deadline. Genuine human comparison remains a separate gate. Neither an eight-brief heldout pilot nor fixture success establishes a statistically significant artistic advantage or superhuman performance.

The original `e0-20260908-v1` holdout literals were exposed to the root development context during code inspection before any E0 outcomes. They are superseded, not untouched test evidence; the immutable [exposure note](../artifacts/overnight/evaluation/e0/exposure_note.json) records this boundary. Current protocol `e0_overnight_v2` contains eight newly authored provisional briefs written directly by the isolated subagent to the sealed file. Their texts were not sent to the root agent. The original protocol/assignments remain preserved. This procedural separation does not make the provisional briefs independently human-authored tests.

## Bounded real runner

[e0_runner.py](../src/kusama/e0_runner.py) implements the real renderer/director entrypoint. It selects two frozen eligible whole references, preserves their bytes/context, generates one parent from a disclosed fixed brief/source-metadata prompt, and asks frozen Qwen for three separate condition-specific action requests from that same parent and the same source images. It validates all three distinct executable instructions and freezes them together before any edit. Whole-image FLUX editing receives the actual saved parent and shared seeds 23/41; preservation remains a prompt request, not a guarantee.

The default execution partition is the four development briefs. One brief uses seven primary image attempts, with at most one identical retry for each saved technical failure. Interrupted or uncertain charged work pauses for reconciliation; valid outputs are never replaced for appearance. SQLite reservations, immutable inputs, saved hashes, separate raw responses and per-stage event exports support resuming without duplicate calls. Completed partial image writes can be recovered with disclosed missing timing. Every load on resume receives its own counted cost record.

Exact entrypoints (the coordinator has executed the first command using the explicit profile shown later; subagent implementation tests used fixtures):

```sh
.venv/bin/python -m kusama.e0_runner run --protocol artifacts/overnight/evaluation_v2/e0 --index <frozen-index-directory> --root artifacts/overnight --brief-id e0-d01
.venv/bin/python -m kusama.e0_runner freeze-runtime --protocol artifacts/overnight/evaluation_v2/e0 --index <frozen-index-directory> --root artifacts/overnight
```

Omitting `--brief-id` executes development only. `freeze-runtime` requires all four development output pools complete, measured live parent/edit/reasoning throughput, an actual charged schema-repair timing and measured model-load overhead; fixtures cannot pass this gate. It freezes renderer/director versions/settings, the index and prior-memory hashes, actual action system/roles/schema and scoped execution-code hashes, public protocol and review-plan hashes, and a bounded aggregate retry reserve. The default image-retry reserve is the smaller of 25 and remaining image headroom after 56 heldout primary images. An explicit `--retry-reserve` is frozen too. Heldout dispatch requires `--heldout --runtime-freeze <saved-file>` and checks the entire remaining heldout plan: seven primaries per pending brief, the remaining aggregate retry reserve, reasoning tokens and measured wall time plus a 25% margin before the configured handoff reserve. The aggregate retry quota is transactional and append-only. Once exhausted, failed outputs stay missing; no retry consumes a protected primary allocation by assumption.

The runner returns IDs/counts/status for heldout execution, never heldout brief or result text. Each heldout episode receives only its own brief/actions and the fixed source/memory snapshot. Its prior experiment memory is deliberately empty. The runner verifies the entire predeclared review plan before dispatch. Complete live pair and pool tasks reuse exact planned assignment IDs, reviewer slots and display orders. An immutable per-episode binding sidecar records each protocol-to-live pair mapping and missing-output exclusion. Incomplete pools preserve the planned order of available children but receive a separate excluded task ID. Unassigned reviewer names cannot silently create new E0 assignments; the frozen slots require actual humans, and registration creates no labels. Software passing these tests does not establish an artistic preference gain.

## Runnable matched development modes

[comparison_modes.py](../src/kusama/comparison_modes.py) provides a compact comparison with four renderer attempts per condition, including any parent, identical source access, profile and seed schedule, a four-call reasoning ceiling, and one four-candidate-or-none human pool-selection opportunity per reviewer per condition. Planning freezes the source bytes/context, empty prior memory, actual prompts/schema/code identity, condition order and budgets. It dispatches no models. All four modes use the same actual two source images for Qwen; FLUX receives Qwen's saved text and, for edits, the selected saved parent.

| Mode | Counted execution |
| --- | --- |
| `direct_best_of_n` | One strong source-visible Qwen prompt, four independent FLUX samples. |
| `grounded_no_iteration` | Four source-grounded Qwen proposals frozen before any image, four independent generations. |
| `ordinary_critique_edit` | One parent and three sequential strong ordinary critiques/edits of the latest successful saved image. |
| `kusama_search` | One parent, a source-relationship branch, a protected alternative branch, and one continuation from the actual compatible archive pool. |

Search retains every realized image in the CVT archive and uses pinned CPU CLIP descriptors for its coordinates. No compatible judge exists, so every quality value is null and continuation selection is a declared seeded exploration fallback. Archive coverage and software completion are not quality evidence. Each final four-image pool includes its parent where applicable; the search archive cannot create extra final selection opportunities.

Actual reasoning costs differ: direct sampling uses one call; the other modes use up to four. Equal ceilings and image counts do not imply equal realized time or compute. Every actual inference, failed attempt, model load and search descriptor computation is recorded. There are no technical retries or automatic schema repairs within this compact four-attempt comparison. A failed or missing result consumes or leaves unused its allocation, and an incomplete pool is excluded from standard selection. No human label or winner is generated. This entrypoint accepts development briefs only; it does not provide a way to expose sealed E0 texts.

Supply a JSON brief file with `brief_id`, `brief`, `partition: "development"`, and attributed `authorship`:

```sh
.venv/bin/python -m kusama.comparison_modes plan --comparison-id matched_dev_01 --brief-file <development-brief.json> --index <frozen-index-directory> --root artifacts/overnight
.venv/bin/python -m kusama.comparison_modes run --comparison-id matched_dev_01 --brief-file <development-brief.json> --index <frozen-index-directory> --root artifacts/overnight
```

`--mode <name>` runs one condition from the same immutable plan. Omitting it uses the frozen, seeded condition order. Replay verifies charged artifacts and decision-time parent selections before reusing outputs; uncertain charged calls require reconciliation. Nine offline fixture tests validate these execution contracts, including interrupted reasoning without a second dispatch. The implementation subagent ran no GPU workload and collected no human reviews. A completed, genuinely reviewed multi-brief matched comparison remains the evidence gate.

## Separate optional E0 study profile

[configs/e0_study_384.json](../configs/e0_study_384.json) declares `flux_base4b_e0_384x512_v1`: 384 × 512, 50 steps, guidance 4.0, BF16, model offload and VAE tiling. It is an explicit alternative for the entire E0 protocol. It changes neither the frozen session configuration nor the matched-comparison default of 512 × 640. Across the development outputs, the measured parent maximum is 110.80 seconds and the edit maximum is 188.38 seconds on the assigned M4 Max. These maxima include the same 384×512 profile and are separate from the larger studio profile. Its technical execution passed; no artistic advantage is established.

Pass the same file explicitly to `run`, `freeze-runtime` and any later heldout `run`. The first dispatch freezes the file path, exact file SHA, named metadata and effective renderer configuration in `render_profile.json`; every episode and job specification carries that identity. An omitted, changed or incompatible profile cannot switch later briefs to a different resolution. The file accepts only the named profile and validated width/height, steps, guidance, precision, offload and VAE tiling fields. Model, device, seeds, budgets and output paths cannot be overridden through it. File changes during execution block the next model load or dispatch; failure or slow execution never triggers an automatic profile change.

Start with one actual development parent and its six edits:

```sh
.venv/bin/python -m kusama.e0_runner run --protocol artifacts/overnight/evaluation_v2/e0 --index <frozen-index-directory> --root artifacts/overnight --brief-id e0-d01 --render-profile configs/e0_study_384.json
```

A complete brief writes a `profile_execution_gate.json` with seven verified image hashes/dimensions and the renderer records. Its status distinguishes live measured execution from fixtures or missing timing. This first technical gate does not establish quality or admit the heldout batch. Complete all four development briefs using the identical profile, then freeze and test the conservative measured-throughput gate:

```sh
.venv/bin/python -m kusama.e0_runner freeze-runtime --protocol artifacts/overnight/evaluation_v2/e0 --index <frozen-index-directory> --root artifacts/overnight --render-profile configs/e0_study_384.json
```

Development evidence from a different render profile cannot supply the heldout throughput estimate. The first actual 384 × 512 gate is recorded separately from fixture tests and later heldout admission. The [development contact sheet](../artifacts/overnight/evaluation_v2/e0/episodes/e0-d01/audits/assistant_six_pool_contact_sheet_v1.png) and [assistant observation](../artifacts/overnight/evaluation_v2/e0/episodes/e0-d01/audits/assistant_six_pool_visual_observation_v1.json) retain all six outputs. Inquiry and ordinary critique converge strongly in this brief, especially at seed 41; exploration removes the egg but retains exterior text-like marks. These observations are attributed assistant diagnostics with condition information, not blinded human judgments. Source IDs appearing in the saved instructions do not give FLUX direct access to those reference images; it receives the parent and text only.

## Disclosed development recovery and remaining execution

[e0_recovery.py](../src/kusama/e0_recovery.py) supports one narrow technical recovery: a saved valid JSON object with exactly the four action keys, where the only schema error is a short `preserve` string. A separately charged Qwen repair must return that exact string as one list element and preserve every other string verbatim. It cannot reinterpret, split or improve an artistic action. Original raw JSON must equal its saved parsed output. The parent, source snapshot, valid sibling responses, original local/global settlements and code identity must match before recovery. Uncertain or contradictory charges block it.

The following no-call command was executed successfully for d02:

```sh
.venv/bin/python -m kusama.e0_recovery prepare --protocol artifacts/overnight/evaluation_v2/e0 --index artifacts/overnight/indexes/849a29aaaec9d1959002 --root artifacts/overnight --brief-id e0-d02 --condition ordinary_critique --render-profile configs/e0_study_384.json
```

It saved [authorization.json](../artifacts/overnight/evaluation_v2/e0/episodes/e0-d02/schema_repairs/ordinary_critique/authorization.json), SHA-256 `0cdb13adf0700ea01883e747e387d5706853915ceb4c93790b47fc76e2ead14c`. It authorizes only the disclosed development schema/code migration and one exact-content repair, followed by the six originally planned edits. It does **not** grant a new resource window. The corresponding command is prepared but was **not executed**:

```sh
.venv/bin/python -m kusama.e0_recovery execute --authorization artifacts/overnight/evaluation_v2/e0/episodes/e0-d02/schema_repairs/ordinary_critique/authorization.json
```

This command remains blocked by the expired root. The new runner rejects silent normal resume across changed execution-code hashes. Recovery preserves the old `runtime_identity.json` and discloses the scoped new code in sidecars; it cannot migrate a heldout outcome after inspection. The current live-vote dataset adapter conservatively quarantines such future recovered edits until it explicitly supports their authorization chain.

The estimator `e0_stage_max_with_bounded_schema_repairs_v2` reserves the measured maxima for parents and edits separately, charges retries at the slower stage, allows at most one predeclared schema repair per action, includes repair/load overhead and retains the 25% wall-time margin. It requires an actual eligible development repair measurement before freezing; no synthetic timing substitutes for it. That measurement is still absent. Previously saved forecast scenarios used disclosed hypothetical repair durations and are not admission records. No `runtime_freeze.json` or runtime admission was created for heldout work.

The actual [d03 diagnostic](../artifacts/overnight/evaluation_v2/e0/episodes/e0-d03/audits/assistant_six_pool_visual_observation_v1.json) found ambiguous target identity and inconsistent scale changes. The [d04 diagnostic](../artifacts/overnight/evaluation_v2/e0/episodes/e0-d04/audits/assistant_six_pool_visual_observation_v1.json) found visible surface/overlay changes, but no unique parent rupture corresponding to the proposed repair. All seven actual images in each pool were inspected without assigning preference labels. Source symbolism in model instructions remains an attributed hypothesis.

The fresh development comparison plan `repair_compare_v2` freezes the current code and opaque review IDs; `repair_compare_v1` remains historical and is not silently updated. No matched-mode inference was executed. Running the four modes would require 16 renderer attempts and at most 16 charged reasoning calls (13 planned); source access and review opportunities remain matched. The separate three-brief inspiration queue also remains unexecuted.
