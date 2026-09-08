# E0 protocol and current execution boundary

Assessment: 8 September 2026. This document describes the new protocol implementation, not an E0 result. [Project direction](project_direction.md) remains authoritative about the research claims.

The [configuration](../configs/e0.yaml) declares four user-supplied development examples and eight assistant-authored provisional holdouts. The preparation command requires an external holdout file or a pre-existing sealed manifest; no holdout text is embedded in source code. It freezes separate sealed brief data, a public manifest containing only holdout IDs/hashes, and 252 review assignments. Holdouts are constructed by this assistant; independent authorship and strong generalization remain unestablished. Development retrieval must never index the sealed directory. The episode accessor exposes a sealed brief only for its own `heldout_execution` episode ID.

The planned E0 contains 12 parents and 72 edits: three alternative actions per parent, each realized at seeds 23 and 41. The conditions are inquiry/relationship, ordinary critique-and-edit, and exploration. Three reviewer slots receive 216 parent/edit assignments and 36 six-candidate-or-none pool assignments. Each pair assignment supports separate quality, development-potential and readiness questions. Multiple questions and image seeds remain correlated observations within a brief; assignment counts do not count independent trials.

The local [prepared manifest](../artifacts/overnight/evaluation_v2/e0/manifest.json), [assignments](../artifacts/overnight/evaluation_v2/e0/review_assignments.jsonl), and [status snapshot](../artifacts/overnight/evaluation_v2/e0_status.json) have zero E0 parent images, zero E0 edit outputs, zero E0 judgments, and zero E0 pool selections. All 84 images remain unexecuted. Existing exploratory paintings are not relabeled as E0. The manifest status is `planned_not_executable`; this module does not reserve resources or dispatch a renderer. Local artifact links require the separately preserved artifact root.

A runner must first freeze references and prior memory, generate/hash the parent, validate all three action requests against the actual operation compiler, and call `freeze_episode` with the matched render profile and shared seeds. Heldout episodes additionally require a recorded development configuration freeze. The parent bytes are verified; changing a frozen record is rejected. The execution adapter must reserve measured wall/GPU time, reasoning tokens and image attempts through the existing execution/resource ledger before dispatch. The evaluation module records already-authorized attempts; it is not a bypass around those contracts.

The missing-output rule permits one identical-request retry after technical failure, with uncertain execution reconciled first. Both executions count. A valid image cannot be replaced because it appears weak. An unresolved pair is excluded from completed-pair analysis and its six-candidate pool remains incomplete. Inspecting that pool separately cannot count as a standard complete-pool selection. Frozen requests and attempts are immutable. The separate E0 runner now registers the exact available parent/edit pairs and a six-child-or-none pool in the real review ledger. A missing output leaves its pair excluded and its pool ineligible for standard selection. Real-label statistical analysis is still pending; the protocol planner’s `protocol_status` identifies its own store scope, while actual human responses live in `ReviewStore`.

Tested commands from the repository root:

```sh
.venv/bin/python -m kusama.evaluation prepare --config configs/e0.yaml --output artifacts/overnight/evaluation_v2/e0
.venv/bin/python -m kusama.evaluation status --protocol artifacts/overnight/evaluation_v2/e0
.venv/bin/python -m pytest -q tests/test_evaluation.py tests/test_datasets.py tests/test_learning.py
```

Preparation is idempotent when bytes agree. The source is [evaluation.py](../src/kusama/evaluation.py). Its offline tests verify exact counts, deterministic image-order assignment, sealed access, frozen parent integrity, failure accounting and retry limits. They do not exercise image generation or human evaluation.

The next evidence gate is to complete a reviewed development continuation, confirm measured resource feasibility, freeze technical/reference/memory inputs, and execute the development E0 portion. A full comparative E0 remains pending. Neither an eight-brief heldout pilot nor fixture success establishes a statistically significant artistic advantage or superhuman performance.

The original `e0-20260908-v1` holdout literals were exposed to the root development context during code inspection before any E0 outcomes. They are superseded, not untouched test evidence; the immutable [exposure note](../artifacts/overnight/evaluation/e0/exposure_note.json) records this boundary. Current protocol `e0_overnight_v2` contains eight newly authored provisional briefs written directly by the isolated subagent to the sealed file. Their texts were not sent to the root agent. The original protocol/assignments remain preserved. This procedural separation does not make the provisional briefs independently human-authored tests.

## Bounded real runner

[e0_runner.py](../src/kusama/e0_runner.py) implements the real renderer/director entrypoint. It selects two frozen eligible whole references, preserves their bytes/context, generates one parent from a disclosed fixed brief/source-metadata prompt, and asks frozen Qwen for three separate condition-specific action requests from that same parent and the same source images. It validates all three distinct executable instructions and freezes them together before any edit. Whole-image FLUX editing receives the actual saved parent and shared seeds 23/41; preservation remains a prompt request, not a guarantee.

The default execution partition is the four development briefs. One brief uses seven primary image attempts, with at most one identical retry for each saved technical failure. Interrupted or uncertain charged work pauses for reconciliation; valid outputs are never replaced for appearance. SQLite reservations, immutable inputs, saved hashes, separate raw responses and per-stage event exports support resuming without duplicate calls. Completed partial image writes can be recovered with disclosed missing timing. Every load on resume receives its own counted cost record.

Exact commands (the real GPU pathway was not executed by the implementation subagent; bounded offline fixture tests verify these wrappers):

```sh
.venv/bin/python -m kusama.e0_runner run --protocol artifacts/overnight/evaluation_v2/e0 --index <frozen-index-directory> --root artifacts/overnight --brief-id e0-d01
.venv/bin/python -m kusama.e0_runner freeze-runtime --protocol artifacts/overnight/evaluation_v2/e0 --index <frozen-index-directory> --root artifacts/overnight
```

Omitting `--brief-id` executes development only. `freeze-runtime` requires all four development output pools complete, measured live render/reasoning throughput and measured model-load overhead; fixtures cannot pass this gate. It freezes renderer/director versions/settings, the index and prior-memory hashes, actual action system/roles/schema and scoped execution-code hashes, public protocol and review-plan hashes, and a bounded aggregate retry reserve. The default reserve is the smaller of 25 and remaining image headroom after 56 heldout primary images. An explicit `--retry-reserve` is frozen too. Heldout dispatch requires `--heldout --runtime-freeze <saved-file>` and checks the entire remaining heldout plan: seven primaries per pending brief, the remaining aggregate retry reserve, reasoning tokens and measured wall time plus a 25% margin before the configured handoff reserve. The aggregate retry quota is transactional and append-only. Once exhausted, failed outputs stay missing; no retry consumes a protected primary allocation by assumption.

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

[configs/e0_study_384.json](../configs/e0_study_384.json) declares `flux_base4b_e0_384x512_v1`: 384 × 512, 50 steps, guidance 4.0, BF16, model offload and VAE tiling. It is an explicit alternative for the entire E0 protocol. It changes neither the frozen session configuration nor the matched-comparison default of 512 × 640. It has no demonstrated throughput or artistic advantage merely because the smaller dimensions are configured.

Pass the same file explicitly to `run`, `freeze-runtime` and any later heldout `run`. The first dispatch freezes the file path, exact file SHA, named metadata and effective renderer configuration in `render_profile.json`; every episode and job specification carries that identity. An omitted, changed or incompatible profile cannot switch later briefs to a different resolution. The file accepts only the named profile and validated width/height, steps, guidance, precision, offload and VAE tiling fields. Model, device, seeds, budgets and output paths cannot be overridden through it. File changes during execution block the next model load or dispatch; failure or slow execution never triggers an automatic profile change.

Start with one actual development parent and its six edits:

```sh
.venv/bin/python -m kusama.e0_runner run --protocol artifacts/overnight/evaluation_v2/e0 --index <frozen-index-directory> --root artifacts/overnight --brief-id e0-d01 --render-profile configs/e0_study_384.json
```

A complete brief writes a `profile_execution_gate.json` with seven verified image hashes/dimensions and the renderer records. Its status distinguishes live measured execution from fixtures or missing timing. This first technical gate does not establish quality or admit the heldout batch. Complete all four development briefs using the identical profile, then freeze and test the conservative measured-throughput gate:

```sh
.venv/bin/python -m kusama.e0_runner freeze-runtime --protocol artifacts/overnight/evaluation_v2/e0 --index <frozen-index-directory> --root artifacts/overnight --render-profile configs/e0_study_384.json
```

Development evidence from a different render profile cannot supply the heldout throughput estimate. The implementation subagent executed only fixture tests; the first actual 384 × 512 development gate and any later heldout admission remain separate runtime evidence.
