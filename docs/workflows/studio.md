# Persistent studio loop

`kusama-studio` provides three foundations: versioned project/inquiry memory, persistent action/branch history, and bounded execution of explicit proposal batches. The dashboard displays projects, episode budgets, branches, decisions, and raw state. The dashboard also supports explicit human selection; execution controls remain in the CLI.

The studio accepts explicit proposal batches with operator or model provenance. The separate [`kusama-direct` workflow](frozen_director.md) now generates one artistic proposal with frozen Qwen and executes it with FLUX; it does not automatically add that run to a studio episode. The newer [reference-informed creative runner](reference_loop.md) connects briefs and reference packets to multiple Qwen directions, FLUX generations, observer proposals and the [human review interface](../evaluation/human_review.md). Studio selection remains explicit, and neither workflow automatically supplies human judgments. Execution checks remain separate from authored observations and decisions. `needs_review` is the executor’s existing status for an output awaiting inspection; it does not require a reviewer UI.

## Start the included development example

For a full artistic run, first establish a [sufficiently complex artistic problem statement](../research/artistic_problem_statements.md). Use the existing inquiry fields to carry its question, stakes, provisional position, viewing conditions, constraints and unresolved questions. These fields preserve the supplied inquiry; schema validation does not assess its artistic adequacy. Record later changes through inquiry revision with authorship and reasons. The scale example below remains a bounded development fixture, not the primary case study or evidence of a completed full artistic inquiry.

Run from the repository root. The example uses the existing mountain parent and masks; it intentionally labels its inquiry and directions as development examples.

For the first real creative experiment, run one [Qwen-directed edit](frozen_director.md) and inspect it before expanding the batch. The three directions below are operator-authored development fixtures, not Qwen proposals or human preference evidence.

```sh
uv run --locked kusama-studio create --project scale-study --name 'Scale study (development example)' --inquiry examples/studio/inquiry.json --parent artifacts/20260907T190926.622194Z_de125bc5/parent.png --author 'Kusama development example'
uv run --locked kusama-studio start --project scale-study --episode first --max-calls 6 --max-actions 6 --max-rounds 2 --device mps --offload model --vae-tiling --local-files-only
uv run --locked kusama-studio propose --project scale-study --episode first --file examples/studio/proposals.json
uv run --locked kusama-studio step --project scale-study --episode first
uv run --locked kusama-studio run --project scale-study --episode first
uv run --locked kusama-studio show --project scale-study
```

`step` executes or recovers one pending action. `run` drains only the declared batch, then returns for inspection and explicit selection. Rendering consumes the episode’s shared call budget. These commands use the existing FLUX backend; starting or proposing makes no image calls.

For another artifact root, put `--artifacts /absolute/path` **before** the subcommand. Backend flags are accepted at `start` and frozen for the episode. Subsequent execution uses those settings, including on recovery.

## Inspect, choose, and continue

After the batch completes, inspect image links in the basic dashboard or paths in `show`. Artifact IDs are `action_id:attempt_index` (for example `medium:0`). Every produced result remains linked to its parent, including failed checks and fallback results.

```sh
uv run --locked kusama-studio observe --project scale-study --artifact medium:0 --note 'Describe what you actually see here' --author 'Your name' --origin operator
uv run --locked kusama-studio select --project scale-study --episode first --artifact medium:0 --reason 'State why this is the next parent' --author 'Your name'
uv run --locked kusama-studio decide --project scale-study --artifact small:0 --decision retain --reason 'Keep this alternative available' --author 'Your name'
```

The quoted notes are placeholders to replace after inspection. Selection does not imply passed execution checks or artistic success. Retain/reject decisions append history and never delete images. You can explicitly select an older artifact, including `root`, after completing the pending batch.

For a second round, author a new proposal JSON with a new batch ID and action IDs. Set `parent_artifact_id` to the selected artifact. Each referenced controlled-edit plan must use that artifact’s actual path and SHA-256; rebuild masks for its image as needed. Submit with `propose`, then `step` or `run`. All directions in a batch share one frozen parent. Full declared fallback budgets must fit the remaining call budget.

```sh
uv run --locked kusama-studio finish --project scale-study --episode first --reason 'Record what was learned and what remains unresolved' --author 'Your name'
# Or cancel pending work:
uv run --locked kusama-studio stop --project scale-study --episode first --reason 'Explain why this episode stops' --author 'Your name'
```

A running operation holds the project lock. Interrupt its CLI with Ctrl-C before stopping; stop cancels pending actions between executions. Interrupted actions resume with `step`/`run`. A charged call without a saved raw image is marked uncertain rather than silently rerendered. Postprocessing failures return control and can be retried without another inference. Closed episodes cannot resume; start a new episode with an explicit new budget. Exhaustion still allows inspection and selection, then finish/stop.

## Inquiry and references

`inquire --project … --inquiry updated.json --reason … --author …` appends an inquiry version. Required inquiry fields: `question`, `stakes`, `position`, `viewing_conditions`; optional string lists: `constraints`, `unresolved`.

`reference --project … --file reference.json` adds immutable source metadata. Required fields: `id`, `title`, `uri`, `provenance`, `permitted_uses`, `source_family`. Optional: `maker`, `date`, `medium`, `context`. Use a new ID to revise source information. This records provenance and permitted-use assertions; it does not fetch or verify the external source.

Each proposal direction requires `id`, `title`, `intended_change`, nonempty `preservation_constraints`, and a `plan` path relative to the proposal file. Optional `source_ids` must already exist in the project and require a `relationship` explaining the proposed transformation; `relationship_kind` is `proposed_analogy` by default, or `documented_relationship`. A batch requires `id`, `parent_artifact_id`, `author`, `origin` (`operator`, `model_proposal`, or `development_example`), and 1–8 directions with distinct intentions.

## Persistence and boundaries

The shared `artifacts/jobs.sqlite3` is authoritative. Studio project state and append-only events commit together. `artifacts/studio_projects/<id>/project.json` is an inspectable projection. Action directories freeze the parent, masks, executable plan, inquiry version, reference packet, author, and origin before execution. Later inquiry revisions and decisions do not rewrite earlier action context. Repeating an identical batch ID returns the original frozen proposal; use a new ID for changes.

The controlled executor owns rendering, immutable output hashes, reservation accounting, and recovery. Each action is a child of an episode ledger job, so all fallback calls charge the same episode limit. Limits cover calls, actions, and rounds; there is no unlimited autonomous loop. Keep the whole artifact root when backing up, including SQLite WAL files if copying while active; paths currently refer to this local workspace.

Tests use a deterministic backend to exercise branching, budgets, snapshots, malformed proposals, stop behavior, output tampering, and recovery. They do not establish artistic quality or claim a new real-model run.
