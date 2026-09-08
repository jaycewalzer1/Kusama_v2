# E0 protocol and current execution boundary

Assessment: 8 September 2026. This document describes the new protocol implementation, not an E0 result. [Project direction](project_direction.md) remains authoritative about the research claims.

The [configuration](../configs/e0.yaml) declares four user-supplied development examples and eight assistant-authored provisional holdouts. The preparation command requires an external holdout file or a pre-existing sealed manifest; no holdout text is embedded in source code. It freezes separate sealed brief data, a public manifest containing only holdout IDs/hashes, and 252 review assignments. Holdouts are constructed by this assistant; independent authorship and strong generalization remain unestablished. Development retrieval must never index the sealed directory. The episode accessor exposes a sealed brief only for its own `heldout_execution` episode ID.

The planned E0 contains 12 parents and 72 edits: three alternative actions per parent, each realized at seeds 23 and 41. The conditions are inquiry/relationship, ordinary critique-and-edit, and exploration. Three reviewer slots receive 216 parent/edit assignments and 36 six-candidate-or-none pool assignments. Each pair assignment supports separate quality, development-potential and readiness questions. Multiple questions and image seeds remain correlated observations within a brief; assignment counts do not count independent trials.

The local [prepared manifest](../artifacts/overnight/evaluation_v2/e0/manifest.json), [assignments](../artifacts/overnight/evaluation_v2/e0/review_assignments.jsonl), and [status snapshot](../artifacts/overnight/evaluation_v2/e0_status.json) have zero E0 parent images, zero E0 edit outputs, zero E0 judgments, and zero E0 pool selections. All 84 images remain unexecuted. Existing exploratory paintings are not relabeled as E0. The manifest status is `planned_not_executable`; this module does not reserve resources or dispatch a renderer. Local artifact links require the separately preserved artifact root.

A runner must first freeze references and prior memory, generate/hash the parent, validate all three action requests against the actual operation compiler, and call `freeze_episode` with the matched render profile and shared seeds. Heldout episodes additionally require a recorded development configuration freeze. The parent bytes are verified; changing a frozen record is rejected. The execution adapter must reserve measured wall/GPU time, reasoning tokens and image attempts through the existing execution/resource ledger before dispatch. The evaluation module records already-authorized attempts; it is not a bypass around those contracts.

The missing-output rule permits one identical-request retry after technical failure, with uncertain execution reconciled first. Both executions count. A valid image cannot be replaced because it appears weak. An unresolved pair is excluded from completed-pair analysis and its six-candidate pool remains incomplete. Inspecting that pool separately cannot count as a standard complete-pool selection. Frozen requests and attempts are immutable. The protocol planner has no automatic E0-to-review-ledger bridge or real-label statistical analysis yet; `protocol_status` explicitly identifies its protocol-store scope instead of treating planned assignments as votes.

Tested commands from the repository root:

```sh
.venv/bin/python -m kusama.evaluation prepare --config configs/e0.yaml --output artifacts/overnight/evaluation_v2/e0
.venv/bin/python -m kusama.evaluation status --protocol artifacts/overnight/evaluation_v2/e0
.venv/bin/python -m pytest -q tests/test_evaluation.py tests/test_datasets.py tests/test_learning.py
```

Preparation is idempotent when bytes agree. The source is [evaluation.py](../src/kusama/evaluation.py). Its offline tests verify exact counts, deterministic image-order assignment, sealed access, frozen parent integrity, failure accounting and retry limits. They do not exercise image generation or human evaluation.

The next evidence gate is to complete a reviewed development continuation, confirm measured resource feasibility, freeze technical/reference/memory inputs, and execute the development E0 portion. A full comparative E0 remains pending. Neither an eight-brief heldout pilot nor fixture success establishes a statistically significant artistic advantage or superhuman performance.

The original `e0-20260908-v1` holdout literals were exposed to the root development context during code inspection before any E0 outcomes. They are superseded, not untouched test evidence; the immutable [exposure note](../artifacts/overnight/evaluation/e0/exposure_note.json) records this boundary. Current protocol `e0_overnight_v2` contains eight newly authored provisional briefs written directly by the isolated subagent to the sealed file. Their texts were not sent to the root agent. The original protocol/assignments remain preserved. This procedural separation does not make the provisional briefs independently human-authored tests.
