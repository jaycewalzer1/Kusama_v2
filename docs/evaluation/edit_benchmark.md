# Development edit-and-observe benchmark

This benchmark makes the editing bottleneck inspectable without creating human judgments or launching inference. It separates input integrity and edit scope, target existence, requested-change fulfillment, exact pixel preservation, and Qwen's own assertions. Its five task types are connection, removal, uniform resizing, detail preservation, and changing a visual relationship.

## Inputs and execution

From the repository root:

```sh
uv run --locked kusama edit-benchmark validate \
  --suite configs/edit_benchmark/development_v1.json
uv run --locked kusama edit-benchmark evaluate \
  --suite configs/edit_benchmark/saved_failures_v1.json \
  --out artifacts/edit_benchmark/saved-diagnostic-new
uv run --locked kusama edit-benchmark evaluate \
  --suite configs/edit_benchmark/development_v1.json \
  --out artifacts/edit_benchmark/prospective-plan-new
```

The module can also be invoked as `python -m kusama.edit_benchmark`. Evaluation is always offline. It never loads Qwen, FLUX or detector weights, invokes a renderer, resumes an old job, changes an original manifest, or writes a review. `--out` must name a new directory; existing reports are never overwritten. `--root` resolves supplied artifact paths after relocation, and hashes must still match.

The [saved diagnostic](../../configs/edit_benchmark/saved_failures_v1.json) contains three mountain/person outputs from the original controlled-edit pilot and both care/control finishing seeds. The [prospective schedule](../../configs/edit_benchmark/development_v1.json) contains five explicit assistant-authored tasks bound to saved real parents. Those parents are familiar development material; a new schedule does not make them fresh or heldout evidence. Prospective entries have no children and remain `unexecuted`.

These configurations retain local artifact paths and SHA-256 identities; they do not include the image bytes in Git. A fresh checkout needs the separately preserved local artifacts. Missing or changed bytes remain visible errors in reports, rather than disappearing from denominators.

## Independent evidence and its limits

Every file input is SHA-256 checked before measurement. The suite admits only the development split. Fixture suites cannot mix with real or prospective cases.

- **Scope preflight:** each declared required region must be fully editable. Protected regions cannot overlap a required edit, and all regions must fit the actual parent. Relationship tasks need at least two declared regions. Region bounds are explicit half-open pixel rectangles, with operator, assistant or model attribution. These are geometric constraints, not semantic localization ground truth.
- **Target existence:** a uniquely strong saved detection from the pinned independent Grounding DINO observer can establish the narrow operational result. Missing, weak, ambiguous or malformed detections remain uncertain. Assistant notes about absent objects remain notes; they do not become independent labels.
- **Fulfillment:** the existing [verification module](../../src/kusama/verification.py) replays saved detector boxes against explicit scale, width, height and anchoring tolerances. The saved report must bind the actual child, and its paired plan must bind the actual parent. Current replay is an additional sidecar, preserving the historical status and raw measurements. Detector nondetection cannot pass semantic removal. General connections and changed relationships remain uncertain until an appropriate independent measurement exists.
- **Preservation:** RGB values are compared at original resolution. All pixels outside an allowed mask, plus explicitly protected detail rectangles, must remain exact. This proves the declared pixel constraint; it does not prove composition quality or semantic preservation.
- **Observer claims:** an assertion identifies its original hash-checked source, exact JSON pointer, author, and parent/child binding pointers. An assistant interpretation of prose is disclosed. A claimed success is `confirmed` false only when independent fulfillment fails; an uncertain independent outcome produces `unsupported`, not a false-success label. Missing assertions are not interpreted as failure.

The `preserve_detail` operation has a deliberately narrow fulfillment definition: exact preservation of its declared detail pixels. A passing detail task does not establish that some other requested intervention succeeded. Other task semantics cannot borrow that pass.

A supplied rectangle may overestimate an object's silhouette. A rejected scope check therefore calls for revisiting the declared geometry; it is not an instruction to silently expand an actual frozen mask. Before new execution, supply or propose a justified, parent-bound region through the current continuation contract.

## Reports and runtime integration

[edit_benchmark.py](../../src/kusama/edit_benchmark.py) provides:

```python
suite = load_suite(suite_path)                  # strict EditSuite / EditCase contracts
scope = preflight(case, root=artifact_root)      # pass / fail / uncertain
result = measure_case(case, root=artifact_root) # independently separated evidence
summary = summarize(results)                   # case counts and brief experimental units
directory = evaluate_suite(suite_path, output_directory, root=artifact_root)
```

A runtime result can be assessed by constructing an `EditCase` with actual parent/child `Asset(path, sha256)` records and the declared mask or protected regions. Without cached independent semantic observations, fulfillment stays uncertain. Thus callers can use fresh pixel preservation immediately while keeping model observations separate.

Every report directory contains a suite snapshot, provenance manifest, individual case records, summary, Markdown inspection report and artifact hash receipt. Measurement failures stay in the report. A failed output measurement does not borrow an older attempt's result. CLI evaluation returns nonzero for measurement/integrity errors; a properly measured failed edit is a successful diagnostic execution. CLI validation returns nonzero for invalid scope.

Counts are descriptive across **briefs**: three mountain methods are one brief, two care/control seeds are another. Neither five images nor five task types implies five independent artistic trials. No preference, readiness or emotional-effect scores are computed.

## Offline diagnostic completed on September 8, 2026

The final [local diagnostic report](../../artifacts/edit_benchmark/20260908_saved_failures_final/report.md) rechecks five saved real children across two briefs with zero inference calls. Both geometric mountain edits pass current width-and-height replay; the masked mountain edit fails. Both care/control semantic connection checks remain uncertain. All five preserve their declared protected pixels exactly. Both care/control plans exclude the declared outer-support endpoints and fail the new scope check. Qwen's seed-41 connection assertion is unsupported by an independent measurement; the historical assistant observation describes an incomplete connection and remains separately attributed.

The report includes a clearly labeled **counterfactual** comparison: a permissive rule accepting any overlap with any required region would admit both care/control plans, whereas full required-region coverage rejects both. This checks the stricter software rule on known historical failures. It does not report the old runtime's behavior as a measured baseline, validate new rendering outcomes, or show artistic improvement.

The separate [prospective report](../../artifacts/edit_benchmark/20260908_prospective_final/report.md) retains all five future tasks as unexecuted. Unit tests use temporary fixtures and do not establish real inference capability. The next technical evidence gate is a newly authorized bounded run on additional development parents with independent measurements appropriate to each task, followed by a controlled workflow comparison. Original human-review and research-learning gates remain separate.
