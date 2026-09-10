# Research run acceptance audit — 9 September 2026

The existing **878-test suite passed before this repair**, while **nine newly added incident and crash regressions failed**. That is why the earlier green suite was insufficient: it did not exercise the interaction between stagnation and synthesis, or interruption after a terminal step had become durable.

## Findings and repair

New plans use `independent-research-v8-run-acceptance`. These are operator-authored workflow controls, not learned artistic judgment.

1. **Stagnation could terminate before synthesis.** After six actions without progress, a run with retained candidates or opened evidence now enters one three-action synthesis checkpoint. It shares the original action/time/token budget. Acquisition or drafting cannot reset or extend that checkpoint. The allowed actions shrink with remaining capacity, so acquisition → draft → commit fits; an explicit unresolved stop remains available. With no leads, research still ends as `stagnated`. The ordinary final-three-action reserve also remains. This bounded synthesis route does not automatically select references or construct a hypothesis; it does not add a replanning search allowance.
2. **A crash after a terminal step could admit another call.** Completion now derives from verified durable steps for new plans, including model stop, rejected response, hypothesis cap, stagnation and step cap. Resume reconciles the terminal job status without a new model call. Existing uncertain-reservation protections remain. Injected interruptions before tool execution and after step registration verify that saved responses are reused across search, acquisition/opening, checkpoint and draft boundaries.
3. **Metadata and image availability were not explicit in active source context.** New decision payloads include `acquisition_required` and `image_count`, while preserving original source metadata and bytes.

Old frozen plans are not migrated or resumed. Read-only reporting preserves their original version/policy semantics. Eight local panel research archives were inspected before and after the repair; their original plan and response hashes, call counts and outcomes remain unchanged.

## What the acceptance gate exercises

[Incident and interruption tests](../../tests/test_research_run_acceptance.py) run the research controller with actual SQLite/resource ledgers and controlled model/source substitutes. They cover successful and declined checkpoints, ignored restrictions, repeated reopening, metadata acquisition, total budgets from 4 to 48 actions, four terminal crash states, and eight interruption boundaries. The existing source, integrity, rendering, token-admission and panel tests remain in the full suite.

[Processor-backed acceptance tests](../../tests/test_research_processor_acceptance.py) replace only model generation, rendered pixels and HTTP transport at the principal execution boundaries. They use the actual pinned cached Qwen processor and FLUX tokenizer, the production response parser, source backend, panel API, frozen plans, source acquisition, image tensor preparation, prompt admission, rendering handoff, ledgers and inspection endpoints. No model weights are loaded and no request reaches a museum.

The 18 processor-backed cases cover:

- Three providers × direct completion, acquisition followed by stagnation, metadata followed by stagnation, and HTTP failure followed by a different valid query.
- Invalid and output-truncated JSON through actual generated-token decoding, with charged rejection and no retry.
- Two committed directions/four saved fixture images; and an instruction accepted by the action schema but rejected by the actual FLUX token limit, followed by a shorter scripted draft.
- Six image tensors and a six-reference pending hypothesis within the 6,392 input / 1,800 output allowance, retaining the complete pending hypothesis.
- The actual saved 15-decision incident, its original problem/context, and all 11 preserved HTTP response bodies through the real source backend. Timestamp-dependent candidate IDs are remapped to the same query/object in the diagnostic store. Two **assistant-authored fixture responses** then draft and commit, producing two fixture images. The diagnostic uses a one-hypothesis cap to test the handoff; it does not rewrite the original two-hypothesis plan. This establishes that the new route can execute, not that Qwen would choose those responses.

## Repeatable commands

Standard software verification (portable, processor cases explicitly skipped):

```sh
uv run --locked pytest -q
```

Full local acceptance, including cached processors and the selected archived incident:

```sh
KUSAMA_PROCESSOR_ACCEPTANCE=1 \
KUSAMA_RECORDED_RESEARCH=artifacts/runs/studio-87f61ec22af04bf3978b/research \
uv run --locked pytest -q \
  --junitxml=artifacts/research_audit/20260909_acceptance/full.xml
```

A focused rerun uses the same environment variables with `tests/test_research_processor_acceptance.py tests/test_research_run_acceptance.py`. An explicitly enabled processor gate fails if its pinned local caches are unavailable. The archived replay needs that separately preserved local run; it cannot run from a fresh checkout. Without `KUSAMA_RECORDED_RESEARCH`, that one archive-dependent case is explicitly skipped. The ordinary passing suite alone is not equivalent to passing the full local acceptance gate.

**Final result: 922 passed, zero failed, zero skipped, in 71.77 seconds**, including all 18 processor-backed cases. The two warnings are existing FastAPI/Starlette dependency deprecations. Generated TypeScript contracts match Pydantic, dashboard JavaScript syntax checking passed, and changed documentation has no broken local links. The local [JUnit result](../../artifacts/research_audit/20260909_acceptance/full.xml), [acceptance summary and code hashes](../../artifacts/research_audit/20260909_acceptance/summary.json), and [archive preservation checks](../../artifacts/research_audit/20260909_acceptance/preservation.json) retain the evidence. These artifact links require this machine’s local store.

## Superseding live-process requirement

The user required tests representative of the entire process. The 922-test result above is a software/processor result; it cannot certify the unmocked process. The [live process acceptance gate](2026-09-09_live_process_acceptance.md) adds actual model/source/render execution evidence and a release command that fails if any required layer is skipped.

## Remaining evidence gate

Passing these tests establishes the specified software behavior under scripted responses and retained HTTP data. It cannot guarantee useful model decisions, future museum availability, GPU/model-load behavior, or artistic quality. The next live gate is a separately admitted bounded run that actually acquires, drafts, reinspects, commits and renders, with no hidden repair or retry; the next artistic gate is human assessment of that work. This audit performs no Qwen/FLUX inference, live acquisition, human review or training. It does not raise the old budgets or relabel failures as successful artworks.
