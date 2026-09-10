# Single-action contract — 9 September 2026

## Observed failure

`studio-ee200a1e1906411ea1a8` started at 21:50:13 UTC and failed at 21:52:08 UTC. Read-only inspection of its saved responses and authoritative ledgers found three charged local Qwen calls, 12,221 measured input/output tokens, two completed reservations and one failed reservation. No reservation was unsettled. There were no images or committed hypotheses.

1. `response-000.json` supplied question analysis and five prepared queries.
2. `response-001.json` requested `Black women agency`, which was absent from that prepared set. The tool rejected it and supplied feedback to search a prepared query or rewrite the set.
3. `response-002.json` emitted five identical JSON objects containing only `action: "rewrite_queries"` and `reason`. The second object began at character 580, producing `Extra data: line 5 column 1 (char 580)`. Every object omitted the required `queries` array. Taking the first object would not recover an executable action.

The third call received 3,949 input tokens and generated 571 tokens against a 1,800-token output cap. Its saved prompt contained the full schema, including the conditional requirement for `queries`. There is no evidence of a lost schema or token truncation in this incident. The exact three raw completions are retained in the portable [regression fixture](../../tests/fixtures/research/repeated_rewrite_queries.json), with UTF-8 hashes and incident attribution.

## Correction for new plans

`independent-research-v16-single-action-contract` retains the existing action set, artistic problem, study/development gates, frozen models and budgets. Its prompt requires one complete JSON object, actual action arguments before the reason, and termination after the closing brace. It explicitly distinguishes preparing queries from searching them. Each decision receives a compact required-field display derived from its frozen schema and allowed actions. All context projections preserve this display and the complete schema.

A scripted replay also exposed an omitted search mode taking the default `text` path despite the panel having no local index. New grounded plans without an index require `mode: "remote"` in their schema and reject other modes before calling the source backend. Feedback names the search mode and the `queries`, `query` and `field` keys. No model query, reference choice or artistic instruction is supplied by the correction.

The JSON parser remains strict. Multiple objects are not split, deduplicated or accepted as a prefix. Missing arguments can receive the existing bounded tool feedback only after valid JSON has passed the response boundary. There is no added inference call, retry, response repair, budget increase, weight update or migration of the failed run.

## Verification and preparation

`KUSAMA_PROCESSOR_ACCEPTANCE=1 .venv/bin/python -m pytest -q` completed with **1,046 passed, five skipped**. The skips require a live unmocked run or separately selected historical incident archives. The new portable incident replay ran: it decoded the exact repeated completion with the cached Qwen processor, retained the three fixture charges, rejected the response and resumed without another generation or source request. A separate scripted complete rewrite then dispatched remote search. Tests also cover required fields during initial inquiry and pending-study commitment, all prompt projections, and rejection of local mode before source dispatch. An integration failure during development caught a hardcoded research-version list in diagnostic rendering; its judge allocation now reads the frozen grounded-search policy, preserving the existing diagnostic check for v16. The final full suite includes that integration test.

The local preparation is `artifacts/preparations/studio-ee200a1e1906411ea1a8-v16/`. It contains a validated `start-run.json` with a fresh submission ID and otherwise identical operator inputs, a separately frozen/preflighted research plan, full/admitted prompt previews and a provenance receipt. No start request was submitted and no resource session was admitted. The original default ceilings remain 42 Qwen calls, six FLUX attempts and 120 minutes, with the named 512 × 640 profile. The standalone research preparation preserves the original project/brief IDs; the receipt records its relationship to the failed run.

Processor-only checks admitted the new initial prompt at **3,754 tokens** and the diagnostic reconstruction of the failure context at **4,275 tokens**, both below the unchanged 6,392-token input allowance with 1,800 output tokens reserved. All **16** artifacts registered in the original research ledger matched their saved hashes; the original request/plan identities and failed status also remained intact. The preparation made **zero new model calls or network requests**. Generated preparation files remain local and are not present in a fresh checkout.

Tests use scripted completions and cached processors; they cannot establish that Qwen will follow the revised prompt or develop better work.

The next real evidence gate remains an explicitly admitted run reaching grounded provisional studies, actual-image inspection and a consequential choice or revision, followed by human assessment. Merely avoiding this parser failure would not establish artistic progress.
