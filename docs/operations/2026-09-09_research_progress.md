# Online research and run readiness — 9 September 2026

The user requested repairs before another run and online-only source selection. This follows the [24-action stall analysis](../experiments/20260909_research_stall.md). No old research plan was resumed or rewritten.

## Changes

- New-run setup always enables the three existing official online collection providers. Removed the reference-library dropdown, network checkbox, local index discovery, index-selection request field and new-run index freezing/dispatch code. No dedicated selector data files existed. Shared retrieval, original reference bytes and saved run evidence remain because historical inspection and separate continuation/CLI workflows use them.
- New research plans use `independent-research-v7-research-progress`. Frozen operator controls retain the first eight and latest four distinct metadata leads, with source IDs, maker/title excerpts and original step numbers. The order is a retention policy, not an artistic ranking. Result/exclusion counts accompany recent search actions; the latest tool error replaces stale success context.
- Repeated remote provider/query/limit actions are blocked before source dispatch. Their already-used Qwen call remains charged. Six actions without new source evidence, a new passage, a new draft or a commitment end as `stagnated`. Query-dependent IDs for the same provider/object URL do not count as new source evidence. Notes and recalls do not reset the counter.
- For research plans of at least four steps, the last three actions are restricted according to the remaining acquisition, inspection, draft and commitment requirements. An explicit unresolved stop remains possible. This does not manufacture a hypothesis or force an artistic choice.
- Panel runs freeze the selected renderer's token contract. Draft and commit check the exact render instruction with the pinned local FLUX tokenizer; an oversized instruction returns a tool error while research can still redraft within its existing budget. This shares token-counting logic with the actual renderer, uses no model weights and makes no network request. The original response and failed instruction remain preserved.
- Empty terminal research outcomes are `partial` with the specific stop reason, rather than `completed`. Panel reads and subsequent manifest events derive call counts from the resource ledger.
- The research prompt encourages short collection queries, inspection of promising leads and compact, uncertain visual experiments. It explicitly distinguishes fallible model notes from evidence. Token projections shorten optional candidate descriptions/older notes and group omission annotations, preserving original contexts, identifiers, maker/title fields and pending hypotheses.

## Verification

The full software suite passed **877 tests**. After the final options-signature cleanup and an additional tokenizer-equivalence test, **27 targeted tests** passed. JavaScript syntax and generated TypeScript/Pydantic contract checks passed. Tests use substitutes; they do not prove future model decisions or artistic quality.

New regressions cover missed-lead retention across an empty search and false model note; duplicate-query blocking and recovery; stagnation and terminal replay; same-object deduplication; protected synthesis steps; overlong instructions followed by model-authored redrafting; online defaults and removal of local inputs; accurate partial status/call counts; and complete search → acquisition → visual inspection → draft → commitment → two renders for each of Met, Art Institute of Chicago and Cleveland, using fixture transport and models.

Processor-only checks projected all 24 saved decision contexts under the new policy: each admitted within 6,392 input tokens (maximum 6,250). A synthetic synthesis case with six image slots, four source contexts and a pending four-reference draft admitted at 6,185 tokens. The first deliberately combined search-plus-draft stress payload exceeded the limit; its failure is retained separately from the subsequent check using the actual draft tool-result shape. This does not promise admission for arbitrarily long questions or hypotheses. Evidence is in the local [token admission record](../../artifacts/research_audit/20260909_progress/token_admission.json); that artifact link requires this machine's saved artifacts.

The previous run still passes authoritative read-only replay with 24 calls and zero hypotheses. The dashboard was restarted after verifying no active panel jobs. Browser inspection confirmed the online-only form, named render profile and displayed source limits, without pressing Start run.

## Remaining gate

No Qwen or FLUX inference, source acquisition, human review or training was performed for this repair. Useful model-selected acquisition, grounded hypothesis construction and an actual rendering still require the user's next admitted run. Its outputs then require artistic assessment; software progression is not evidence of artistic advantage.
