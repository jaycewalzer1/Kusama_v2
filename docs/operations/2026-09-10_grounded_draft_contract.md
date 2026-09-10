# Grounded draft contract — 10 September 2026

The user reported repeated property-attribution errors in the latest run,
`studio-901a68c896cc482b8262`. A read-only audit verified its registered research
artifacts, source bytes and invocation records. It ended `partial` / `model_stop`
at **17:27:33 UTC**, with **seven completed Qwen calls / 43,062 actual tokens**, one
acquired reference, three rejected drafts, no committed hypotheses and no studies.
All seven charges are settled. Two source requests completed, charging 453,343 bytes.
The original [manifest](../../artifacts/runs/studio-901a68c896cc482b8262/manifest.json)
and [research report](../../artifacts/runs/studio-901a68c896cc482b8262/research/report.json)
remain unchanged.

## Cause

The attribution rejections were correct. Qwen acquired and inspected Cleveland's
*Treasure Box (Wakahuia)* from the optional chance shelf. Its three drafts attributed
agency and sacred materiality to a cited description of carved decoration, mixing
its proposed interpretation and visual observations with documented source claims.
Its discarded-property fields also contained its own reasons for discarding, and
two cited sentences about hanging the box or Cook's visit did not support the
described function. Supplying a real claim ID cannot make such wording extractive.

The same drafts omitted `single_source_justification`, `formal_properties` and
`visible_predicates`. The runtime stopped at the grounding errors before reporting
those missing requirements; the generation schema still treated them as optional.
Repeated generic feedback showed neither the mismatched property/citation pair nor
the complete set of missing requirements.

The model's final assertion of an exhaustive search is unsupported: it performed
no prepared-query searches after the chance bootstrap. This is a retained model
statement, not an assessment of collection coverage or artistic adequacy. The source
choice and proposed cross-cultural analogy remain Qwen's unverified judgments.

## New-plan correction

`independent-research-v28-grounded-draft-contract` retains the extractive attribution
validator and freezes the following operator-authored controls:

- Generation requires both property evidence objects, each containing a nonempty
  `claim_id` or `quote`. Evidence precedes its property; the source role follows.
- Generation requires at least two formal properties and one structured numeric
  visible predicate. A one-source hypothesis requires a nonempty justification;
  its justification is generated first so the decoder resolves the one-/multi-source
  alternatives before long text fields. Multi-source hypotheses retain their existing behavior. These are existing tool
  prerequisites, now compiled into the generation contract.
- A rejected draft reports missing prerequisites together with grounding errors.
  The next decision receives `draft_feedback`, pairing each failed model property
  with the actual cited text, source ID and citation. It survives all context
  projections; long cited text is explicitly marked as an excerpt.
- Instructions distinguish an extractive source feature being discarded from the
  model's reason for discarding it. Interpretations and proposed changes belong in
  `transformation`. No replacement property, reference choice or artistic effect is
  supplied by the software.

The token constraint enforces response structure, not semantic entailment. Final
source-attribution, polarity, rendering and inspection checks remain authoritative.
Valid paraphrases outside the conservative extractive contract can still be rejected;
the feedback explains how to represent the supported source content separately.

The exact three completions and a portable source-text projection are preserved in
the [regression fixture](../../tests/fixtures/research/unsupported_property_attribution.json).
A separate scripted test uses extractive properties to exercise draft and commit;
that synthetic example is never written into the live run or attributed to Qwen.
No inference, network acquisition, human review or training was performed by this
repair. Old inputs, failed responses, images, hashes and charges remain preserved.

Software and cached-tokenizer checks validate the new contract and rejection
feedback. Real model use of that feedback, useful source selection, realized
alternatives and human assessment remain the next evidence gates. A new admitted
run is required; the stopped run is not rewritten or replayed.

A [processor-only prospective reconstruction](../../artifacts/diagnostics/20260910-grounded-draft-contract/verified-audit.json)
of the decision after the first failed draft admits the full v28 feedback at
**7,040 input tokens**, within the unchanged **14,584** input allowance. No context
projection is needed; the source image and all feedback are retained. The separately
saved diagnostic plan creates no inference session and has not been executed.

An initial full-suite run was interrupted after 760 passing tests because an
ambiguous one-/multi-source grammar branch made the pinned tokenizer check
excessively slow. Resolving that branch at the first hypothesis key corrected the
overhead; all 13 focused tests then passed, including the pinned-tokenizer cases,
in 46.93 seconds. This measures test execution, not live Qwen throughput.

Final verification: **1,105 passed, five skipped** in 146.63 seconds with
`KUSAMA_PROCESSOR_ACCEPTANCE=1 .venv/bin/python -m pytest -q`. The skips require an
unmocked live run/dashboard or separately selected historical incident archives.
The new pinned-tokenizer regressions ran. Local documentation links and
`git diff --check` also passed. These are software checks, not artistic outcomes.

Local artifact links require the separately preserved artifact directory and do
not resolve in a fresh checkout.
