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


## Duplicate-source follow-up

The later user-reported run `studio-001496cc76ec406ead30` froze v28. Saved steps
003 and 006 each list `artic_11390` twice. Steps 004–005 replace the duplicate
with an unacquired candidate and contain unsupported property citations. These
four rejections are preserved in the [portable regression fixture](../../tests/fixtures/research/duplicate_source_drafts.json),
with hashes of the original raw completions and opened source text. This is a
snapshot of rejected steps, not a terminal execution or artistic assessment.

V28's call to the hypothesis validator raised on duplicate IDs before returning
other collected problems. Its prompt builder caught that same exception and omitted
`draft_feedback`; synthesis also supplied only a generic next action.

New `independent-research-v29-complete-draft-feedback` plans freeze
`evidence_first_complete_draft_v2`. They retain strict validation while collecting
hypothesis-level errors, duplicate canonical IDs, unopened/unacquired sources,
property/citation failures and missing requirements together. Alias IDs count as
the same source. A duplicate list does not avoid the single-source justification
requirement. Malformed fields receive field diagnostics without being passed to
source-grounding helpers. Instructions identify each source as a single contribution;
no sources are merged, quotations authored, completions repaired or retries made.
The generation grammar remains v28; uniqueness and entailment are runtime checks.
Historical policy branches, raw evidence and run settings remain unchanged.

Regression coverage replays all four completions, checks alias duplication,
malformed-field handling, historical v28 behavior and retention of feedback through
all context projection stages. The prior synthetic successful draft/commit check
still applies. No Qwen/FLUX inference, network acquisition, human judgment or training
was performed. Real model correction after feedback, meaningful alternatives and
human assessment remain the next evidence gates.

Follow-up verification: `KUSAMA_PROCESSOR_ACCEPTANCE=1 .venv/bin/python -m pytest -q`
passed **1,108 tests**, with five live/archive-dependent skips, in 192.53 seconds.
A focused run including the subsequently added historical-v28 regression passed
**16 tests**, with one opt-in tokenizer skip. The full suite ran the cached-tokenizer
checks. Documentation target checks and `git diff --check` passed. These results
establish software behavior only, not reliable live drafting or artistic progress.


## Grounded generation follow-up

The user's next reported rejection came from v29 run `studio-7549ca07b8524d3789dc`,
step 003. Qwen listed `artic_7634` twice and attached its interpretation of dissolution
without resolution to source claims. All four source-property fields failed
extractive entailment. The [exact completion and source fixture](../../tests/fixtures/research/grounded_generation_incident.json)
retain the raw-response hash. V29 diagnosed the failure correctly but did not prevent
its generation; clearer feedback alone did not establish compliance.

New `independent-research-v30-grounded-generation` plans freeze
`distinct_sources_extractive_spans_v1`. A stateful character parser intersects the
existing JSON token grammar with the frozen opened evidence:

- Each source ID must be an eligible canonical ID not already used. Sources with
  images require actual saved inspection of those image hashes. Previously inspected
  sources remain eligible even after another image becomes active.
- Claim IDs must belong to the selected source; quoted text must be a literal
  contiguous span of its opened text. Properties must be literal contiguous spans
  of the selected claim, or quote when no claim is selected. Negation words required
  by the existing validator cannot be removed. JSON escaping and Unicode are retained.
- Qwen chooses the source, citation and span; no phrase is ranked, chosen, repaired or
  supplied as a replacement. Its role, transformation, composition and artistic
  interpretation remain free text under existing bounds.
- Available source count bounds the source array. With only one source the grammar
  requires the existing justification branch. Without eligible evidence it excludes
  draft/commit and retains other allowed actions, including stop.
- The complete generation evidence survives context projections and is included in
  token admission. Irreducible overflow retains an uncharged capacity checkpoint.
  This adds evidence-context overhead; no throughput improvement is claimed.

The runtime validators remain authoritative. The new constraints address these
specific duplicate/citation/entailment failures; they do not guarantee sufficient
output tokens, useful interpretations, successful rendering or artistic quality.
Historical plans, failures, charges and raw bytes remain unchanged. No inference,
network acquisition, training or human judgment was performed by this repair.

The [processor-only prospective reconstruction](../../artifacts/diagnostics/20260910-grounded-generation/admission.json)
of step 003 admits the full new context at **8,344 input tokens** against **14,584**
available, with no projection. This is separately attributed diagnostic evidence,
not a replacement decision. Local artifact links require the preserved artifact tree.
Regression tests exercise the exact failure, duplicate and unknown IDs, foreign claim
IDs, fabricated quotes, unsupported properties, polarity, escaped quotes/Unicode,
source-count branches and successful synthetic draft/commit execution. The real cached
Qwen tokenizer admits a supported completion and blocks duplicates and unsupported
properties. No frozen model weights are loaded for these tests.

Verification: the processor-enabled full suite passed **1,125 tests**, with five
live/archive-dependent skips, in 163.40 seconds. A subsequent span-boundary
refinement (including escaped-character length) passed **51 focused tests** with
the cached tokenizer enabled; an additional zero-evidence stop-token check passed.
Documentation target checks and `git diff --check` passed. These checks establish
protocol behavior and context admission, not successful live artistic research.
