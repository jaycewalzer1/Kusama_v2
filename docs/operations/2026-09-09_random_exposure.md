# Random exposure before artistic selection — 9 September 2026

The user requested randomness in the material encountered during inspiration research,
without making novelty an output reward. New `independent-research-v18-random-exposure`
plans add a bounded exposure policy alongside the existing inquiry, grounding and
paged-context controls. Existing frozen plans and their evidence remain unchanged.

## Implemented behavior

- New panel runs default to **up to two chance leads per research session**. Run
  settings allow one or zero. Independent research requests opt in with
  `random_exposure_count: 1` or `2`; their default remains zero.
- Planning freezes a fresh seed, the count, breadth-query list and schedule. An
  optional operator `random_exposure_seed` permits reproducible experiments. Qwen
  cannot set the seed, request another draw or change the sampling policy.
- After the initial question analysis, the controller samples eligible whole works
  without replacement, independently of the question, relevance scores and quality
  labels. It deduplicates normalized work IDs, respects frozen source/family
  exclusions and uses only enabled providers. Detail crops do not get extra slots.
- Existing index/corpus/catalog candidates form the pool when available. With an
  empty pool and authorized network access, one seeded choice of provider and one
  operator breadth query obtains at most eight candidates. The controller samples
  from that eligible set. There is no replacement query/provider after failure or
  an empty result, and no top-up of a small existing pool.
- New panel runs still use online official collections. They do not silently import
  local references. The breadth list reuses the standing catalog's operator-authored
  terms, such as textile, vessel, preparatory drawing and scientific instrument.
- Chance leads have explicit `operator_authored_random_sampling` attribution and
  a separate small `chance_candidates` context shelf. Relevance ranking does not
  evict them. The strongest context projections may shorten descriptions while
  preserving both leads and their exposure attribution. Full evidence remains saved.
- Qwen can ignore or reject any lead. Acquisition retains the existing relevance
  gate and model-authored `relevance_override` for an indirect connection. Exposure
  itself neither downloads an image nor authorizes a hypothesis. Actual-image
  inspection, supported properties and all later selection contracts still apply.
- Explicit rejection removes a lead from the shelf. Opening/acquisition moves it
  into the existing source-evidence flow. There is no requirement to incorporate a
  random source and no score for doing so.

## Budgets and recovery

No Qwen or FLUX calls are added to the frozen caps. A bootstrap consumes the existing
network request/byte/storage limits: at most one metadata search plus eight Met
object requests (nine requests total), or one ARTIC/Cleveland search request. The
panel's existing total ceilings remain 64 requests and 16 acquisitions across initial
and optional reopened research. Exploring a lead uses existing action, acquisition
and time budgets; the effect on useful research time has not been measured.

The source store retains `chance_exposure_started.json`, `chance_exposure.json` and
a hashed snapshot containing the seed, exact eligible pool, selected work IDs,
bootstrap query/provider and any error. Research steps bind the snapshot hash and
the actual returned leads. Completed draws replay exactly without a new request;
an incomplete draw fails closed for explicit reconciliation. Failed requests retain
their existing accounting. New plans never rewrite or retry an old run.

## Evidence and limits

The focused verification command passed **117 tests with zero skips**, including
the pinned local tokenizer with chance leads plus 24 dense search results:

```sh
KUSAMA_PROCESSOR_ACCEPTANCE=1 .venv/bin/python -m pytest -q \
  tests/test_research_exposure.py tests/test_panel_runs.py \
  tests/test_research_context.py tests/test_research_sources.py \
  tests/test_research_grounded.py tests/test_research_response_contract.py \
  tests/test_development_research.py tests/test_study_cycle.py tests/test_contract_types.py
```

Dashboard JavaScript syntax, generated contract consistency and new documentation
links also passed. These commands load cached processors where selected, not model
weights for inference.

The broader `.venv/bin/python -m pytest -q` regression run passed **1,031 tests**
with **36 opt-in checks skipped** (cached processors, retained incident inputs and
live-process acceptance). The three processor checks in the focused command above
were separately enabled and passed. This is not an all-gates or live-run acceptance
claim; the two existing dependency deprecation warnings remain.

Offline tests cover seed stability and variation, whole-work deduplication, exclusions,
metadata-only exposure, disabled network access, retained failures, interrupted draws,
snapshot tampering, model acquisition/rejection, context projections and panel-to-saved-
prompt delivery with replay. HTTP and model inference use fixtures; these are software
checks, not evidence of artistic advantage. This change performs no live acquisition,
Qwen/FLUX inference, human review or training.

The exposure distribution remains shaped by museum access, catalog coverage,
operator breadth queries, API ranking and the eight-result ceiling. It is not a
uniform sample of art, a measure of semantic distance or a guarantee of novelty.
Metadata-first selection can still discard a source before seeing its image. Chance
juxtapositions may distract from an inquiry or encourage invented connections; source
grounding and explicit artistic reasons remain necessary, and software cannot
establish their artistic adequacy.

The next evidence gate is an admitted full artistic run showing which chance leads
were seen, acquired, discarded or transformed, followed by human assessment of the
resulting studies and selected-parent revision. Whether selection suppresses useful
surprises, or favors superficial combinations, remains unresolved. Any later comparison
must disclose the additional search/action expenditure and match overall access and
budgets; chance exposure is not an artistic reward or a learned taste model.
