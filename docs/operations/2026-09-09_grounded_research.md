FAIL — the recorded artistic run does not meet the new acceptance criteria.

# Grounded research repair, 9 September 2026

New plans use `independent-research-v13-grounded-search`. The preserved 16-step run `studio-29fd86d2de4540338c92` was a completed execution with one opened source and a pressured commitment, not a validated artistic success. Its source property attribution conflicts with the saved description, and its visible test asks for a viewer response instead of measurable image conditions. Original plans, actions, images and raw responses remain unchanged. The recovery/repair implementation was not modified.

## Prompt and image evidence

The exact saved prompt for decision 7 is **21,668 characters** and includes `output_schema`, hypothesis fields and validation constraints. The historical projection stripped schema annotation titles. Empty objects in the viewer produced empty-looking entries; these were not evidence that the model received no schema. The [raw string](../../artifacts/research_prompt_audit/step-007-prompt.txt) and [audit](../../artifacts/research_prompt_audit/prompt_audit.json) preserve the check. V13 projections preserve the schema including titles and remove lower-relevance candidates and older actions first.

A separately bounded, user-requested image diagnostic made **one real Qwen call**, with **127 input tokens and 168 generated tokens**, no caption, no network and no FLUX call. The original source image hash was `009dfe22776d2a60987910a0066edfcb1cbc773d559ea08d6e000cb9dcc06368`. Qwen described the long neck, greenish-blue surface, irregular wooden base and smaller figure below. These details are absent from the source caption, and the input record binds the actual image bytes. This demonstrates visual input access for this probe; it does not certify all observations or interpretation. See the [probe report](../../artifacts/research_prompt_audit/vision-probe/probe_report.json) and [raw response](../../artifacts/research_prompt_audit/vision-probe/response-000.json). Runs with only recorded text inputs are labeled caption-only; absent records are not captured.

## New workflow

Step 0 records model-authored subject, medium, era, concepts and possible artists/movements. Optional operator seed artists, seed works and medium constraints retain their attribution. The model prepares 3–5 concrete queries of 1–3 words, selecting artist, subject, technique, object or work fields. The raw question cannot enter a provider through the research action. Each remote query searches every configured provider, merges objects, preserves per-provider failures, and shows matched terms. Empty results require a different next query.

ARTIC uses Elasticsearch fields `artist_title`, `subject_titles`, `term_titles` and `classification_titles`; work queries use `title`. Cleveland uses its documented `artists`, `type`, `medium` and `title` filters, and `q` for keyword/subject queries. Cleveland documents no `keywords` query parameter, so one was not invented. The prompt explains Met's all-term matching and the short-query limit. Provider adapters were checked against [ARTIC documentation](https://api.artic.edu/docs/) and [Cleveland documentation](https://openaccess-api.clevelandart.org/); fixture transports verify parameter construction, not live result relevance.

Candidate relevance is a separately recorded Qwen score against the current question, with an explanation. Scores are model assertions, not human labels or trained artistic judgment. Unknown/below-threshold candidates cannot be acquired without a separate, explicit `relevance_override`; an ordinary action reason does not bypass the gate. Retention ranks these scores and drops rejected candidates. After the configured number of unsuccessful reformulations has actually been searched, the run exits with `no relevant material in configured collections`. Errors consume two stagnation units and idle actions one; the prompt shows both the counter and its reason.

Drafts accept encountered candidate aliases or acquired canonical IDs. IDs such as `hypothesis_01` normalize to `hypothesis_1` without changing the raw model completion. Draft hashes reject unchanged resubmissions. A successful draft attaches its images automatically, then allows commit, revised draft, reject, search or stop. In synthesis, a pending draft permits only committing it unchanged or stopping. Reopening an attached source is rejected. Commit requires 2–3 observations per image supporting the transfer; at least two must pass a disclosed caption-paraphrase screen.

Each transferred/discarded property cites an opened claim or exact quotation. The entailment gate is deliberately **conservative and extractive**: an unsupported paraphrase fails, even if it might be defensible. Creative changes belong in the transformation, not attributed source facts. Negation removal is rejected. This does not implement a general semantic entailment model. The observation screen is lexical and cannot prove that an observation is visually true or catch every semantic paraphrase. These limitations remain visible in the reports.

Hypotheses require two opened sources or an explicit single-source justification. At least two concrete transferred formal properties must also appear verbatim in the render instruction. New real drafts validate **composition plus render instruction** together using the pinned FLUX tokenizer and the frozen maximum of 512 tokens; both are sent to FLUX. Fixtures explicitly label substitute counting. Named render profiles remain distinct.

Visible tests include structured numeric predicates over object counts or canvas fractions. A subsequent charged Qwen VLM call measures each rendered image; software evaluates the comparisons. New panel budgets display and reserve one additional judge call per planned image. A report cannot PASS from a render alone: it requires two relevant opened sources, an unpressured commit, no unverified/contradictory grounding, and passing render predicates. Assessment records bind the research plan, rendered bytes, judge decision/response and charged ledger call. Missing or altered assessment evidence returns FAIL. VLM measurements are not human artistic-quality judgments.

Historical/pressured commits are labeled `committed under synthesis pressure, N sources, grounding unverified`. Reports project this label without rewriting recovery output. The viewer distinguishes `(empty)` from `(not captured)`, shows draft diffs, exact prompt text, per-step tokens and projection removals. Snapshot paths are relative to their owning run/source root; consumers resolve them at I/O boundaries. External index/resource roots and historical provenance still require explicit reconciliation when moving a whole run.

## Replay and verification

The [replay harness](../../src/kusama/research_replay.py) and [regression test](../../tests/test_research_trace_replay.py) were built after the production changes and focused tests. They use the original [16-step trace fixture](../../tests/fixtures/research/trace_16_step.json), not assistant-authored replacement actions. Original step hashes, raw completions and attributed source evidence remain in the fixture; image bytes stay in the original local artifact archive.

The replay rejects requested steps **5, 8, 9, 11, 12, 13 and 15**, blocks **3** on relevance, and rejects **16** on grounding. All 16 original actions also lack a reason, so they all fail the new rationale requirement. Pointwise checks retain each step's recorded incoming evidence to expose later failures; a separate causal replay stops after three rejected calls. It does not pretend blocked acquisition somehow occurred in a new run. Replay PASS means the expected software rejections occurred; recorded artistic acceptance remains FAIL.

```sh
uv run --locked python -m kusama.research_replay \
  tests/fixtures/research/trace_16_step.json \
  --output artifacts/research_prompt_audit/replay_report.json
uv run --locked pytest -q
KUSAMA_PROCESSOR_ACCEPTANCE=1 uv run --locked pytest -q \
  tests/test_research_grounded.py tests/test_research_processor_acceptance.py
uv run --locked python scripts/generate_contract_types.py --check
```

Verification results are recorded in the final handoff below. Earlier recovery fixtures explicitly retain their pre-v13 protocol through `tests/research_legacy.py`; current v13 tests separately exercise the new executor, ledgers, provider queries, canonical IDs, reinspection, rendering, judge evidence, resume and source-count rules. No historical success is presented as a new v13 live artistic run.

The next evidence gate is a separately admitted full v13 artistic run: useful relevant sources, accurate visual/source interpretation, a rendered predicate check and actual human assessment followed by selected-parent revision. No human votes, learned weights or artistic advantage were established by this repair. Local artifact links above require the preserved workspace and do not resolve in a fresh checkout.

## Final verification handoff

- Full suite: **986 passed, 29 optional checks skipped**.
- Explicit cached-processor/current-rule/16-step replay run: **48 passed, zero skipped**, including the three preserved incident archives. These cover 28 of the optional checks above; the actual full v13 live-process gate remains unrun.
- Final combined processor, replay, recovery, render and integrity check: **108 passed, zero skipped**.
- Follow-up research/source/render/integrity verification: **130 passed** with its one opt-in processor case subsequently covered by the explicit run.
- TypeScript declarations match Pydantic, JavaScript syntax checks pass, and all local documentation links in the changed guides resolve in this workspace.
- The replay exposed and fixed omitted-versus-empty optional fields producing different draft hashes. Its required relevance, grounding and unchanged-draft failures now come directly from the actual executor without supplying missing model reasons.

The full suite's live-process check still requires a newly completed unmocked v13 run and a running dashboard. It is not satisfied by the older v11 render or by the single image-access diagnostic.
