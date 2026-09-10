# Independent reference research

Implemented **8 September 2026**. Frozen Qwen can choose questions, searches, sources, details, contextual passages and reference sets before proposing paintings. Newly planned visual-development loops can also opt into research. The [automatic-pairing workflow](source_connected_studio.md) remains available as a simpler comparison. Neither model is trained by these commands.

A full artistic run starts with an operator-supplied, assessed [artistic problem statement](../research/artistic_problem_statements.md). The software preserves the statement and its authorship; it does not validate conceptual adequacy. Documented context, model-proposed analogies, operator-authored tools and policies, and human judgments remain separate.

Run commands from the repository root after the [normal setup](../../README.md#setup-from-a-fresh-checkout). Replace `/absolute/path/...` values with actual inputs. Generated requests, ledgers and images belong under ignored `artifacts/`, not in Git.

## Optional chance exposure

New panel runs offer up to two random source leads per research session, adjustable
under **Run settings → Chance leads per research session**. Qwen may ignore, reject
or investigate them. They are sampled before model selection, without a novelty
reward or an obligation to use them. For standalone research requests set
`random_exposure_count` to `1` or `2` (default `0`); optionally supply
`random_exposure_seed` (an integer from 0 to 9007199254740991) to reproduce a draw
within the same eligible pool. Otherwise a new seed is frozen during planning.

The controller samples eligible whole works from the available source pool. If it
is empty, authorized online access permits one random breadth query/provider from
the operator's saved policy. This uses existing limits and returns metadata first;
acquisition and image inspection remain explicit model actions. Catalog bias and
search ranking still shape exposure. Completed draws, empty results and failures
are retained without redraw. See the [implementation and evidence boundaries](../operations/2026-09-09_random_exposure.md).

## Resource boundaries

Research planning freezes inputs and writes local source-store records without Qwen/FLUX inference. Research execution and rendering require an **existing, explicitly admitted resource session** at `resource_root`, with sufficient remaining time and quota. A directory or empty database is insufficient. Planning does not create, extend or renew authorization. The historical overnight session remains expired; follow [resource and recovery boundaries](../operations/recovery.md) when establishing a separate contract for new inference.

New v19 research plans default to **16,384 combined tokens per Qwen call**, with **1,800 reserved for output** and **14,584 available for input**, including image tokens. The panel displays and freezes the same limit for research, actual-image inspection, development choices and reopened research. Its default 42-call allowance is 688,128 tokens. A standalone request can explicitly set `total_token_limit` between 8,192 and 16,384, but its admitted session must allow that per-call reservation. The separate legacy development workflow retains its frozen 8,192-token contract. Existing plans and charges are unchanged. See [expanded context](../operations/2026-09-09_expanded_context.md).

Network acquisition is separately bounded by source configuration and defaults to disabled. Previewing a catalog makes no remote request. `catalog` and `acquire` can request remote data when enabled; neither runs Qwen or FLUX. Descriptor indexing is a separate computation. `research show` and the dashboard research routes read saved evidence without creating an inference session or source database.

## Prepare a broader standing catalog

[configs/research_sources.json](../../configs/research_sources.json) supplies coverage queries and source limits. These queries are operator scaffolding, not Qwen discoveries or a completeness guarantee. The enabled official collection providers are `met`, `artic` and `cleveland`.

Preview a bounded catalog without remote acquisition:

```sh
uv run --locked kusama research-sources plan \
  --root artifacts/research_catalog_preview \
  --config configs/research_sources.json \
  --query-limit 3 --result-limit 4 --provider artic
```

The command initializes a local source store and prints its query/request ceiling. Configuration is frozen when that store is created. To permit acquisition, prepare a separate configuration with `allow_network: true`, explicit budgets and appropriate exclusions **before creating a new root**. Pass the same configuration and optional `--index` on subsequent commands for that root.

The following commands perform remote metadata retrieval and, for `acquire`, the selected download:

```sh
uv run --locked kusama research-sources catalog \
  --root artifacts/research_catalog_01 \
  --config /absolute/path/research-sources-online.json \
  --query-limit 3 --result-limit 4 --provider artic

uv run --locked kusama research-sources acquire \
  --root artifacts/research_catalog_01 \
  --config /absolute/path/research-sources-online.json \
  --candidate-id candidate_ID_FROM_THE_CATALOG

uv run --locked kusama research-sources show \
  --root artifacts/research_catalog_01 \
  --config /absolute/path/research-sources-online.json
```

`catalog` saves metadata candidates without downloading their images. `acquire` requires a recorded candidate ID and reuses the corpus rights and provenance contracts. Failed or unsettled requests/acquisitions retain their reservations and are not silently retried. `show --source-id SOURCE_ID` opens saved source evidence and records a source snapshot; this source-backend command is distinct from the read-only research-report route.

Reuse admitted images across later sessions by building a **new** frozen index, using separately authorized descriptor computation:

```sh
uv run --locked kusama index build \
  --corpus artifacts/research_catalog_01/corpus \
  --output artifacts/research_indexes \
  --device cpu --resource-root /absolute/path/to/admitted-session
```

Use the exact directory printed by the command as a new request's `index_path`. Each research session has its own source store; standing image access comes from its frozen index.

To let Qwen discover the catalog's metadata candidates in local searches, set `source_config.catalog_path` to the standing source-store **root**, such as `artifacts/research_catalog_01`, and optionally set `catalog_limit` (default 64, range 1–300). The root must contain `inputs.json` and `research_sources.sqlite3`. A new research session copies a bounded candidate snapshot and the original API metadata bytes without network requests. It preserves candidate IDs and freezes `catalog_import.json`; resume uses that snapshot even if the standing catalog later grows or moves. The episode's index can differ from the catalog's index.

Catalog import does not copy acquired images or text bodies. Qwen can search imported metadata and explicitly acquire a candidate under its own session's source limits; this requires network permission for any missing remote bytes. For reusable acquired images, build the index as above. Context seeds and rights policies can be supplied to new sessions, but the catalog does not automatically become an unrestricted global text archive.

### Source settings and limitations

The [8 September source audit](../operations/2026-09-08_research_source_audit.md) records live API checks, failure-path tests and current limitations. Met search uses the paginated v1.1 endpoint, inspecting only the requested first page; results may differ from the retired route. Invalid records and unavailable individual Met objects retain exclusion evidence. Local matching uses source metadata rather than the query that discovered a candidate. Prompt search excerpts are bounded and marked; full evidence remains saved.

The defaults allow 48 requests, 64 MiB of charged network bytes, 256 MiB of source storage, 12 acquisitions and 300 candidates per source session, with additional individual-response/image/text bounds. These are ceilings, not expected usage. `allow_network` is false in both the schema and checked-in configuration.

Additional text sources require `context_policies`: an exact HTTPS host, permitted path prefix and explicitly allowed `OperationRight` with `license`, `evidence_url`, `evidence_field`, `attribution` and any `conditions`. `context_seeds` supply URLs, titles and authors. Inspect the [source contracts](../../src/kusama/research_sources.py) when preparing them. Text permission does not grant image, derivative or training rights. HTML is parsed as inert text; scripts and styles are excluded. Redirects and local/IP-address destinations are rejected. Arbitrary web browsing, PDF interpretation and comprehensive historical coverage are not implemented by this backend.

Set `excluded_source_ids` and `excluded_source_families` before freezing a run when a comparison requires held-out sources. These explicit exclusions do not automatically discover related sources or sealed outcomes. Do not place sealed outcomes in source text, context seeds or prior context.

Text, contextual, historical and contradiction modes use bounded lexical evidence; “contradiction” searches a model-written counter-question, not a verified contradiction detector. Relationship search uses saved typed relationships. Visual search uses the frozen index; text-to-CLIP requires `enable_clip_text: true`, and encoding a newly acquired image uses the matching CPU encoder. Exploratory encounters use disclosed seeded policies. These scores do not estimate artistic nuance or quality.

## Plan and run model-directed research

Save a `ResearchRequest` JSON file such as `/absolute/path/research-request.json`. This example uses the user's primary Mutu case-study framing, not a Mutu quotation. Supply the actual index and admitted resource root. `index_path` may be `null` for an explicitly network-enabled session without standing image references.

```json
{
  "id": "mutu_inquiry_01",
  "problem": "How can the Black female body—historically subjected to objectification, colonial classification, and imposed beauty standards—be reconstructed as a source of power, self-definition, and new possibilities for existence?",
  "authorship": "User-supplied primary Mutu case-study framing; not a quotation from Mutu",
  "project_id": "mutu_case_study",
  "brief_id": "mutu_problem_01",
  "partition": "development",
  "index_path": "/absolute/path/to/frozen-index",
  "resource_root": "/absolute/path/to/admitted-session",
  "source_config": {"allow_network": false},
  "max_steps": 24,
  "max_hypotheses": 4,
  "max_sources_per_hypothesis": 4,
  "max_images_per_call": 6,
  "max_wall_seconds": 3600,
  "cache_dir": ".cache/huggingface",
  "device": "mps",
  "precision": "bf16",
  "fixture": false
}
```

`source_config` embeds configuration values, not a filename. For broader policies, copy the parsed contents of `configs/research_sources.json` there before planning. Device and resources must match the admitted session. Optional `parent` contains `path` and the actual `sha256`; optional `prior_context` carries at most 4,000 characters of attributed earlier context. The parent is copied without replacing its original bytes.

```sh
uv run --locked kusama research plan \
  --output artifacts/research/mutu-01 --request /absolute/path/research-request.json
uv run --locked kusama research preflight --output artifacts/research/mutu-01
uv run --locked kusama research show --output artifacts/research/mutu-01
```

Planning writes `plan.json`, its identity seal, source configuration and the optional copied parent. Preflight validates frozen implementation/input identities and evidence; it does not prove model availability, remaining authorization or artistic readiness. After admitting the actual inference resources:

```sh
uv run --locked kusama research run --output artifacts/research/mutu-01
uv run --locked kusama research show --output artifacts/research/mutu-01
uv run --locked kusama research export \
  --output artifacts/research/mutu-01 \
  --destination artifacts/research/mutu-01/hypotheses.json
```

`run` dispatches Qwen. `export` saves committed hypotheses and source snapshots without rendering. Reusing `run` replays frozen records where possible and grants no new budget. Changed inputs or implementation require a new plan. Uncertain charged inference must reconcile its original evidence before another dispatch. Rejected responses receive no implicit repair call.

New plans automatically [page oversized research context](../operations/2026-09-09_context_capacity.md). `sources_page` identifies a saved result page and provides a `recall` cursor: supply `recall_steps`, `offset`, `limit` and your reason to read another page without repeating the search. Full results and original source bytes stay saved. If even the essential context cannot fit, research records `context_paused` and preserves the session without reserving a call; the panel reports retained partial work instead of a fatal context error. A pause is not a model decision or a successful artistic outcome.

New plans use **`independent-research-v28-grounded-draft-contract`**. The [grounded draft correction](../operations/2026-09-10_grounded_draft_contract.md) requires evidence before each attributed property, at least two formal properties, structured visual predicates and a justification for a one-source hypothesis. Rejected drafts receive `draft_feedback` showing failed wording beside its citation and other missing requirements; copy contiguous supported spans and keep interpretations/discard reasons in `transformation`. The extractive validator remains strict. The [bounded-generation correction](../operations/2026-09-09_bounded_json_generation.md) enforces required action arguments and existing string limits during Qwen generation, including valid tokens crossing string boundaries; it does not repair saved responses or supply artistic content. Strict final parsing and existing action validation remain. The [single-action repair](../operations/2026-09-09_single_action_contract.md) adds an explicit one-object response protocol and a required-field display derived from the frozen schema for the currently allowed actions. `rewrite_queries` must include 3–5 `queries` objects, each with `query` and `field`; a reason alone does not rewrite anything. `search` uses a prepared query and requires `mode: "remote"` when no local index exists. Multiple JSON objects remain a terminal rejected response; no prefix is accepted, no missing argument is supplied, and no repair call is made.

The [development-cycle extension](../operations/2026-09-09_development_cycle.md) lets the panel reopen research from an inspected branch with a model-authored working question, while preserving the original operator problem and context. The grounded-search rules remain; the [study-development extension](../operations/2026-09-09_studies_before_commitment.md) adds provisional alternatives and image-led choice. Read the [grounded-search contract and verification record](../operations/2026-09-09_grounded_research.md) for the action fields, relevance gate, citations, reinspection and acceptance criteria.

`analyze_question` is the one-time step-0 analysis, with 3–5 short concrete queries; `rewrite_queries` supplies later reformulations. Search fans out over configured providers. `assess_candidates` records question-specific model relevance scores; acquisition needs a score at/above the frozen threshold (default 0.55) or explicit `relevance_override`. Optional request fields are `seed_artists`, `seed_works`, `medium_constraint`, `relevance_threshold` and `max_reformulations` (default 3). These do not authorize inference or network access implicitly.

Every action requires a nonempty reason. Exact raw completions remain saved on parse failures. Drafts require cited transferred/discarded source properties and measurable `visible_predicates`, plus two sources or a `single_source_justification`. Draft automatically attaches images. Commit requires `reinspection` observations; unchanged redrafts and repeated actions are rejected. A pending synthesis draft permits commit or stop only. Source citations use conservative extractive entailment; image-observation novelty uses a lexical caption screen. Neither is a validated artistic judge.

Actual pinned Qwen processor counting still precedes model loading/reservation. Admission uses the plan's frozen combined limit, reserving 1,800 tokens for output. Disclosed projections and candidate paging remain available when a prompt exceeds that allowance; the problem, operator context, images and pending draft remain intact. Tighter projections may remove schema display annotations while preserving validation constraints. Full and admitted contexts, the raw serialized prompt, token counts and omission paths remain inspectable. Irreducible inputs save an uncharged capacity checkpoint and partial work. New snapshots use paths relative to their run/source root.

Composition and render instruction are validated together against the pinned FLUX tokenizer and frozen maximum of 512 tokens. Initial FLUX input concatenates those two fields and attaches only explicitly selected `render_images`. Rendering diagnostics disclose one charged VLM geometry measurement call per image. Their report stays FAIL until all four diagnostic acceptance criteria pass. Full panel study development uses qualitative image inspections and an artistic decision instead; these do not manufacture geometry measurements or satisfy that separate diagnostic gate. No automatic retry, manufactured source property or human review is supplied by these controls.

## Render committed hypotheses

Rendering has a separate request and output directory. This preparation script reads completed research and an existing session, selects **all** committed hypotheses, assigns one seed to each and writes the full `Config.to_dict()` contract. It makes no model call and creates no resource session. Replace the admitted-session path before using it.

```sh
uv run --locked python - <<'PY'
from pathlib import Path
from kusama.config import Config
from kusama.resource_snapshot import session_snapshot
from kusama.research import read_report
from kusama.research_render import ResearchRenderRequest

research_output = Path('artifacts/research/mutu-01').resolve()
output = Path('artifacts/research_renders/mutu-01').resolve()
resource_root = Path('/absolute/path/to/admitted-session').resolve()
session = session_snapshot(resource_root)
profile_id = 'flux_base4b_study_512_v1'
profile = session['config']['render_profiles'][profile_id]
config = Config(output_dir=output,
    cache_dir=Path(session['config']['cache_dir']).resolve(),
    device=session['config']['device'], local_files_only=True,
    max_retries=0, **profile)
report = read_report(research_output)
ids = [hypothesis['id'] for hypothesis in report['hypotheses']]
request = ResearchRenderRequest.model_validate({
    'id': 'mutu_render_01', 'research_output': research_output,
    'hypothesis_ids': ids, 'seeds': [17 + 2 * i for i in range(len(ids))],
    'resource_root': resource_root, 'render_profile_id': profile_id,
    'render_config': config.to_dict(),
    'selection_origin': 'all_model_hypotheses',
    'selection_reason': 'Realize every committed hypothesis for individual inspection.'
})
destination = Path('artifacts/research_render_requests/mutu-01.json')
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open('x') as handle:
    handle.write(request.model_dump_json(indent=2))
print(destination)
PY

uv run --locked kusama research render-plan \
  --output artifacts/research_renders/mutu-01 \
  --request artifacts/research_render_requests/mutu-01.json
uv run --locked kusama research render-preflight --output artifacts/research_renders/mutu-01
uv run --locked kusama research render --output artifacts/research_renders/mutu-01
```

`render-plan` freezes evidence, hypotheses and prompts; `render-preflight` validates them; `render` performs charged image calls. At least one committed hypothesis is required. A subset needs `selection_origin: "operator_selection"` and a reason. Supply exactly one or two distinct seeds per hypothesis; each hypothesis receives its consecutive seed block. Selection does not establish human preference.

The example explicitly selects `flux_base4b_study_512_v1` at 512 × 640. `flux_base4b_1024_v1` and the separately admitted `flux_base4b_e0_384x512_v1` remain different profiles. A missing/different profile is rejected, with no automatic substitution or demonstrated smaller-profile quality gain. Initial-generation configuration requires its exact output directory, cached checkpoints, zero retries and no `parent_image`. Prompts exceeding the renderer's token capacity are rejected rather than truncated.

Images retain prompts, source/hypothesis linkage and hashes; failures remain charged and saved. At least two images create a review collection **inside the render output directory**, without votes. Open that directory with the existing review dashboard:

```sh
uv run --locked kusama review --artifacts artifacts/research_renders/mutu-01 --port 8766
```

## Keep research open during development

For a **new** [bounded development](bounded_development.md) `LoopRequest`, add this optional extension:

```json
{
  "research": {
    "index_path": "/absolute/path/to/frozen-index",
    "source_config": {"allow_network": false},
    "max_steps": 12,
    "max_sources": 4,
    "max_wall_seconds": 1800
  }
}
```

This fragment is not a complete request. Preserve the actual parent hash, original brief/inquiry, requirements, explicit seeds, render profile and all other required fields. Set `render_config.output_dir` to the new output directory, then use:

```sh
uv run --locked kusama develop plan \
  --output artifacts/development/research-01 --request /absolute/path/new-loop-request.json
uv run --locked kusama develop preflight --output artifacts/development/research-01
uv run --locked kusama develop run --output artifacts/development/research-01
```

Before each editing round, an additional Qwen controller sees the actual parent and decides whether to keep references or formulate a research question. Requested research saves evidence under `research/round-NN/`, commits at most one hypothesis and freezes adopted inputs in `round-NN-sources.json`. A rejected controller or research without selected sources stops that path rather than inventing references. Resume reuses the adopted snapshots. Image references retain exact pixels and context; text-only references retain their text, original bytes, provenance and bounded prompt excerpt. Text references are never passed to an image decoder.

For `R` rounds and `S` research steps per round, the combined ceiling is `R * (4 + S)` Qwen calls: three existing development calls, one research controller and up to `S` research calls per round. Renderer attempts remain `2 * R`. All share the admitted inference session. Network/request/acquisition limits are per research session, with aggregate caps recorded in the development plan. The parent loop's deadline also bounds new research dispatches.

Opt-in plans use `bounded-development-v6-reference-research` and allow up to six explicitly budgeted references, counting image and text sources together. Without `research`, the two-reference legacy path and original budgets remain. Do not add research to an already frozen plan.

## Inspect, preserve and verify

```sh
uv run --locked kusama review --artifacts artifacts --port 8765
```

Open **Inspiration explorer and development gallery → Research sessions**. The dashboard finds research plans in the supplied artifact tree, including nested development research. It shows original/current questions, chronological actions and rejected leads, hypotheses, selected reference sets, images/context, authorship and frozen policies. Read-only routes are `/api/research`, `/api/research/{id}` and `/api/research/{id}/images/{image_id}`. Image display checks hashes, containment and rights. Research execution remains a CLI operation.

Preserve plans/seals, decision-time inputs, raw responses, immutable `step-NNN.json` records, `jobs.sqlite3`, source ledgers, original source bytes, copied images, exports and resource/cost ledgers together. Reports are projections, not replacements for evidence. Absolute paths and external index/resource roots require explicit reconciliation when moved; original frozen records stay intact. Local artifacts do not resolve from a fresh checkout without restoration.

```sh
uv run --locked pytest -q tests/test_research_sources.py tests/test_research.py \
  tests/test_research_render.py tests/test_development_research.py tests/test_research_dashboard.py
uv run --locked python scripts/generate_contract_types.py --check
```

Fixtures exercise source access, passage inspection, budgets, replay, rendering transfer, development adoption and dashboard integrity. They are labeled and cannot dispatch real Qwen or FLUX backends. Passing these checks does not establish original discovery, historical correctness, artistic quality or an advantage over simpler workflows. The next evidence gate is separately admitted real research and rendering followed by inspection of the actual proposals and images, preserving weak results and uncertainty for human review.


For study development, set `study_development: true` in both the new research request and render request, allow at least three hypotheses, and explicitly admit `N + 2` extra Qwen calls plus one possible FLUX call beyond the `N` initial images. Each hypothesis must state `artistic_operation`, `study_reason` and `distinct_from_previous`; research commit freezes the study without choosing a winning artistic interpretation. The panel supplies this policy by default. Separate rendering diagnostics retain the optional geometric measurement path.
