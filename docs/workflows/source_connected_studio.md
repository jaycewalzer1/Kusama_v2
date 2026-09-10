# Source-connected frozen studio

This workflow extends the existing Python application, SQLite ledgers and FastAPI
dashboard. The earlier curated-packet runner remains available. Qwen interprets
the selected reference images; FLUX receives a saved text prompt for initial
generation and a saved parent image for editing. Both models remain frozen.

For model-directed questions, repeated searches and source sets of one or several references, use [independent reference research](independent_research.md). It adds a separate bounded research/render path while preserving this automatic-pairing workflow for comparison.

Run commands from the repository root after `uv sync --locked --dev`. The
`kusama` umbrella command and `python -m kusama` use the same implementation.

```sh
uv run --locked kusama doctor --root artifacts/overnight
uv run --locked kusama ingest report --root artifacts/overnight/corpus
uv run --locked kusama verify --root artifacts/overnight
uv run --locked kusama review --artifacts artifacts/overnight --port 8766
```

The artifact root freezes its session configuration on first use. The overnight
root has an eight-hour horizon and a protected 45-minute handoff window. An
expired root is inspectable but cannot silently become a new inference budget.
The saved session expired at 12:40:47 UTC on 8 September 2026; new inference dispatch ended at 11:55:47 UTC. Starting another authorized session requires explicit reconciliation of prior calls and a separate resource contract. There is no automatic migration of frozen E0 paths or budgets. The exact prepared recovery command and current blocker are documented in the [E0 recovery section](../operations/2026-09-08_dev_panel.md).

## Propose without rendering

Before a new full artistic run, the operator establishes a [sufficiently complex artistic problem statement](../research/artistic_problem_statements.md), then carries its question, stakes and unresolved tension into the supplied `--brief`. Preserve the relationship between the larger problem and any narrower episode brief. The documented primary case study is the user's Mutu inquiry; the short connection/loneliness command below illustrates the interface and is not a complete problem-development record. The CLI accepts supplied text and does not validate conceptual adequacy or construct an independently validated starting problem.

The following uses the final audited starter index. Its path is local evidence,
not a file included in Git. A fresh checkout needs the corpus and index restored
or rebuilt through `kusama ingest ingest` and `kusama index build`.

```sh
uv run --locked kusama inspire \
  --index artifacts/overnight/indexes/849a29aaaec9d1959002 \
  --brief 'How can connection itself produce loneliness? Avoid prescribing a literal phone or social-media image.' \
  --run-id connection_loneliness --directions 6
```

Choose a new run ID for new work. This command makes real Qwen calls, snapshots
the selected shelf and exact source bytes, and retains invalid proposals. It
does not call FLUX. `--scope part --local-need '…' --whole-context '…'` conditions
retrieval and proposals on a selected-part need within its whole composition.
It does not create a semantic segmentation mask.

For a new development run, optional `--memory` builds the experiment-event
projection and freezes up to four past events, the complete candidate set and a
bounded attributed text excerpt. Future and sealed events are excluded. Resume
uses that same snapshot; later outcomes do not alter earlier model inputs. This
is a lexical memory baseline, with no demonstrated learned advantage.

```sh
uv run --locked kusama inspire --resume artifacts/overnight/runs/connection_loneliness
```

Resume reuses frozen inputs. Unsupported legacy wire versions are preserved and
require a separately named new run. An explicit `--revalidate-metadata` pass can
recover only missing coordinator identifiers, neutral titles, the fixed brief,
and null influence-level metadata from supported saved flat-wire responses. It
makes no model call, changes no artistic instruction, preserves the rejection,
and writes a separately hashed attribution sidecar.

## Develop saved images

Replace the example run path below with the actual inspiration result. At least
six valid distinct proposed directions are required before initial generation.

```sh
uv run --locked kusama run --run artifacts/overnight/runs/connection_loneliness --stage all
uv run --locked kusama finish --run artifacts/overnight/runs/connection_loneliness --artifacts artifacts/overnight --stage all
```

`run --stage all` creates six whole-image alternatives, computes the frozen CLIP
archive, asks for four local proposals, executes two seeds for each, and observes
the eight actual parent/child pairs. Individual stages are `generate`, `archive`,
`propose-revisions`, `revise`, `observe`, and `gallery`. `resume` is an alias of
the same durable episode executor; it does not grant another model invocation.

An observer may report an empty `visible_changes` list. For earlier responses
rejected only because the old schema required a change, the explicit
`--stage revalidate-observations` command accepts the unchanged report in a
hashed sidecar after verifying the original charged response and exact image
pair. It preserves the original rejection and makes no inference call. This
does not turn a no-change report into a successful intervention.

There is no validated quality model. Archive selection therefore uses a declared
seeded exploratory fallback, leaves quality null, and preserves two protected
directions. Four ordinary and four protected image attempts follow the initial
six. Failures consume their allocations. `finish` can add exactly two attempts
from an actually observed protected child, bringing the ceiling to 16. If the
required prior outputs or observations are absent, it reports the unmet gate.

Local editing uses FLUX on the full parent followed by a hard rectangular
compositor. Pixels outside the rectangle are preserved; semantic correctness
and boundary quality inside it are not guaranteed. Raw images, measured
spillover, actual instructions, lineage and hashes remain available.

The live dashboard shows source details, proposed connections, whole images,
continuations and statement-free review collections. It extends the existing
HTML/JavaScript interface; generated TypeScript declarations follow the Pydantic
contracts. A React rewrite was not required for this working path. Preparing a
continuation or submitting a vote does not automatically render. Explicit
execution by registered ID uses the shared model lease and reservations.

See [continuation and review contracts](continuation.md),
[experiment memory](../data/experiment_memory.md), [E0 protocol](../operations/2026-09-08_dev_panel.md),
and [training eligibility](../models/model_and_training_status.md). Model observations,
operator selection, and actual human reviews remain separately attributed.

## Verify and preserve

```sh
uv run --locked pytest -q
uv run --locked python scripts/generate_contract_types.py --check
uv run --locked kusama verify --root artifacts/overnight
uv run --locked kusama export --root artifacts/overnight --output artifacts/overnight/exports/morning
```

Export requires a fresh output directory. Run it after workers have settled for
the final handoff. It preserves artifact bytes, takes consistent SQLite backups,
and writes checksums and an absolute-to-portable path map. Model caches and code
are separate. Restoring to another path requires explicit path reconciliation;
original frozen manifests are never rewritten to imply that relocation occurred
at the original decision time. Integrity checks do not evaluate artistic merit.

## Method sources and limits

The connection contract separates source identification, mapped attributes and
proposed implications, informed by the IEI framework in
[Peng et al. (2025)](https://arxiv.org/html/2504.13120v1). Its visual-mashup task
and comparisons do not validate this studio, general artistic merit or Qwen's
new analogies. No CreativeMashup images or labels were ingested or used to train.

The archive retains distinct descriptor niches, following the separation between
variation and performance in [MAP-Elites](https://arxiv.org/abs/1504.04909).
The [pyribs QDAIF](https://docs.pyribs.org/en/stable/tutorials/qdaif.html) and
[human-feedback tutorial](https://docs.pyribs.org/en/stable/tutorials/qdhf.html)
provide implementation references for archives, variation and evaluation. Here,
CLIP descriptors measure representation distance; absent a validated judge or
human feedback, they cannot supply artistic quality. The custom NumPy CVT
implementation is a modest exploration mechanism, not an evaluated advantage.

## Search method actually used

Reference retrieval runs SQLite FTS5/BM25 over the complete query, cosine similarity over pinned normalized CLIP embeddings for whole views and geometric crops, and matches to explicitly typed relationship claims. Reciprocal-rank fusion uses `k=60`; parent/source-family deduplication and a declared relevant/distant/underrepresented shelf policy follow it. Sparse typed relationships trigger recorded fallback, not invented connections. The director ultimately sees at most two source images per call.

Creative search uses a custom spherical CVT-MAP-Elites-style archive. Spherical k-means fits 32 frozen niches from eligible reference CLIP vectors before candidate search. Real images enter the compatible niche; every candidate remains in the event store. With no validated judge, quality is null, retention uses a seeded exploratory ordering, and the active pool preserves descriptor-different and protected alternatives. This is a small archive-guided branching experiment, not evidence of quality optimization. E0 instead uses a fixed same-parent three-action ×two-seed design and a two-source lexical shelf; it does not use archive selection. Repository code search during implementation uses `rg`/`rg --files`.
