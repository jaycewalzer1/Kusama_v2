# Experiment memory — 8 September 2026

`kusama.memory` builds a separate append-only SQLite projection from the actual
execution, studio continuation, human review and local cost ledgers. It makes no
model calls, writes nothing to those source databases and reads them with SQLite
`mode=ro` transactions. It does not import mutable manifests as historical facts.

```sh
.venv/bin/python -m kusama.memory build --root artifacts/overnight
.venv/bin/python -m kusama.memory query --root artifacts/overnight \
  --brief 'Enclosure supports a figure; care and control' \
  --decision-at '2026-09-08T05:30:00+00:00' \
  --episode-id next_development --purpose development --limit 8
.venv/bin/python -m pytest -q tests/test_memory.py
```

Use the actual decision cutoff, including a timezone. Naive timestamps and future
cutoffs are rejected. The cutoff is exclusive: `available_at < decision_at`.
The example is a historical lookup, not a claim that a decision occurred then.

The Python interface is `build_memory(root)` and
`ExperimentMemory(root).query(brief, decision_at, episode_id, purpose='development',
exclude_source_families=(), limit=8)`. `ExperimentMemory` is a context manager.
`MemoryEvent` and `MemoryQuery` are strict Pydantic contracts. Query results contain
`snapshot_id`, `eligible_candidates` with scores and provenance, and `selected`
attributed events with exact artifact paths and hashes.

Future development inspiration runs can opt in with `python -m kusama.inspiration
--memory` or `inspire(..., use_memory=True)`. This first builds memory, records the
actual decision cutoff, queries at most four events and freezes the full candidate
snapshot and canonical SHA-256 in the immutable run spec and `memory.json`. All
direction calls receive the same excerpt of at most 1,800 characters. The excerpt
identifies event ID, kind, attribution and time and explicitly says that prior
images were not presented. Resume validates and replays the saved excerpt without
building or querying memory again. The feature defaults off; it cannot be added
to frozen old inputs, and the E0 fixed empty-memory condition is separate.

The normalized memory preserves action instructions and intended effects, actual
image evidence, selected parent and realized child hashes, available source IDs
and family IDs, observations with their original attribution, retained failures,
call reservations, measured runtime and explicitly incomplete cost estimates.
An intended effect remains a prediction. A model observation or operator selection
does not become a human preference label. Pairwise reviews, pool selections and
advancements are separate `human_review` events only when actual rows exist in
the human review ledger; advancement is not converted into a quality vote.

Global resource reservation/settlement events also retain actual token counts
where present; session-wide unassigned configuration events remain unclassified.
Each event preserves source database, table, event ID or submission ID, sequence
where available, timestamp and canonical source-row checksum. Exact source rows
are retained in `raw_row_json` for audit. Output/response/parent/child files are
hashed at ingest. Selected files and source authority rows are verified again at
query time. Missing or changed evidence prevents that query. Ingest issues remain
in immutable build reports; changed prior event identities are not overwritten.

Development queries exclude sealed and unclassified episodes. Heldout queries
allow only the requested episode's own preceding non-review state and exclude
all other episodes. Identical image bytes shared by distinct episodes cannot
declassify a sealed review. `--exclude-source-family` excludes every event whose
available source families intersect the supplied value. Unknown lineage and
cross-episode review ownership are conservatively unclassified. The builder does
not open sealed brief files to infer partitions.

Ranking is a transparent lexical baseline over operation, relationship, context
and observed-consequence text. Payload token overlap has weight 2 and brief
context has weight .25; ties use event time and ID. The bounded result reserves
slots for a matching observation, actual image/transition and failed case when
available. Budget events are retrieved on budget-related queries. This is not a
learned artistic judgment. The full eligible ranked set and selected events are
frozen under `memory/queries/<snapshot_id>.json` and `memory_queries` in
`memory/memory.sqlite3`. Future rejected content and counts are deliberately absent
from the decision snapshot, so later ingestion cannot alter an earlier query.

At the live check on 8 September 2026, the memory held 406 actual events across
`care_control` and `care_control_v2`: 132 artifact, 100 budget, 127 execution,
28 failure, 15 action and 4 context events, with zero ingest issues and zero human
reviews. Continuation preparation had begun; no completed child observation was
yet available in that snapshot. These are dated event counts, not unique images,
independent experiments or artistic successes. Re-run build to ingest later
append-only events. Eleven fixture tests cover cutoffs, future-image/observation/
review stability, source integrity, append-only logs, failure retention,
continuation lineage, shared-image leakage and genuine-review separation.

A later read-only-source build at approximately 07:06 UTC contained 1,321 events:
330 artifact, 414 budget, 441 execution, 51 action, 40 failure, 19 context and 26
observation events. It reported zero ingest issues; 1,258 events were classified
as development and 63 remained unclassified. A live historical query selected an
attributed model proposal, a finishing observation, an actual image artifact and
a retained schema failure. The [query snapshot](../artifacts/overnight/memory/queries/2d860044b385a1fdf79a5f35c6b0941b57aacdc94d4670e2b8cee1faeac28ccd.json)
preserves its exact cutoff and complete candidate set. It was an assistant
retrieval check, not a new director decision or evidence that the retrieved
notes improved an image. Memory did not enter the E0 condition, whose prior
snapshot remains explicitly empty.

The current memory does not establish useful artistic transfer or learning. The
next evidence gate is an attributed memory-assisted decision compared with a
matched source-only decision, followed by genuine human evaluation of the actual
outcomes. Local ignored artifacts and absolute file links do not ship in Git and
will not resolve in a fresh checkout.
