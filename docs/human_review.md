# Human review

The local dashboard at `http://127.0.0.1:8765/` now includes a human comparison queue. Start it with `kusama-dashboard --artifacts artifacts`. Enter a reviewer name, choose an experiment, and open a comparison.

Each comparison asks two independent questions:

- **Which is better finished work?** Judge the images as they stand.
- **Which idea deserves further development?** Judge their potential after further work.

Each answer is A, B, tie, or unsure. Tie is an equality judgment; unsure is an inability to judge confidently. Confidence is optional for each answer, and notes are optional. No option is selected by default. Both questions must be answered before saving. Saving another review appends a new record; previous reviews remain visible.

The **All images in this experiment** disclosure provides a contact sheet of every candidate. Its development buttons prepare a choice; a reason and an explicit save are still required.

Images occupy equal display areas and can be opened at full size. A/B placement is randomly assigned per reviewer and comparison, then remains stable across reloads. Candidate labels, hashes, parent relationships, and prior reviews appear under the provenance disclosure. Both alternatives and before/after comparisons use the same questions.

## Register an experiment

Registration is a Python API intended for the generation workflow, not a browser file-upload endpoint. All candidates must be existing image files inside the configured artifacts root. The creative runner registers its completed experiment automatically.

```python
from pathlib import Path
from kusama.reviews import ReviewStore

root = Path('artifacts').resolve()
store = ReviewStore(root)
store.register_collection(
    'brief-001',
    title='A brief with two directions',
    brief='The exact artistic inquiry shown to the human reviewer.',
    candidates=[
        {'id': 'direction-1-r1', 'path': root / 'brief-001' / 'direction-1-r1.png',
         'label': 'Direction 1, realization 1', 'direction_id': 'direction-1'},
        {'id': 'direction-1-r2', 'path': root / 'brief-001' / 'direction-1-r2.png',
         'label': 'Direction 1, realization 2', 'direction_id': 'direction-1'},
        {'id': 'direction-2-r1', 'path': root / 'brief-001' / 'direction-2-r1.png',
         'label': 'Direction 2, realization 1', 'direction_id': 'direction-2'},
    ],
    pairs=[
        {'id': 'direction-1-realisations',
         'candidate_ids': ['direction-1-r1', 'direction-1-r2'], 'kind': 'alternatives'},
        {'id': 'directions-1-vs-2',
         'candidate_ids': ['direction-1-r1', 'direction-2-r1'], 'kind': 'alternatives'},
    ],
)
```

Candidate IDs and comparison IDs allow 1–100 letters, digits, underscores, or hyphens. Candidate fields are `id`, `path`, optional `label`, `direction_id`, `parent_candidate_id`, and `studio`. A parent ID must refer to another candidate in the collection. Optional `studio` metadata has exactly `project_id`, `episode_id`, and `artifact_id`. Other creative details remain in the generation manifest.

There may be 2–100 candidates and 1–300 comparisons. Omit `pairs` to generate every unordered pair, provided that stays within the comparison limit. The 12-direction creative run registers a balanced plan of **108 comparisons**: 12 sibling comparisons between realizations of the same direction, plus 96 comparisons across directions. Every candidate appears in nine comparisons, and all 66 direction-pair relationships are represented. Each comparison asks two independent questions, giving **216 planned dimension questions**, not 216 completed human judgments. Pair scheduling is frozen separately from reviewer-specific A/B placement. See [comparison design](comparison_design.md) for the exact coverage and reproducibility rules.

Registration snapshots dimensions, paths and SHA-256 values. Collection IDs are immutable: identical registration is idempotent; changing an image, brief, or comparison plan requires a new collection ID. Missing or changed images cannot be displayed or reviewed. Symlinks and paths outside the artifacts directory are rejected.

## Explicit development choices

The disclosure **Choose a candidate to develop next** records an explicit candidate, reviewer name, and reason. It does not infer a choice from votes and does not start image generation. These choices are returned by `store.history(collection_id)['advancements']`. A later creative continuation can use them as evidence of the human's selection.

Existing studio projects also expose **Select for development**, which calls the established studio selection method. This changes the episode's selected artifact only when its batch has finished and the episode is open. It does not execute the next generation. The generic creative collection choice and legacy studio selection are separate records.

## Durable records and local access

`artifacts/reviews.sqlite3` contains immutable collection snapshots and append-only human votes and advancement events. Database triggers prevent updates or deletion through the application database. Each vote retains its reviewer, timestamp, exact A/B candidate IDs and hashes, both choices and confidences, notes, and submission ID. Repeated submission of the same request is idempotent, including concurrent retries. A changed request must have a new submission ID.

HTTP routes live under `/api/reviews/collections`. Read routes list collections, fetch a collection, fetch candidate images, obtain a reviewer-specific pair, and return history. Writes are `POST .../{collection_id}/pairs/{pair_id}/votes` and `POST .../{collection_id}/advance`; they require JSON and `X-Kusama-Review: 1`. The existing loopback host and same-origin checks remain in place. The reviewer name is attribution in a local application, not account authentication.

Opening the queue does not create a review database when none exists. Automated verification uses temporary collections and does not add synthetic preferences to the real experiment history. Human votes remain distinct from Qwen judgments and technical image validation.

## Observer response status

Run details separately show valid Qwen responses out of the planned direction pairs, unresolved rejected pairs, and observer status. A run ready for human review can still have a partial observer pass. `rejected_response_count` includes retained rejected attempts that may later have been resolved; it is not the number of unresolved pairs. A valid response means it passed the response contract, not that its artistic judgment is correct. Qwen observations remain separate from human votes.

## Live generation progress

While any run is busy, the dashboard polls run metadata every 15 seconds and updates the selected run's saved-image count, gallery, timing and cost evidence. Existing image elements remain loaded. Current run selection, page/table scroll, active comparisons and unsaved review inputs are preserved; polling does not submit votes or reopen comparison forms. It stops when runs settle, discovers newly registered review collections, and retries failed reads after 60 seconds. The manual Refresh button remains available.

## Monetary display

Run details read a manifest's `cost_accounting` descriptor and display each currency separately. An explicit valid `run_ids` list aggregates the primary run and its separately authorized recovery work; older descriptors retain their single `run_id` scope. Creative run call counts and limits use recorded `total_model_calls` and `authorized_maximum_model_calls` when present, while the original primary plan stays visible in the raw manifest. Missing records display **Not recorded**. Incomplete records display **Total unknown** with the known subtotal; a local API fee of zero is not presented as zero total cost. Estimated totals are labeled as estimates. See [cost accounting](cost_accounting.md) for rate configuration and scope.
