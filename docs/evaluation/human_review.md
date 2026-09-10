# Historical review storage

The development panel no longer provides A/B comparisons, pair votes or pool-ranking controls. Its review API routes and comparison browser tests were removed on 8 September 2026 at the user's request. Use the [development panel](../workflows/dev_panel.md) to inspect saved work, make attributed development selections, prepare continuations and record observations.

Existing `reviews.sqlite3` stores remain preserved. Shared review contracts and readers support historical review provenance, existing human-advancement IDs in selected-parent continuation, and grouped dataset lineage. Existing labels are never replaced, fabricated or inferred from model observations. Historical stores may contain empty review collections; their existence is not evidence of a human judgment.

Continuation and dataset code retain this compatibility so old images, parent selections and outcomes remain interpretable. The removed E0, Phase 1 and workflow comparators are not available for new comparisons. See the [retirement record](../operations/2026-09-08_dev_panel.md).

Operator and model selections stay explicitly attributed. Observations do not become finished-quality or development-potential votes. None of these records starts model inference or establishes learned artistic advantage.
