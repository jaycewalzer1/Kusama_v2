# Planned human diagnostic comparisons

A 12-direction × 2-realization experiment creates 24 candidates. The versioned `balanced_108_v1` schedule plans **108 image comparisons**, each asking two independent questions:

1. Which is better finished work?
2. Which idea deserves further development?

That is **216 planned dimension questions**, not 216 completed human judgments. A/B, tie and unsure are recorded separately for each dimension. Repeated submissions or later revisions do not create new unique comparison coverage.

## Coverage

Every candidate appears in exactly nine comparisons. These include all 12 comparisons between a direction's own two realizations, plus 96 comparisons across directions. Every one of the 66 unordered direction relationships appears at least once: 36 receive one realization pairing, and 30 receive two.

Across directions, realization combinations are balanced: 24 first/first pairs, 24 second/second pairs, and 48 mixed-realization pairs. This avoids the earlier 78-pair schedule's exposure imbalance, where each first realization appeared 12 times and each second realization only once.

For smaller experiments (1–11 directions), all candidate pairs are scheduled. Even 22 candidates produce only 231 pairs, within the review store's 300-pair limit.

## Reproducibility

The default design seed is `kusama-diagnostic-v1`. SHA-256 hashes of the seed and direction IDs determine a stable direction cycle, independent of Qwen's batch order or the artworks' apparent quality. A second deterministic hash ordering intermixes the comparisons. The review store independently assigns A/B sides for each reviewer.

The 12-position direction cycle uses two matched realization comparisons for direction distances 1, 2 and 6, and one comparison for the remaining direction pairs. Alternating realization assignments provides equal candidate exposure and balanced realization combinations. The complete cycle, seed, design version, pair count and per-candidate counts are retained in `design_metadata`.

The schedule is a manageable diagnostic sample. Equal exposure alone does not prove an unbiased ranking, preference-model improvement, or generalization to new briefs. The two answers on each comparison share a stimulus and are not statistically independent examples. Human advancement remains an explicit, separate decision.

Python interface:

```python
from kusama.comparison_design import diagnostic_pairs, design_metadata

ids = [f'd{i:02d}' for i in range(1, 13)]
pairs = diagnostic_pairs(ids)
metadata = design_metadata(ids)
```

Existing registered review collections are immutable. A design change must not replace a collection already holding images, assignments or votes; create a separately identified collection when a different schedule is required.
