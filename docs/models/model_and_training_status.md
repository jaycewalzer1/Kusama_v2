# Model and training status

**Implementation update — 9 September 2026:** the optional LoRA/QLoRA judge/director training code, associated tests and dependency diagnostics have been removed at the user’s request. The supported training command is now `kusama train selector`. Dataset exports remain available independently of any training method. The dated records below preserve the earlier unexecuted training assessment.

Assessment: **8 September 2026, 13:48:22 UTC**. Qwen and FLUX remain frozen. No research selector, judge adapter, director adapter or integration adapter was trained. No human preference learning, transfer advantage or improvement against simpler workflows has been demonstrated.

The read-only review backup contains **9 collections, 42 candidate registrations resolving to 37 distinct hash-verified images, 0 human pairwise votes, 0 pool choices and 0 advancements**. The complete studio episode contributes 16 distinct images; the three complete E0 pools contribute 21. The saved d02 parent is not yet registered for review. These registrations and model/assistant observations add zero human labels.

The immutable [final dataset/training status](../../artifacts/overnight/evaluation_v2/dataset_training_status_final.json) records the cutoff, review database backup hash, exact commands, implementation identities, dataset hashes and missing requirements. Fresh outputs are under `artifacts/overnight/evaluation_v2/final_20260908T134822Z/`. Each kind has zero source events and zero eligible rows in every split. The five kinds beyond selector/judge require explicit upstream action/retrieval/continuation event envelopes; generated images are not automatically converted to supported targets. Zero rejected rows means no event envelopes were supplied.

| Dataset kind | Snapshot ID | Eligible rows | Rejected rows | Eligible groups |
| --- | --- | ---: | ---: | ---: |
| selector | `a3936409babef948aa7bf5cc` | 0 | 0 | 0 |
| judge_sft | `d336eb9ba7c2f78057190961` | 0 | 0 | 0 |
| director_sft | `d4cafb79bb52bda11a76119d` | 0 | 0 | 0 |
| retrieval | `410f79847f53ea9ed88f8bbe` | 0 | 0 | 0 |
| action_preferences | `9885f5bed27863fbd20214cf` | 0 | 0 | 0 |
| continuations | `6c901ff089c05652288807a0` | 0 | 0 | 0 |
| practice_critique | `b9e62f05d00872f49e59e87a` | 0 | 0 | 0 |

Selector, judge and director validators each returned **`insufficient_data`, exit code 2**, with a null checkpoint. No `--execute` or `--smoke` flag was used. No model load, fit, dependency installation or training update occurred. The selector needs genuine labels across at least five independent groups, including three training groups, one development group and two original label classes per reviewer/dimension. Judge/director need at least 20 eligible training rows, three training groups and one development group. Fixture labels remain software-test data only.

Historical [05:05 status](../../artifacts/overnight/evaluation_v2/dataset_training_status.json) and [06:48 status](../../artifacts/overnight/evaluation_v2/dataset_training_status_current.json), snapshots and validator results remain unchanged. The historical pre-overnight review database also contained zero quality/potential votes; earlier user comments about perceived vastness remain separate valid observations, not invented preference labels.

[Dataset exports](../../src/kusama/datasets.py) accept typed source events for selector/judge, director actions, retrieval decisions, matched action preferences, continuations and practice critiques. Human-only export rejects missing/synthetic labels. Every input event must predate its decision. Test labels are unavailable through the training loader. Connected project, brief, trajectory, ancestry and image-hash groups are formed before filtering or augmentation; linked sealed or unresolved rows quarantine their group. At least five unsealed groups are required before approximate 60/20/20 splits; smaller sets remain `insufficient_groups`. Crops and source duplicates additionally group for a source-family-transfer claim.

The new [live lineage resolver](../../src/kusama/dataset_lineage.py) follows hash-registered images, original generation briefs and Studio parent/import links. It resolves hash-bound public development E0 episodes and matched comparison plans, groups exact repeated brief wording and scopes candidate IDs to their collection. Unknown, conflicting or unsupported ancestry is quarantined; explicit curator declarations bind exact collection bytes and cannot bypass a recognized bad or sealed protocol. Export checks the exact A/B hashes recorded in the vote against the current candidates. Sealed brief text is withheld before admission. With zero real votes, the final build reads no review input/label examples and does not inspect sealed text.

A read-only live audit grouped all six care/control collections under their original brief rather than treating disjoint branches as independent projects. The 77-test focused independent review and final 706-test suite validated mechanical grouping and presentation integrity using fixtures. Differently worded versions of one brief still need curator grouping, and source-family transfer needs explicit source provenance. Future E0 edits made through the new runtime-recovery sidecar will remain conservatively quarantined until the adapter validates that authorization chain. An explicit declaration cannot silently bypass this restriction. No empirical learning result follows from an empty dataset or these tests.

The [cheap selector](../../src/kusama/learning.py) implements normalized CLIP vectors and `[eA-eB, abs(eA-eB)]` features, train-only standardization, per-reviewer/per-dimension multinomial logistic regression with L2 regularization, and development-group-balanced regularization selection. Reversal maps A/B correctly, preserves tie/unsure, and retains the original review/group identity. Personalization requires at least three train groups, one development group, and two original label classes. Otherwise it reports an unsupported personalization and a constant baseline when possible. A pinned CLIP embedding cache must identify model, revision, preprocessing and image hashes. The NumPy optimizer was exercised with explicitly fixture-only vectors; real CLIP selection has not been trained. Development reports include log loss, Brier score, order-reversal error and pairwise favorite retention. That retention is not a six-candidate-pool metric.

The former judge/director training entrypoints and their multimodal training helpers are no longer implemented. They never produced a trained checkpoint or a verified GPU integration result. Historical result files, dataset snapshots and the original design specification remain preserved; they do not describe a currently executable training path.

The [CLIP model card](https://huggingface.co/openai/clip-vit-base-patch32) identifies the selector's descriptor family. Its frozen embeddings do not by themselves establish artistic quality. DPO, online RL, renderer adaptation and learned reranking remain future research proposals.

To validate the selector against the saved snapshot, choose a fresh output directory:

```sh
.venv/bin/python -m kusama.learning selector --snapshot artifacts/overnight/evaluation_v2/final_20260908T134822Z/datasets/a3936409babef948aa7bf5cc --output artifacts/overnight/training/selector-next-check
```

Dataset creation returns 0 even for an empty but valid snapshot. Insufficient-data/capability selector results return exit code 2. Result records are immutable and the preflight rejects output-directory reuse. `--execute` enables selector fitting only after its data and embedding-cache prerequisites pass; it does not update Qwen or FLUX. The next learning gate is sufficient grouped human feedback and a pinned CLIP embedding cache, followed by a comparison against frozen baselines. Local artifact paths require the separately preserved artifact root.

## Future problem-statement model

Direction added **8 September 2026**: develop a model that can construct sufficiently complex artistic problem statements before image direction and rendering. The [problem-statement guide](../research/artistic_problem_statements.md) defines adequacy, the user-supplied Mutu primary case study, attribution requirements and proposed evaluation gates. Operators provide and assess these statements today. No dedicated formulation model, dataset contract or validated training result is established; the dataset kinds and existing selector above do not implement this capability.

First collect grounded, attributed formulations and genuine human assessments across independent problems. Compare formulation quality and downstream artistic development with human-authored and frozen-prompt baselines, grouping related problems and excluding sealed outcomes from inputs. Architecture and training method remain open. This future goal neither changes the dated training evidence above nor supersedes the current reviewed-continuation gate.

## Frozen search archive

[search_archive.py](../../src/kusama/search_archive.py) adds a modest CVT archive without training a quality model. It fits up to 32 normalized 512-dimensional centroids once from permitted full-reference vectors, records descriptor/source/version hashes and split exclusions, and rejects explicitly sealed/test pools. Fewer distinct source vectors produce fewer centroids, disclosed in the manifest. The custom NumPy implementation follows the centroid/niche distinction described in the official [pyribs CVTArchive documentation](https://docs.pyribs.org/en/stable/api/ribs.archives.CVTArchive.html); it does not add a pyribs dependency or claim a tested optimizer advantage.

Every verified realized candidate is retained in an append-only SQLite store scoped by brief, reviewer context, descriptor, judge and anchor versions. Only compatible situated quality can replace a scored niche winner. Without a judge, quality remains null and a declared seeded exploration policy determines retention. An active pool reserves at most four scored favorites (or explicitly named exploratory fallback slots), two alternatives selected for descriptor difference, and two protected alternatives. Protected assignments cannot be reclaimed by later high scores; an excess protection request is recorded while the underlying candidate remains saved. Different descriptors indicate representational coverage, not novelty, semantics, preference or emotional effect. Six offline archive tests cover sealed rejection, compatible replacement, complete event retention, scope isolation, idempotency and protected capacity. Live fitting/insertion belongs to the separately recorded rendering run.

The initial bookkeeping was copied to `artifacts/overnight/evaluation/`; newly audited canonical data and protocol bookkeeping lives under `artifacts/overnight/evaluation_v2/`. Original `artifacts/overnight_20260908/` bytes are preserved; copied immutable status records still identify those original paths. The [canonical location sidecar](../../artifacts/overnight/evaluation/canonical_locations.json) resolves copied snapshots and results without rewriting their provenance.
