# Kusama v2: direction, current state and evidence gates

Assessment date: **8 September 2026**. Capability and review-store assessment updated at **2026-09-08T06:21:25Z**; source/index verification is dated 05:56 UTC and saved v2 recovery verification is dated 06:19 UTC. The earlier 04:17 technical verification and Maya memory counts below remain explicitly historical. Later workers may still be executing: this is a dated assessment, not a live status feed. [Assessment evidence](../artifacts/overnight/recovery_audits/capability_assessment_evidence.json) records the checked stores and artifact hashes.

Design source: [Kusama v2 technical architecture and research specification, version 2.1](https://docs.google.com/document/d/1NmD6y8WxAOiCQdIFmCGhflEIxU9mleDU/edit). An [offline text extraction](architecture_spec_v2.1_source.md) is included. That document assumes an empty repository and describes proposed research. This assessment compares it with the code and saved evidence that actually exist.

## Where we are

**Kusama is a working, instrumented, frozen-model artistic experiment system. It is at an early research stage, before demonstrated preference learning or a demonstrated advantage over strong artistic workflows.**

The first renderer experiment, controlled editing, a frozen visual director, reference-conditioned proposals, initial painting studies, durable records and human review are real. The newer source-connected studio adds bounded official collection ingestion, audited whole/detail retrieval, source-relationship proposals, an explicit selected-parent bridge, realized branches, decision-time experiment memory and gated dataset exports. These provide a foundation for testing the larger ambition. The central research loop—use critique and past outcomes to make better decisions, then prove that improvement on unfamiliar work—has not yet been demonstrated.

There is no defensible percentage of completion toward “superhuman artistic abilities.” The infrastructure is substantially further along than the artistic evidence. The source's first vertical slice has been executed; its E0 comparative experiment has not been completed; H1–H4 remain untested. Calling this a capable experimental prototype is justified. Calling it a self-improving artist is premature.

## What the ambition means

The source's distinctive ambition is to find and transform aesthetically and intellectually consequential reference relationships beyond human capability **on defined tasks**, then develop stronger work and sustained practices. It is broader than imitating an artist's appearance or making attractive images.

The potential advantage is a growing record of which connections, interventions and critiques actually helped: what was tried, what changed, what different viewers valued, what deserved further development, and whether those lessons transferred. More source images, model parameters or polished explanations cannot establish that advantage by themselves.

A defensible future superiority claim would identify the task, human comparison group, references/tools available, time and resource limits, presentation conditions, independent reviewers, held-out projects and uncertainty. It would establish a bounded result under that protocol. It would not establish universal beauty, historical importance or superiority to every artist.

**We are on a plausible path to testing this ambition, not on an already validated path to achieving it.** Staying on that path means making the next artistic comparison informative. Frozen models can improve as a system through better tools, context, retrieval and search; weight training is a separate possible improvement, not the definition of all progress.

## What the system does today

The earlier curated-reference workflow remains supported:

1. An operator supplies a brief, curated source records and local reference images. The runner validates and snapshots them.
2. Frozen Qwen sees the images and context and proposes concrete pictorial directions, source relationships, predicted effects and FLUX prompts.
3. Frozen FLUX receives each saved **text prompt** and makes two initial images using different seeds. These have no parent image. In this workflow, the source images are inputs to Qwen, not direct image conditioning for FLUX.
4. An optional frozen Qwen pass observes each sibling pair and predicts finished-quality and development-potential preferences, with a possible revision. These are model predictions.
5. Humans can compare saved images in the dashboard and explicitly choose a candidate to develop. Choices are recorded; they do not execute another image call.

The newer [source-connected studio](source_connected_studio.md) adds a second executable path. Bounded official APIs supply rights-tagged source records and byte-preserved images. Frozen CLIP, lexical search and unvalidated typed relationships assemble a whole/detail shelf. Qwen sees selected source images and proposes source roles, transformations, predicted effects, whole/part relationships and a standalone rendering instruction. FLUX realizes saved instructions; a versioned archive retains descriptor coverage and protected alternatives. With no validated quality model or real human preference labels, selection uses an explicitly attributed seeded exploratory policy.

The [continuation bridge](continuation.md) now carries an exact saved parent, inquiry, references, proposed revision, preservation constraints, two seeds and a bounded budget into the persistent studio. It accepts either an actual human advancement submission or a separately attributed operator/model selection. Qwen can propose a local intervention after seeing the actual parent. The controlled editor supports whole-parent edits or FLUX followed by a rectangular compositor; preserved outside pixels do not guarantee successful local semantics or edges. Actual children receive lineage, observations and statement-free follow-up review collections. Preparing a continuation or voting does not execute it; execution remains an explicit action.

Optional [experiment memory](experiment_memory.md) retrieves attributed preceding actions, actual artifacts, observations and failures. Each new development decision freezes its eligible candidate set, cutoff and at most four admitted events; Qwen receives a bounded text excerpt and is told that the prior images were not shown. Resume reuses that exact snapshot. Sealed episodes are excluded from development memory; the heldout API permits only its own preceding non-review state. Memory usefulness remains unvalidated.

The overnight profiles are explicit and separate: `flux_base4b_1024_v1` at 1024 × 1024 and `flux_base4b_study_512_v1` at 512 × 640. Both declare 50 steps, guidance 4.0, BF16, model offload and VAE tiling. The E0-only alternative `flux_base4b_e0_384x512_v1` explicitly declares 384 × 512 with those settings; its smaller dimensions do not by themselves establish throughput or quality. Profile identity and actual execution remain in frozen inputs. See [session profiles](../configs/overnight.yaml) and [E0 profile rules](EXPERIMENT_PROTOCOL.md).

Today we are primarily **testing the tools and collecting the ingredients of an artistic decision dataset**. The reviewed parent → revision → follow-up pathway exists in software, but no actual human advancement has exercised that evidence gate in the checked review stores. Saving feedback, changing a prompt and retrieving a history are different from training a selector or showing that experience improves unfamiliar work.

## Capability map

| Area | What exists | What remains unproven or absent |
| --- | --- | --- |
| Image execution | Real FLUX generation/editing; deterministic geometry and local repair; saved hashes and lineage | Reliable semantic control for arbitrary edits; general improvement in artistic quality |
| Direction | Real frozen Qwen multimodal proposals with saved prompts and raw responses | A learned action policy; independently discovered diversity in the scaffolded V3 run |
| References | Curated packets plus 200 audited official records, geometric crops, whole/detail lexical/CLIP retrieval and 54 typed machine relationships | Comprehensive cultural discovery, verified semantic localization, human retrieval usefulness and learned connection selection |
| Inquiry and development | Versioned inquiry, explicit human/operator/model parent-selection contracts, realized branches and follow-up review registration | A genuinely reviewed continuation showing improvement; sustained developing practice |
| Observation | Frozen Qwen sibling predictions and actual parent/child visible-change observations, separately attributed | Reliable valid responses in every case; validated visual accuracy or independent artistic judgment |
| Human feedback | Balanced A/B queues, quality/potential and pool/none choices, readiness/context/exposure fields, explicit advancement and follow-up collections | Actual labels in the checked stores; validated review protocol and practice critique panels |
| Resources and recovery | Shared invocation reservations, protected exploratory allocations, immutable inputs, no-call recovery sidecars and cost journals | Benefit of the allocation policy; complete economic totals without sourced rates |
| Experiment memory | Append-only event projection, decision-time cutoff checks, candidate snapshots and optional bounded prompt integration | Demonstrated artistic benefit from memory; complete lineage/family resolution |
| Learning | Grouped immutable dataset exports, sealed-input guards, cheap-selector implementation and optional SFT mechanics gates | Adequate genuine human data, locally verified multimodal training, research adapters, DPO and evaluated learned releases |
| Comparative research | Historical diagnostic schedule, E0 planner/runner, explicit profile/throughput gates and matched development-mode entrypoints | Completed human-reviewed matched comparisons, E0 outcome, human benchmark results, H1–H4 and transfer studies |

The earlier entry points remain [creative.py](../src/kusama/creative.py), [director.py](../src/kusama/director.py), [backend.py](../src/kusama/backend.py), [references.py](../src/kusama/references.py), [studio.py](../src/kusama/studio.py), [reviews.py](../src/kusama/reviews.py), [execution.py](../src/kusama/execution.py), [ledger.py](../src/kusama/ledger.py) and [costs.py](../src/kusama/costs.py). The new path uses [corpus.py](../src/kusama/corpus.py), [retrieval.py](../src/kusama/retrieval.py), [inspiration.py](../src/kusama/inspiration.py), [episodes.py](../src/kusama/episodes.py), [continuation.py](../src/kusama/continuation.py), [memory.py](../src/kusama/memory.py), [datasets.py](../src/kusama/datasets.py) and [learning.py](../src/kusama/learning.py). These extend the existing package and ledgers rather than reproducing the specification's proposed folder tree.

## Evidence we can claim

### Earlier interventions

- **Original mountain experiment:** one parent and two independent whole-image edits executed. Neither edit fulfilled the full one-third-size intervention and preservation request. This established real inference and exposed a control weakness. See [execution evidence](verification.md) and [visual review](visual_review.md).
- **Qwen-directed mountain edit:** one real Qwen proposal and one FLUX edit completed. The human response about greater vastness was uncertain. This established the handoff and a useful negative/uncertain result, not an artistic preference win. See [director verification](director_verification.md).
- **Controlled scale experiment:** code scaled the original figure, while FLUX repaired the exposed background. Measured height ratio was 0.3365 and changed protected pixels were zero. The user confirmed a change in the feeling of vastness. The question did not establish an explicit preference or finished-quality improvement. The compositional intervention was supplied by the operator; Qwen formulated it. See [scale experiment](scale_experiment.md).

### Maya memory V3 snapshot

Local run: [`20260908T022716_creative_ee890a8f`](../artifacts/20260908T022716_creative_ee890a8f/manifest.json). The manifest was last updated at `2026-09-08T04:17:11.653718+00:00` in the completed diagnostic pass.

| Item | Observed state |
| --- | --- |
| Source packet | 12 reference records: four images and eight text-only sources |
| Directions | 12 validated Qwen formulations within 12 operator-supplied composition challenges |
| Saved paintings | 24 RGB PNGs at 512 × 640; all decoded and matched their manifest hashes during this audit |
| Observer completion | All 12 original pairs attempted; 7 validated responses, 5 unresolved pairs (d02, d04, d05, d06, d10); failures preserved |
| Overall status | `needs_human_review`; observer status `partial`, with the bounded recovery job closed. Technical integrity passed; artistic success remains unassessed. |
| Calls | 48 primary invocations + 2 recovery invocations = **50 actual invocations**; original plan 48, combined authorized ceiling 60 |
| Human queue | One registered 108-pair collection, asking two questions per pair |
| Human quality/potential feedback | 0 submitted votes; 0 explicit advancement events in `artifacts/reviews.sqlite3` |
| Final run verification | [Technical integrity report](../artifacts/20260908T022716_creative_ee890a8f/verification.json) passed; it explicitly reports 7 valid/5 unresolved observer pairs, 43 usable outputs and 7 retained failed attempts |

The counts above describe that completed historical diagnostic, whose bounded recovery was closed. They do not describe active later jobs. Consult current manifests before resuming or reporting later progress. Partial valid observer coverage does not prevent humans from reviewing saved images; model judgments remain unreliable and must not substitute for human feedback.

The [24-image contact sheet](../artifacts/20260908T022716_creative_ee890a8f/contact-sheet-24.png) shows different compositions, including figures, surfaces, landscapes and abstraction. My contact-sheet inspection is an assistant observation, not an independent human quality assessment. The visible breadth is partly explained by [operator composition scaffolding](../references/maya_memory/diversity_scaffold.json); it cannot be credited entirely to Qwen's autonomous invention. The preceding V2 run retained six closely related directions before rejecting a duplicate title, with zero paintings generated.

The current 108 comparisons × 2 questions = **216 planned dimension responses**, not 216 collected judgments, not 216 independent trials, and not the source specification's E0. E0 proposes 12 briefs, 84 images, three reviewers, 216 parent/edit judgments and 36 pool selections. The current single-brief generation study does not satisfy those conditions. Historical human feedback about vastness exists separately; zero current quality/potential votes does not mean the project has never received any human feedback.

### Source-connected studio and recovery snapshot

The [05:56 UTC source audit](../artifacts/overnight/source_audit_final/summary.json) verifies 200 records from four official providers, 400 geometric crops, 502 documented statements, 43 machine interpretation claims and 40 unvalidated device seeds. The frozen final index contains 199 whole views and 398 crops after excluding one faint specimen and its descendants. Every admitted view passed the recorded rights/hash checks, and all 54 typed relationships exactly matched their saved backing claims. A CPU-only 12-query replay reproduced the same source IDs and ordering. These checks establish reproducibility and grounding, not cultural completeness or human usefulness. [Coverage gaps](SOURCE_AUDIT.md) include imprecise dates, unresolved place/culture fields, sparse distant relationships and unestablished contemporary-art representation.

The `care_control_v2` run has six source-connected initial FLUX images under the named 512 × 640 study profile. Its [06:19 UTC saved-response replay](../artifacts/overnight/recovery_audits/care_control_v2_no_call_replay.json) retained all six connections and image hashes, preserved the renderer's generation projection, and left the completed observation job unchanged. Source proposal accounting stayed at seven charged Qwen invocations. The replay used normal locking with explicit model-load, dispatch and reservation traps: it made zero new calls and loaded no model. This is a recovery check, not an artistic evaluation or a total count for all later branch work.

One failed flat-wire response was recovered through an explicit coordinator-metadata sidecar. It supplies only absent request ID, neutral title, fixed external brief and null source influence levels. Replaying the sidecar recomputes the result from exact original raw bytes; it cannot supply a missing rendering instruction or change a transformation. The original failed response, repair call and costs remain recorded. The later `connection-flat-wire-v3-medium` system prompt explicitly requests a two-dimensional collage or painting; this operator-authored change applies to future runs and does not retroactively change the v2 requests or demonstrate better outcomes.

The older V3 diversity challenges, the current source-pairing rule, archive niches, protected allocations and seeded fallback selections are operator scaffolding. Qwen's artistic proposals and parent-conditioned local revisions remain model-authored; assistant observations and geometric checks remain separately attributed. The selected-parent bridge has executable operator/model paths and a contract for real advancement IDs. It has not yet supplied evidence of improvement from actual human choice.

At **06:21:25 UTC**, read-only checks found **zero actual pairwise votes and zero advancements** in both the historical and overnight review databases; the overnight pool-vote table also held **zero selections**. No human quality/development labels were created by preparation, generation, observation, recovery or this assessment. Historical comments about vastness remain legitimate separate human observations. No trained research selector/adapter, learned artistic advantage or completed human benchmark victory is established. [Training and eligibility status](MODEL_AND_TRAINING_STATUS.md) distinguishes implemented checks from unexecuted multimodal training mechanics.

### Historical software and resource evidence — 04:17 UTC

The suite at the **04:17 UTC historical verification** passed **402 tests**, with two dependency deprecation warnings, in 15.40 seconds (`.venv/bin/python -m pytest -q`). This is not the current repository's test total. Those tests exercised software contracts using fixtures and controlled substitutes; they did not evaluate the artistic merit of the paintings. Later implementation and test evidence is recorded separately.

The V3 primary/recovery cost records total approximately **80.70 minutes of recorded operation runtime**, including model loads and failed work. This is not an all-project total or complete human-time measurement. Local API fees are recorded as zero; electricity and hardware totals remain unknown without rates. See [cost accounting](cost_accounting.md).

Generated evidence and SQLite databases are local and ignored by Git. Their links will not work in a fresh checkout without a separately preserved artifact root. The original documentation audit ran no model inference and added no reviews. The historical 04:17 update incorporated its separately executed final observer pass and [cost snapshot](../artifacts/20260908T022716_creative_ee890a8f/cost-summary.json), with no new human votes or advancements. Its checkout then had no Git commits; that is a historical provenance note, not a statement about the current repository. The current capability assessment and saved-response replay also made no inference calls or human labels.

## What would move us forward

The following is an evidence sequence derived from the source and the observed gaps. Several software components now exist; the milestones below require their actual reviewed use and comparison. This is not an instruction to start every stage at once.

### 1. Turn the existing studies into reviewed development

Collect actual quality and development-potential reviews of the saved studies, including ties, uncertainty and rejected possibilities. Record explicit choices about what to develop and why. Preserve the failed observer responses; diagnose/recover them under the existing bounded invocation contract if useful. Human review need not wait for model opinions.

Use the implemented continuation bridge with an actual human advancement. Verify that the chosen image hash, reviewer decision, brief/inquiry, source context, proposed change, preservation requirements, seeds and budget survive the transition. Preserve operator/model authorship and keep execution explicit. Current operator/model-selected branches test execution; they do not substitute for this human-feedback gate.

**Completion evidence:** a reviewer-selected saved parent → a declared intervention → actual saved alternatives → visible-change observations → independent finished-quality/potential review. Include at least one retained alternative where useful. A revision being generated, or a selection button being clicked, is not the end of this milestone.

### 2. Establish that the process helps

Use several briefs and reserve unfamiliar briefs before tuning the protocol. Exercise the implemented E0 or matched-development runner under its declared profile, input-freeze and throughput gates; do not rename the older 24-image study as E0. Validate the implemented review context, exposure/time, readiness and best-candidate-or-none fields with actual reviewers. Protocol registration and runnable code do not constitute completed comparisons.

Compare direct best-of-N, a strong ordinary critique-and-edit workflow and the declared Kusama process at matched source access and total resource budgets. Record operator scaffolding and human effort in every condition. Separate a fixed-pool selection test from a generation-workflow comparison.

**Completion evidence:** humans assess actual artifacts on held-out briefs; failures and costs remain in the comparison; results report uncertainty and differences by reviewer/brief. First establish whether useful feedback changes outcomes. The source's eight sealed E0 briefs and proposed 55% preference target are planning choices, not automatic statistical proof.

### 3. Build usable memory and only the retrieval that resolves a measured gap

Use the implemented dataset exporter on genuinely reviewed transitions, then audit its immutable snapshots. Keep related projects, descendants and crops together when splitting. Exclude source/artist families when testing transfer to those families; ordinary new-project transfer may use a declared fixed shared reference archive. Verify that upstream event envelopes disclose all relevant relationships: exporter guards cannot reconstruct omitted provenance. Keep human, model and automatically derived labels distinct.

Evaluate the existing audited reference retrieval and optional experiment memory. Check frozen candidate sets, selected sources, proposed relationships and decision-time cutoffs on actual decisions. Separate documented context from inferred analogy. Expand the archive only when missed references or weak coverage demonstrably limit the work.

**Completion evidence:** snapshot validation rejects leakage; source bytes and labels resolve by hash; a matched comparison shows whether retrieval or experiment memory improves unfamiliar work. An indexed archive or a complete event log alone does not pass this gate.

### 4. Test learned decisions separately

Exercise the implemented cheap selector against frozen baselines once sufficient varied human feedback exists. Then test a reviewer-conditioned judge; train a director only from reviewed actions and development outcomes. The optional training executor still needs an actual supported-device check of multimodal masks, intended gradients, frozen parameters and save/reload before a research run. Local inference fit does not establish training fit.

Keep renderer weights frozen initially, as specified. Version judge adaptation, director SFT, optional DPO, retrieval changes and critique assistance separately. The source's 500/1,000/3,000 learning-curve checkpoints are diagnostic planning quantities, not guarantees of sufficiency.

**Completion evidence:** each learned component improves its declared held-out outcome over the otherwise identical frozen system, with cost and uncertainty reported. Lower training loss or greater agreement with the model's own judgments cannot establish artistic progress.

### 5. Test breadth and sustained practice

Develop several works across new subjects, source families, practices and reviewers. Evaluate series before and after critique with relevant independent humans, preserving dissent and delayed responses. Test whether prior successful operations transfer without collapsing to one appearance. Give protected exploratory branches explicit resources and measure whether that helps.

**Completion evidence:** repeated improvements in unfamiliar projects and consequential continuations at declared budgets. Only after bounded human comparisons should the project make a bounded superiority claim. Broader media require real tools and outcomes in those media.

## Keep the research claims separate

| Source claim | Required comparison | Current evidence |
| --- | --- | --- |
| H1: reviewer-conditioned selection helps | Same candidates and target reviewer/context; compare profile, ID-only and frozen few-shot conditioning alongside the generic baseline, separating conditioning from weight adaptation | Untested; no trained selector or current V3 human votes |
| H2: branching and experiment memory help | Strong simpler workflows, matched access/cost, actual returned works | Bounded branching and memory implemented; advantage untested |
| H3: weight adaptation helps | Otherwise identical trained versus untrained system | Untested; no weight updates |
| H4: useful operations transfer | Unfamiliar subject/source/practice groups under declared split rules | Untested |
| Later: critique improves a practice | Matched development opportunities and subsequent bodies of work | No demonstrated practice-level outcome |

The immediate priority is **reviewed, traceable artistic development that can be compared with a strong baseline**. This is how the repo moves from producing experiments to learning which artistic decisions are worth retaining.

## Maintenance

Read [AGENTS.md](../AGENTS.md) for coding-agent working rules. Keep commands in the workflow guides: [source-connected studio](source_connected_studio.md), [continuation](continuation.md), [experiment memory](experiment_memory.md), [reference loop](reference_loop.md), [human review](human_review.md), [studio](studio.md), [controlled editing](controlled_editing.md) and [recovery](recovery.md).

When evidence changes, update the snapshot timestamp, run state, actual review counts, demonstrated outcome and next gate together. Preserve older experimental records. Cite actual artifacts or explicit human responses; do not promote an ambition, parser success, model prediction or planned budget into a measured result.
