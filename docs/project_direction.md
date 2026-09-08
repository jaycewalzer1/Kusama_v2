# Kusama v2: direction, current state and evidence gates

Assessment date: **8 September 2026**. Runtime evidence updated through **2026-09-08T04:17:27Z** from the completed technical verification. The original documentation audit was at 04:04:36Z; its earlier execution counts have been updated below. This remains a dated assessment, not a live status feed.

Design source: [Kusama v2 technical architecture and research specification, version 2.1](https://docs.google.com/document/d/1NmD6y8WxAOiCQdIFmCGhflEIxU9mleDU/edit). An [offline text extraction](architecture_spec_v2.1_source.md) is included. That document assumes an empty repository and describes proposed research. This assessment compares it with the code and saved evidence that actually exist.

## Where we are

**Kusama is a working, instrumented, frozen-model artistic experiment system. It is at an early research stage, before demonstrated preference learning or a demonstrated advantage over strong artistic workflows.**

The first renderer experiment, controlled editing, a frozen visual director, reference-conditioned proposals, initial painting studies, durable records and human review are real. These provide a credible foundation for testing the larger ambition. The central research loop—use critique and past outcomes to make better decisions, then prove that improvement on unfamiliar work—has not yet been demonstrated.

There is no defensible percentage of completion toward “superhuman artistic abilities.” The infrastructure is substantially further along than the artistic evidence. The source's first vertical slice has been executed; its E0 comparative experiment has not been completed; H1–H4 remain untested. Calling this a capable experimental prototype is justified. Calling it a self-improving artist is premature.

## What the ambition means

The source's distinctive ambition is to find and transform aesthetically and intellectually consequential reference relationships beyond human capability **on defined tasks**, then develop stronger work and sustained practices. It is broader than imitating an artist's appearance or making attractive images.

The potential advantage is a growing record of which connections, interventions and critiques actually helped: what was tried, what changed, what different viewers valued, what deserved further development, and whether those lessons transferred. More source images, model parameters or polished explanations cannot establish that advantage by themselves.

A defensible future superiority claim would identify the task, human comparison group, references/tools available, time and resource limits, presentation conditions, independent reviewers, held-out projects and uncertainty. It would establish a bounded result under that protocol. It would not establish universal beauty, historical importance or superiority to every artist.

**We are on a plausible path to testing this ambition, not on an already validated path to achieving it.** Staying on that path means making the next artistic comparison informative. Frozen models can improve as a system through better tools, context, retrieval and search; weight training is a separate possible improvement, not the definition of all progress.

## What the system does today

The current creative workflow is:

1. An operator supplies a brief, curated source records and local reference images. The runner validates and snapshots them.
2. Frozen Qwen sees the images and context and proposes concrete pictorial directions, source relationships, predicted effects and FLUX prompts.
3. Frozen FLUX receives each saved **text prompt** and makes two initial images using different seeds. These have no parent image. In this workflow, the source images are inputs to Qwen, not direct image conditioning for FLUX.
4. An optional frozen Qwen pass observes each sibling pair and predicts finished-quality and development-potential preferences, with a possible revision. These are model predictions.
5. Humans can compare saved images in the dashboard and explicitly choose a candidate to develop. Choices are recorded; they do not execute another image call.

The separate controlled editor supports explicit geometry, masks, compositing and bounded repair. The persistent studio records inquiry versions, branches, selected parents and manually submitted batches. The single-edit director, creative runner and studio are not yet one integrated system that carries a human choice through revision and follow-up evaluation automatically.

Today we are primarily **testing the tools and collecting the ingredients of an artistic decision dataset**. Some prior feedback can be supplied in a later prompt. Saving feedback, changing a prompt and keeping a history are different from training a selector or showing that retrieved experience improves new work.

## Capability map

| Area | What exists | What remains unproven or absent |
| --- | --- | --- |
| Image execution | Real FLUX generation/editing; deterministic geometry and local repair; saved hashes and lineage | Reliable semantic control for arbitrary edits; general improvement in artistic quality |
| Direction | Real frozen Qwen multimodal proposals with saved prompts and raw responses | A learned action policy; independently discovered diversity in the scaffolded V3 run |
| References | Small sourced packets, metadata, source IDs and byte-preserved images | Automated broad discovery, lexical/CLIP retrieval, a typed connection store and learned connection selection |
| Inquiry and development | Versioned studio inquiry, explicit branches and parent selection | An integrated reviewed continuation workflow; evidence of a sustained developing practice |
| Observation | Frozen Qwen sibling observations and predicted preferences; partial real execution | Reliable valid responses for every pair; validated visual accuracy or independent artistic judgment |
| Human feedback | Balanced comparison queue, stable A/B assignment, separate quality/potential answers, explicit advancement | A substantial reviewed dataset; complete exposure/time/context protocol and practice critique panels |
| Resources and recovery | Invocation reservation, immutable inputs, recovery records, runtime/memory and cost journals | Full research allocation policy with protected exploration and demonstrated benefit; complete economic totals without configured rates |
| Experiment memory | Persistent before/action/after evidence and event histories in existing workflows | Decision-time retrieval with validated split boundaries and evidence that memory improves outcomes |
| Learning | Frozen inference and records from which future datasets could be built | Dataset snapshots, sealed split enforcement, calibrated reviewer-conditioned J, D/J adapters, SFT/DPO and evaluated releases |
| Comparative research | Software tests, geometric benchmarks and a within-brief human diagnostic schedule | Completed E0, strong equal-cost workflow comparisons, human benchmark results, H1–H4 and transfer studies |

Implementation entry points: [creative.py](../src/kusama/creative.py), [director.py](../src/kusama/director.py), [backend.py](../src/kusama/backend.py), [references.py](../src/kusama/references.py), [studio.py](../src/kusama/studio.py), [reviews.py](../src/kusama/reviews.py), [execution.py](../src/kusama/execution.py), [ledger.py](../src/kusama/ledger.py) and [costs.py](../src/kusama/costs.py). The actual layout differs from the specification's proposed future module tree.

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

The counts above are a completed diagnostic snapshot. No inference remains running. Consult the current manifest before resuming or reporting later progress. Partial valid observer coverage does not prevent humans from reviewing the saved images; model judgments remain unreliable and must not substitute for human feedback.

The [24-image contact sheet](../artifacts/20260908T022716_creative_ee890a8f/contact-sheet-24.png) shows different compositions, including figures, surfaces, landscapes and abstraction. My contact-sheet inspection is an assistant observation, not an independent human quality assessment. The visible breadth is partly explained by [operator composition scaffolding](../references/maya_memory/diversity_scaffold.json); it cannot be credited entirely to Qwen's autonomous invention. The preceding V2 run retained six closely related directions before rejecting a duplicate title, with zero paintings generated.

The current 108 comparisons × 2 questions = **216 planned dimension responses**, not 216 collected judgments, not 216 independent trials, and not the source specification's E0. E0 proposes 12 briefs, 84 images, three reviewers, 216 parent/edit judgments and 36 pool selections. The current single-brief generation study does not satisfy those conditions. Historical human feedback about vastness exists separately; zero current quality/potential votes does not mean the project has never received any human feedback.

### Software and resource evidence

The updated suite passed **402 tests**, with two dependency deprecation warnings, in 15.40 seconds during final software verification (`.venv/bin/python -m pytest -q`). These tests exercise software contracts using fixtures and controlled substitutes. They do not evaluate the artistic merit of the paintings. Earlier test totals in milestone documents are historical.

The V3 primary/recovery cost records total approximately **80.70 minutes of recorded operation runtime**, including model loads and failed work. This is not an all-project total or complete human-time measurement. Local API fees are recorded as zero; electricity and hardware totals remain unknown without rates. See [cost accounting](cost_accounting.md).

Generated evidence and SQLite databases are local and ignored by Git. Their links will not work in a fresh checkout without a separately preserved artifact root. The original documentation audit ran no model inference and added no reviews. This update incorporates the separately executed final observer pass and [cost snapshot](../artifacts/20260908T022716_creative_ee890a8f/cost-summary.json); no human votes or advancement events were added. It reflects local code without a Git commit identifier: this checkout had no commits at audit time.

## What would move us forward

The following is a recommended implementation sequence derived from the source and the observed gaps. It is a roadmap, not a report of completed work or an instruction to start all stages at once.

### 1. Turn the existing studies into reviewed development

Collect actual quality and development-potential reviews of the saved studies, including ties, uncertainty and rejected possibilities. Record explicit choices about what to develop and why. Preserve the failed observer responses; diagnose/recover them under the existing bounded invocation contract if useful. Human review need not wait for model opinions.

Build the smallest missing bridge from an explicit creative advancement to a studio parent and executable revision. Carry the chosen image hash, reviewer decision, brief/inquiry, source context, proposed change, preservation requirements, seeds and budget into the transition. Preserve operator/model authorship. Keep execution explicit under the workflow's existing controls.

**Completion evidence:** a reviewer-selected saved parent → a declared intervention → actual saved alternatives → visible-change observations → independent finished-quality/potential review. Include at least one retained alternative where useful. A revision being generated, or a selection button being clicked, is not the end of this milestone.

### 2. Establish that the process helps

Use several briefs and reserve unfamiliar briefs before tuning the protocol. Implement the source's E0 design or a clearly versioned smaller pilot; do not rename this 24-image study as E0. Add the missing review context, exposure/time records, readiness judgments and best-candidate-or-none pool choices where the protocol requires them.

Compare direct best-of-N, a strong ordinary critique-and-edit workflow and the declared Kusama process at matched source access and total resource budgets. Record operator scaffolding and human effort in every condition. Separate a fixed-pool selection test from a generation-workflow comparison.

**Completion evidence:** humans assess actual artifacts on held-out briefs; failures and costs remain in the comparison; results report uncertainty and differences by reviewer/brief. First establish whether useful feedback changes outcomes. The source's eight sealed E0 briefs and proposed 55% preference target are planning choices, not automatic statistical proof.

### 3. Build usable memory and only the retrieval that resolves a measured gap

Connect reviewed transitions to immutable dataset snapshots. Keep related projects, descendants and crops together when splitting. Exclude source/artist families when testing transfer to those families; ordinary new-project transfer may use a declared fixed shared reference archive. Freeze decision-time inputs so later outcomes cannot leak into earlier decisions. Keep human, model and automatically derived labels distinct.

Add a small audited retrieval layer over references and relevant prior experiments. Log candidate sets, selected sources and proposed relationships. Separate documented context from inferred analogy. Expand the archive when missed references or weak coverage demonstrably limit the work.

**Completion evidence:** snapshot validation rejects leakage; source bytes and labels resolve by hash; a matched comparison shows whether retrieval or experiment memory improves unfamiliar work. An indexed archive or a complete event log alone does not pass this gate.

### 4. Test learned decisions separately

Start with cheap/frozen selection baselines and sufficient varied human feedback. Then test a reviewer-conditioned judge; train a director only from reviewed actions and development outcomes. Profile the actual training device, multimodal masks, intended gradients, frozen parameters and save/reload before larger runs. Local inference fit does not establish training fit.

Keep renderer weights frozen initially, as specified. Version judge adaptation, director SFT, optional DPO, retrieval changes and critique assistance separately. The source's 500/1,000/3,000 learning-curve checkpoints are diagnostic planning quantities, not guarantees of sufficiency.

**Completion evidence:** each learned component improves its declared held-out outcome over the otherwise identical frozen system, with cost and uncertainty reported. Lower training loss or greater agreement with the model's own judgments cannot establish artistic progress.

### 5. Test breadth and sustained practice

Develop several works across new subjects, source families, practices and reviewers. Evaluate series before and after critique with relevant independent humans, preserving dissent and delayed responses. Test whether prior successful operations transfer without collapsing to one appearance. Give protected exploratory branches explicit resources and measure whether that helps.

**Completion evidence:** repeated improvements in unfamiliar projects and consequential continuations at declared budgets. Only after bounded human comparisons should the project make a bounded superiority claim. Broader media require real tools and outcomes in those media.

## Keep the research claims separate

| Source claim | Required comparison | Current evidence |
| --- | --- | --- |
| H1: reviewer-conditioned selection helps | Same candidates and target reviewer/context; compare profile, ID-only and frozen few-shot conditioning alongside the generic baseline, separating conditioning from weight adaptation | Untested; no trained selector or current V3 human votes |
| H2: branching and experiment memory help | Strong simpler workflows, matched access/cost, actual returned works | Infrastructure partly implemented; advantage untested |
| H3: weight adaptation helps | Otherwise identical trained versus untrained system | Untested; no weight updates |
| H4: useful operations transfer | Unfamiliar subject/source/practice groups under declared split rules | Untested |
| Later: critique improves a practice | Matched development opportunities and subsequent bodies of work | No demonstrated practice-level outcome |

The immediate priority is **reviewed, traceable artistic development that can be compared with a strong baseline**. This is how the repo moves from producing experiments to learning which artistic decisions are worth retaining.

## Maintenance

Read [AGENTS.md](../AGENTS.md) for coding-agent working rules. Keep commands in their existing workflow guides: [reference loop](reference_loop.md), [human review](human_review.md), [studio](studio.md), [controlled editing](controlled_editing.md) and [recovery](recovery.md).

When evidence changes, update the snapshot timestamp, run state, actual review counts, demonstrated outcome and next gate together. Preserve older experimental records. Cite actual artifacts or explicit human responses; do not promote an ambition, parser success, model prediction or planned budget into a measured result.
