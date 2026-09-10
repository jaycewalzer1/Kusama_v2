# Kusama research documentation

Start with [project direction and evidence gates](research/project_direction.md), then read [implementation status](operations/implementation_status.md) and [model and training status](models/model_and_training_status.md). Kusama currently uses frozen models; learned artistic improvement remains unproven. The folders separate proposed research, executable procedures and observed results.

| Folder | Purpose |
| --- | --- |
| [research/](research/) | Research direction, hypotheses, evidence gates and the source design specification |
| [models/](models/) | Model roles, frozen checkpoints, training status and learning prerequisites |
| [data/](data/) | Reference provenance, source audits and decision-time experiment memory |
| [evaluation/](evaluation/) | Comparison protocols, human review, measurement and benchmark procedures |
| [experiments/](experiments/) | Recorded runs, technical results, visual observations and historical human feedback |
| [workflows/](workflows/) | Executable guides for generation, direction, editing and continuation |
| [operations/](operations/) | Implementation status, recovery, resource accounting and dated handoffs |

## Research

- [Project direction](research/project_direction.md): current evidence, unresolved artistic questions and the next evidence gates.
- [Artistic problem statements](research/artistic_problem_statements.md): the required starting point for a full artistic run, the user-supplied Mutu primary case study and the future formulation-model research goal.
- [Version 2.1 source specification](research/architecture_spec_v2.1_source.md): preserved design proposal; its architecture and training recipes do not describe implemented capabilities.

## Models

- [Model and training status](models/model_and_training_status.md): component roles, selector training eligibility, retired judge/director training and historical evidence.

## Data

- [Starter-source audit](data/source_audit.md): collection coverage, rights, attribution, hashes, retrieval and remaining gaps.
- [Experiment memory](data/experiment_memory.md): ledger projection, frozen decision-time inputs, leakage controls and grouped exports.
- [Reference library](../references/README.md): curated source packets and their provenance, kept alongside the reference data.

## Evaluation

Run/workflow comparison tooling was retired at the user’s request on 8 September 2026. See the [retirement record](operations/2026-09-08_dev_panel.md); dated handoffs remain historical.

- [Edit-control benchmark](evaluation/edit_benchmark.md): hash-bound saved failures, prospective tasks, independent measurements and model assertions.

- [Historical review storage](evaluation/human_review.md): preserved review provenance and continuation/dataset compatibility.
- [Verification and repair benchmark](evaluation/verification_benchmark.md): measurement procedures, execution commands and the recorded pilot.
- [Content capability diagnostic](evaluation/content_probe.md): bounded test of adult artistic nudity and controversial subjects, with separate prompt and image assessments.

## Experiment records

- [9 September single-action response failure](operations/2026-09-09_single_action_contract.md): repeated incomplete JSON, preserved charges, new-plan protocol correction and unexecuted replacement preparation.

These are historical records. Their timestamps and attribution remain part of the evidence; use current manifests and ledgers for execution status.

- [9 September stagnation after acquisition](experiments/20260909_stagnated_after_acquisition.md): the latest 15-call run, verified image acquisition, no hypothesis, and the gap between stagnation termination and synthesis controls.

- [9 September research stall](experiments/20260909_research_stall.md): the latest 24-action run, missed source evidence, repeated searches, zero hypotheses and rendering, and proposed next evidence gates.

- [First milestone verification](experiments/verification.md): initial FLUX generation/edit execution and environment measurements.
- [Initial visual review](experiments/visual_review.md): instruction-following assessment of the original parent and edits.
- [Frozen director verification](experiments/director_verification.md): the first Qwen-directed edit and the human response about its intended effect.
- [Controlled scale experiment](experiments/scale_experiment.md): figure geometry, background repair, preservation and perceived vastness.
- [Creative loop verification](experiments/creative_verification.md): Maya memory studies, operator scaffolding, observer failures and recovery.
- [Content diagnostic, 8 September 2026](experiments/content_probe_20260908.md): real adult-nudity and controversial-subject results, with attributed assessments and technical verification.

## Workflows

- [Development panel](workflows/dev_panel.md): workspace navigation, run inspection, development controls, research, sources and tools.

- [Independent reference research](workflows/independent_research.md): standing catalogs, model-directed inquiry and reference sets, explicit research-to-render execution, dashboard evidence and opt-in research during painting.
- [Qwen-selected image prompts](workflows/image_prompts.md): optional inspected reference images passed directly to FLUX for generation and parent revisions.
- [Bounded visual development](workflows/bounded_development.md): proposal, controlled alternatives, observation, attributed selection, recovery and stop under frozen budgets.

Run commands from the repository root. Start with the [repository setup instructions](../README.md#setup-from-a-fresh-checkout).

- [Source-connected studio](workflows/source_connected_studio.md): bounded ingestion, retrieval, proposals and realized branches.
- [Selected-parent continuation](workflows/continuation.md): carry an attributed selection into a frozen revision and follow-up review.
- [Reference-informed creative loop](workflows/reference_loop.md): curated packets, sibling generations, observation and resume controls.
- [Persistent studio](workflows/studio.md): inquiry state, action history and explicit branch selection.
- [Frozen Qwen director](workflows/frozen_director.md): propose and execute a single image edit.
- [Controlled editing](workflows/controlled_editing.md): edit plans, masks, preservation checks and bounded fallbacks.

## Operations

- [10 September grounded draft contract](operations/2026-09-10_grounded_draft_contract.md): three preserved attribution rejections, evidence-first generation requirements and specific citation feedback without supplied artistic content.

- [9 September repeated JSON failure and live follow-through](operations/2026-09-09_bounded_json_generation.md): bounded schema generation, corrected token filtering and explicitly authorized monitored runs.

- [9 September malformed inspection JSON](operations/2026-09-09_json_generation.md): preserved missing-bracket failure, syntax filtering during new-plan generation and an unexecuted replacement.

- [9 September context capacity repair](operations/2026-09-09_context_capacity.md): token-counted result pages, complete saved-result recall, capacity checkpoints and preserved charged work.
- [9 September expanded context](operations/2026-09-09_expanded_context.md): 16,384-token new-plan limits, matching panel/resource allowances and preserved historical charges.

- [9 September random research exposure](operations/2026-09-09_random_exposure.md): saved chance leads before selection, bounded online bootstrap, catalog bias and retained draws/failures.

- [9 September grounded research repair](operations/2026-09-09_grounded_research.md): raw prompt/image probe, v13 workflow, conservative grounding, measurable render checks and the 16-step replay.

- [9 September nested-hypothesis rejection](operations/2026-09-09_nested_hypothesis.md): missing nested arguments and a later rejection hiding earlier committed work; exact incident replay and v10 correction.

- [9 September live process acceptance](operations/2026-09-09_live_process_acceptance.md): unmocked browser-to-image diagnostic, strict outcome verification, and a release gate that cannot pass with skipped live checks.

- [9 September research acceptance audit](operations/2026-09-09_research_acceptance.md): incident reproduction, bounded synthesis, crash recovery, real cached processor tests, and the repeatable full acceptance gate.

- [9 September online research repair](operations/2026-09-09_research_progress.md): online-only new runs, candidate retention, stagnation controls, early prompt checks, verification and the remaining live evidence gate.

- [Implementation status](operations/implementation_status.md): implemented contracts and their validation boundaries.
- [Recovery](operations/recovery.md): immutable inputs, durable jobs, interrupted work and reconciliation.
- [Cost accounting](operations/cost_accounting.md): invocation costs, local estimates and unknown rates.
- [8 September 2026 code maintenance](operations/2026-09-08_code_cleanup.md): unused imports, lazy CLI dispatch, SQLite connection cleanup and regression verification.
- [8 September 2026 overnight handoff](operations/2026-09-08_overnight_handoff.md): dated results, outstanding work and the expired inference window.
- [8 September 2026 Phase 1 handoff](operations/2026-09-08_phase1_handoff.md): implemented evaluation harness, real cached diagnostic, fixture demonstration and pending review.
- [8 September 2026 bounded development handoff](operations/2026-09-08_bounded_development_handoff.md): edit benchmark, controlled autonomous development, matched workflow and memory comparisons, and retained live contract failures.

- [8 September research/source audit](operations/2026-09-08_research_source_audit.md): online collection checks, action/retrieval defects, repairs, preserved failures and remaining limits.

## Maintaining the documentation

Place new documents by purpose and link them here. Use lowercase snake_case filenames; prefix dated handoffs with `YYYY-MM-DD`. Keep run observations in experiment records and reusable commands in workflow or operations guides. Existing mixed protocol/pilot records remain intact and are indexed by their primary purpose.

Preserve original source text, historical results, failure records and attribution when updating assessments. Software checks, model predictions, assistant observations and human judgments are different kinds of evidence. The next artistic evidence gate remains a genuine human comparison, explicit parent choice, controlled revision and follow-up review.

Relative links to `artifacts/` require the separately preserved local artifact directory and will not resolve in a fresh checkout. This documentation layout does not relocate run artifacts, reference bytes, model caches or frozen inputs. [AGENTS.md](../AGENTS.md) contains the repository working rules.

- [Studies before artistic commitment — 9 September](operations/2026-09-09_studies_before_commitment.md): complete rendering handoff, provisional alternatives, image inspections, model choice and one follow-up.
- [Image-led development cycles — 9 September](operations/2026-09-09_development_cycle.md): bounded research after inspection, repeated revision, retained branches and human-review pauses.
