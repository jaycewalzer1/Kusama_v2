# Kusama v2: direction, current state and evidence gates

Assessment date: **8 September 2026**. Final overnight evidence updated at **13:54 UTC**; prior source/index, recovery, Maya memory and review-store snapshots below retain their own dates. The overnight [integrity report](../../artifacts/overnight/integrity_audit_final_complete.json), [implementation status](../operations/implementation_status.md) and [morning handoff](../operations/2026-09-08_overnight_handoff.md) supersede earlier overnight execution totals. That frozen inference window expired and its workers settled. Later software milestones and separately authorized diagnostics are recorded below; current execution status comes from their own manifests and ledgers.

Design source: [Kusama v2 technical architecture and research specification, version 2.1](https://docs.google.com/document/d/1NmD6y8WxAOiCQdIFmCGhflEIxU9mleDU/edit). An [offline text extraction](architecture_spec_v2.1_source.md) is included. That document assumes an empty repository and describes proposed research. This assessment compares it with the code and saved evidence that actually exist.

## Training implementation removal — 9 September 2026

At the user’s request, the optional LoRA/QLoRA judge/director training implementation, its tests and dependency checks have been removed. The CLIP selector and grouped dataset exports remain. No trained adapter existed in the inspected local artifacts; historical results and the source specification remain preserved. Qwen and FLUX continue to run frozen. See [current model and training status](../models/model_and_training_status.md). The next artistic evidence gate remains human assessment of selected-parent revisions and their alternatives.

## Current interface and comparator retirement — 8 September 2026

The user requested a minimal development panel and removal of run comparison tooling. The [panel revision](../operations/2026-09-08_dev_panel.md) supersedes the implementation claims below about the E0 runner, matched workflow comparators, Phase 1 harness and A/B dashboard. Those dedicated engines, tests, configuration and active comparison guides have been removed. Standalone development, independent research, controlled editing, source inspection, memory and dataset/learning infrastructure remain. Historical evidence and existing review/lineage records are preserved. This change makes no new artistic or learned-advantage claim; dated observations below retain their original scope.

The same-day follow-up makes a question-led **New run** screen the main entry point, with explicit local execution limits and a unified run workspace for research, images, activity and files. This joins the existing frozen research and rendering engines; fixture integration validates the software path, not the artistic adequacy of the question or outputs. See the [panel workflow](../workflows/dev_panel.md).

A later read-only audit of local run `studio-ded477219788401096d8` verified three charged research calls and zero committed hypotheses. The third response was rejected because Qwen supplied `limit: 8`, absent from the action schema; its `visual` mode with `provider: "met"` also conflicts with the backend's remote-only provider rule. The new `independent-research-v2-search-contract` accepts integer search limits of 1–8, retains the existing ceiling, and clarifies source-access capabilities and provider rules. The dashboard identifies responses rejected before execution. Verification passed 121 relevant fixture tests plus JavaScript syntax checking. The original run remains unchanged; this repair made no inference calls, human judgments or training updates. A separately admitted new run must establish whether the revised instructions produce valid searches and useful references; artistic adequacy remains unassessed. See the [research workflow](../workflows/independent_research.md).

A subsequent read-only audit of run `studio-6d9ad81c00eb46bbb5c3` verified one charged call and no hypotheses: the model omitted the required action `reason` and again paired local `historical` mode with `provider: "met"`. New plans now use `independent-research-v3-optional-rationale`, recording absent explanations as null without inventing model reasoning. Operational fields remain required, the prompt adds explicit local/remote syntax examples, and invalid provider combinations still return tool errors without network access. Verification passed 134 relevant fixture tests plus JavaScript syntax checking, including a complete fixture research path with missing explanations and preserved rejection/replay behavior. No old run was rewritten or retried, and no inference, human judgment or training was performed by this repair. Valid live search choices and artistic usefulness remain the next evidence gates.

A further read-only audit of `studio-96ce69639da84ca5a537` verified four charged reasoning calls, nine completed network requests, eight metadata candidates, zero acquisitions and zero hypotheses. Qwen rejected the results in its `reason` field, but the validator required separate `note` text. New plans use `independent-research-v4-action-requirements`: notebook/rejection text may remain in either original field, conditional action requirements are exposed in the frozen JSON Schema, and recent search summaries retain mode/provider/limit for interpreting tool errors. The 147 relevant fixture tests passed; the original run and raw response remain preserved. This repair performed no inference, network acquisition, human review or training. Finding relevant sources, inspecting their images and developing useful hypotheses remain unverified live evidence gates.

The [8 September research/source audit](../operations/2026-09-08_research_source_audit.md) adds `independent-research-v5-source-audit`: Met endpoint migration, corrected candidate relevance, richer metadata, bounded prompt excerpts, malformed-record handling and retained HTTP-failure accounting. All 855 software tests passed. Live audit probes verified eligible acquisition and cached reopening across all three providers, while preserving a Met result excluded for lacking an eligible image. The audit made 16 network requests, charged 1,028,760 bytes and recorded six diagnostic acquisitions across separate stores. It made no Qwen/FLUX calls, human judgments or training updates and left the three failed artistic runs unchanged. Real model-directed discovery and artistic usefulness remain unverified; a newly admitted research run is the next evidence gate.

A subsequent user-reported token-admission failure exposed a gap in that audit: 6,811 prompt tokens exceeded the actual 6,392 input allowance after reserving 1,800 output tokens from the 8,192 combined ceiling. The [token-admission follow-up](../operations/2026-09-08_research_source_audit.md#follow-up-token-admission-failure) introduces `independent-research-v6-token-admission`, with exact pinned-processor counting before model loading/reservation, compact JSON, disclosed optional-context projections and separately frozen full/admitted contexts. Processor-only verification reproduced the saved 6,811-token prompt; compact serialization admitted the same content at 5,872 tokens, and a six-image-slot stress check at 6,328. No Qwen/FLUX inference or network request was made, and the failed run was preserved. Final verification passed 861 software tests. This is an admission fix, not evidence of valid future model choices or useful artistic discovery.

## Grounded generation follow-up — 10 September 2026

Run `studio-7549ca07b8524d3789dc` demonstrated that v29 feedback alone did not
prevent repeated source IDs or interpretations attributed to real claim IDs.
New v30 plans constrain these fields during generation: distinct eligible sources,
source-specific citations and literal contiguous properties preserving negation.
Qwen still chooses the source and extractive span and authors the transformation;
this is operator-authored protocol enforcement, not learned artistic judgment.
A processor-only reconstruction of the reported decision fits at 8,344 input tokens
against 14,584 available. See [grounded generation](../operations/2026-09-10_grounded_draft_contract.md#grounded-generation-follow-up).
No original completions were changed or inference retried. Real artistic usefulness
and reliable completion of the larger workflow remain unverified.

## Draft feedback follow-up — 10 September 2026

The user-reported v28 run `studio-001496cc76ec406ead30` retained four rejected
drafts in steps 003–006: repeated `artic_11390` IDs, then an unacquired candidate
and unsupported citations, then repeated IDs again. Hypothesis-level validation
short-circuited the combined feedback and removed it from the next prompt.
New v29 plans diagnose these failures together without accepting duplicates,
supplying source properties or replaying charged work. See the
[follow-up record](../operations/2026-09-10_grounded_draft_contract.md#duplicate-source-follow-up).
Live use of the corrected feedback and artistic usefulness remain unverified.

## JSON generation failure — 9 September 2026

The **10 September 17:27 UTC** attempt `studio-901a68c896cc482b8262` ended
`model_stop` after seven settled Qwen calls, one acquired/inspected reference and
three rejected drafts; no studies were generated. The attribution check correctly
rejected model interpretations presented as source facts, but its generic feedback
hid other missing draft requirements. The [v28 draft-contract correction](../operations/2026-09-10_grounded_draft_contract.md)
exposes those requirements during generation and compares rejected properties with
their actual citations. Original evidence remains unchanged. This is a software
repair; grounded live drafting, artistic usefulness and human assessment remain
unverified. Final processor-enabled verification passed **1,105 tests**, with five
live/archive-dependent tests skipped; this repair performed no inference.

The subsequent v20 run `studio-7f9e54f2551d4f7480a0` failed at its first call by
exhausting 1,800 output tokens inside a repetitive reason. The earlier syntax-only
test gate did not establish live reliability. The [v21 follow-through](../operations/2026-09-09_bounded_json_generation.md)
adds bounded action-schema generation and corrects valid string-boundary token
filtering. The user explicitly authorized fresh monitored attempts; their actual
outcomes, including failures and charges, remain the evidence gate.

Follow-through as of 10 September: v25 retains complete descriptions through smaller
candidate pages; 1,086 processor-enabled software tests passed (five skipped). Five
fresh attempts consumed 38 Qwen calls and produced no studies. The last two supplied
19 consecutive valid responses, but source selection remains unresolved and the
last run reached its wall-time limit during repeated machine sleep. A scheduled
follow-up waits for an awake machine on external power before further bounded work.
The full live artistic-process gate is still unmet; no success rate or artistic
improvement is established.

The next awake-machine attempt ended after seven valid calls but two premature
drafts from unacquired metadata. V26 gates drafting on acquired/opened and inspected
source pixels; 1,087 processor-enabled tests passed with five skipped. Fresh run
`studio-116416c64b4345d589f7` ended after six valid calls when an unscored acquisition
omitted its required relevance_override. V27 compiles this existing conditional
requirement into generation. Seven attempts total 51 Qwen calls and no studies;
the full process remains unverified.

Run `studio-404d3b553ae24c5c8cb9` failed on a missing array bracket in its first
image inspection after five charged Qwen calls, one acquired reference and no
committed hypotheses or generated studies. The completion used 173 of 1,800
output tokens; increasing context did not prevent this syntax error. New v20
plans freeze syntax-only token filtering during generation, retaining strict
parsing and existing budgets. The [incident record](../operations/2026-09-09_json_generation.md)
distinguishes tokenizer/generation-loop tests from unverified live reliability.
The original responses and charges remain preserved. No artistic improvement,
human judgment or weight update is established; the next evidence gate remains
a real admitted run with grounded alternatives, inspected studies and human
assessment of the revision.

## Random exposure before selection — 9 September 2026

The user requested chance encounters during inspiration research without a novelty
reward. New v18 plans support a frozen seed and a separate shelf of up to two randomly
sampled eligible works; new panel runs enable it by default. An empty online pool
permits one random operator breadth query/provider within existing limits. Qwen may
discard every lead; acquisition, image inspection and artistic selection remain
separate. The [implementation record](../operations/2026-09-09_random_exposure.md)
documents sampling bias, replay, failure accounting and software verification.

This changes exposure, not learned taste. It establishes no artistic improvement and
creates no human labels or weight update. The next evidence gate is a full admitted
run with observed use/discard decisions and human assessment of its studies and
selected-parent revision. Randomness may add distraction or superficial connections;
whether the model selects useful surprises remains unverified.

## Context capacity repair — 9 September 2026

The user's subsequent request to increase the limit is implemented for new v19 plans as **16,384 combined Qwen tokens**, leaving **14,584 input tokens** after reserving 1,800 for output. The default panel's 42 calls now allow 688,128 tokens; exact processor admission, candidate paging and capacity checkpoints remain in place. Research, image inspections, development choices and reopened research use the same frozen allowance. The [expanded-context record](../operations/2026-09-09_expanded_context.md) distinguishes software verification from unmeasured model runtime and artistic outcomes. Existing plans and charges retain their original limits.

Run `studio-b70366976547419586e8` stopped before its third decision because 21 search results remained in the prompt after the old compression stages. Its short artistic question was not the cause. The [context-capacity repair](../operations/2026-09-09_context_capacity.md) adds token-counted candidate pages, complete saved-result recall and a capacity checkpoint instead of an uncaught context exception. The failing context now admits at 5,854 input tokens; a current-policy reconstruction admits at 5,930 against the unchanged 6,392-token input limit. The two existing Qwen charges and all saved research remain intact; the repair performed no new inference. The remaining artistic uncertainty is whether the model uses the available evidence to develop consequential alternatives, not whether this result set fits in one prompt.

## Single-action response failure — 9 September 2026

The newest reported run, `studio-ee200a1e1906411ea1a8`, failed after three charged Qwen calls and no images. Its third completion repeated the same incomplete `rewrite_queries` object five times; every copy omitted `queries`. The saved schema contained that requirement, and the completion did not exhaust its token allowance. New v16 plans make the single-object protocol and conditional required fields more explicit. A replay diagnostic also exposed local text search as the default for an omitted mode; new plans without a local index require explicit remote mode. The original responses, charges and failed status are preserved. See the [incident and verification record](../operations/2026-09-09_single_action_contract.md).

This is a prompt and protocol correction, not established improvement in model compliance or artistic inquiry. The prepared replacement retains the operator's question and budgets without dispatching inference. The next gate is a newly admitted real run that reaches grounded alternatives and actual-image inspection; human assessment remains necessary.

## Image-led research and repeated development — 9 September 2026

The user requested a bounded cycle in place of the panel's single follow-up. New v15 research/v5 render plans retain the three-realized-study gate, then allow frozen Qwen to reopen research from an inspected branch, revise or combine any saved branches, reformulate, or pause for human review. Every generated revision is inspected before the next choice. The original operator problem and context remain separate from the model's changing question. New panel limits are displayed and frozen: by default three development actions, one optional eight-action research session, up to 42 Qwen calls and six FLUX attempts. Old frozen requests remain unchanged. See the [development-cycle record](../operations/2026-09-09_development_cycle.md).

This is an orchestration change, not evidence of improved artistic judgment. Initial studies are still planned/rendered as a batch. The next gate is an admitted real cycle showing an image-led change in inquiry or action, followed by human assessment of its selected revision and retained alternatives. Software verification does not create human labels, artistic preference or weight updates.

## Studies before artistic commitment — 9 September 2026

The user's review recognizes the Veranda Post → agency/collective dependence → crown proposal as conceptual exploration, while identifying a premature commitment before discovery through making. The [study-development iteration](../operations/2026-09-09_studies_before_commitment.md) adds three provisional studies with distinct operations and reasons to new panel runs, full composition/instruction handoff, actual-image inspections, and a model-attributed continue/combine/abandon/reformulate/stop decision. One optional follow-up is rendered from the saved selected parent/references (or a fresh reformulation), then inspected. The user's crown examples remain illustrative, not model prescriptions.

These are frozen-model workflow capabilities. Exact duplicate checks do not establish substantive artistic difference, and model observations do not establish human preference. Existing frozen runs are preserved; this implementation launches no new artistic inference and creates no human labels or weight updates. The next gate is an explicitly admitted run demonstrating meaningful alternatives and a direction changed by seeing, followed by human assessment of the selected revision and alternatives. The earlier geometric acceptance gate remains separate from this qualitative development process.

## Grounding repair — 9 September 2026

The [v13 grounded research repair](../operations/2026-09-09_grounded_research.md) supersedes the interpretation of the 16-call v11 execution below as a process acceptance result. That run completed rendering but fails the new source-relevance, grounding and measurable-render criteria. New plans add query decomposition, provider fan-out, model relevance gates, cited properties, image reinspection, draft loop controls and explicit PASS/FAIL reports. A separate one-call caption-withheld Qwen probe confirms visual input access, not reliable interpretation. Extractive entailment and lexical caption screening are conservative software gates with disclosed limits, not learned artistic judgment. Original runs and recovery evidence remain unchanged. A full v13 artistic run and human assessment remain unperformed.

## Latest repair and verification — 9 September 2026

The subsequent [image-prompt capability](../workflows/image_prompts.md) adds optional model-selected reference images to new v12 independent-research/panel runs and v7 bounded-development revisions. Qwen chooses exact inspected source views and their purposes; FLUX receives the saved instruction plus those images in a recorded order. Initial references are distinct from lineage parents; edits retain the actual parent/crop and add separate references. Whole rendering supports two optional references; masked editing supports one. Fixture checks establish input routing, copied-byte integrity, bounded choices and no duplicate dispatch on resume. No real image-conditioned run, human preference or artistic improvement is claimed by this software change. The next gate is separately admitted real feature transfer and parent revision followed by assessment of the actual work. Earlier dated runs below retain their original text-only initial-generation behavior.

The user’s subsequent run `studio-809c0f1646f94714a85f` failed after 13 calls because its final draft omitted `hypothesis.sources`. One hypothesis had already been validly committed, but the panel blocked rendering after the later rejection. This exposes a gap in the prior 940-test release gate. The [v10 correction](../operations/2026-09-09_nested_hypothesis.md) moves invalid nested hypothesis arguments into bounded tool feedback and preserves earlier commitments through rendering with a disclosed partial status after terminal rejection. The live v10 check demonstrated nested-field recovery but then exposed search/candidate context overflow; v11 adds disclosed metadata deduplication/excerpts without changing protected inputs or token caps. Original evidence is unchanged; current verification is recorded in that correction.

The separate v11 run `studio-29fd86d2de4540338c92` completed at **17:43:41 UTC** with 16 Qwen calls, one live acquired reference, one committed hypothesis and one real FLUX image under the original study profile. The expanded combined gate passed **968 tests with zero skips**, including the retained failures and actual live evidence. It now refuses to use an older success to hide failed/partial/in-progress runs with the same implementation. Failed diagnostics and costs remain in the audit. No human labels or training updates were created; review of the saved work and source interpretation remains the next artistic evidence gate.

## Earlier live-process acceptance — 9 September 2026

At **16:05:14 UTC**, `studio-8677ed3036854e3084a2` completed the actual browser-to-worker research/render process under `independent-research-v9-tool-argument-feedback`: 19 real Qwen calls, one live acquired reference, one model-authored committed hypothesis and one real FLUX image at the unchanged 512 × 640 study profile. The fixed synthesis checkpoint closed with one committed direction out of a maximum of two. The [live-process audit](../operations/2026-09-09_live_process_acceptance.md) records the preceding failed and stopped attempts and their costs. It exposed and corrected missing operational arguments terminating at the parse boundary; those typed actions now return bounded tool feedback before side effects. Malformed JSON/types still reject, and old frozen runs are unchanged.

The combined release gate passed **940 tests with zero skipped**, including 21 actual cached-processor cases, the archived incident replay and verification of the completed unmocked process with live dashboard endpoints. This supersedes the earlier software-only completion claim below. The live trace verifies one full current-panel execution, not general reliability or artistic merit. The next evidence gate is assessment of the saved work and source interpretation, then selected-parent revision and follow-up review. This audit created no human votes, advancements or training updates.

## Earlier research acceptance audit — 9 September 2026

The [run acceptance audit](../operations/2026-09-09_research_acceptance.md) found nine failing incident/crash regressions despite the previous 878-test suite passing. New `independent-research-v8-run-acceptance` plans join stagnation to a single bounded synthesis checkpoint and reconstruct terminal outcomes from durable decisions on resume. The full acceptance gate passed **922 tests with zero skipped**, including 18 cached-processor cases, source transport replay, interruption boundaries and panel-to-render execution with scripted generation. The latest 15-call incident reaches the checkpoint under replay; assistant-authored fixture draft/commit responses then exercise rendering. This establishes software reachability, not Qwen's ability to propose an adequate hypothesis. Old runs remain preserved, with no new inference, human judgment or training. The next live and artistic evidence gates remain actual model-generated hypotheses, rendering and human assessment.

## Earlier live research failures — 9 September 2026

The later [audit of `studio-87f61ec22af04bf3978b`](../experiments/20260909_stagnated_after_acquisition.md) verified 15 Qwen calls, one actual acquisition and one saved image supplied on six subsequent decisions, but zero drafts, commitments or FLUX calls. V7 candidate retention, duplicate blocking and `partial` reporting worked. The run stopped after six unproductive actions with nine actions unused: stagnation termination can occur before the independently configured final-three-step synthesis phase. This exposes a gap in the preceding controller repair. The next software gate is a bounded synthesis/replanning checkpoint before stagnation termination; the next live gate remains an actual grounded hypothesis, rendering and artistic assessment. The run is unchanged and this audit performs no inference.

A [read-only audit of `studio-5ffc9255fbe94d5ab668`](../experiments/20260909_research_stall.md) verified 24 completed Qwen calls (109,109 actual combined tokens), 20 searches, four notes, 29 completed network requests, zero acquisitions, zero hypotheses and zero FLUX calls. The run exhausted its step cap after approximately eight minutes. Qwen overlooked a returned Sojourner Truth reference, retained unsupported archive-wide conclusions in its notes and repeated identical searches. The context retains only the latest search results while retaining model notes; the controller has no enforced stagnation transition. The panel marks the empty terminal outcome `completed` and leaves its manifest call count stale. Token admission succeeded; no artistic output was available to assess. This audit changed documentation only and preserved all frozen evidence. The immediate gate for new plans is candidate retention, bounded stagnation handling and accurate completion reporting, followed by separately admitted real acquisition, inspection, hypothesis commitment and rendering. Earlier dated assessments below retain their original scope.

## Where we are

The subsequent [9 September progress repair](../operations/2026-09-09_research_progress.md) makes new panel runs online-only and introduces versioned candidate retention, repeat/stagnation controls, protected synthesis steps, early FLUX prompt token checks and accurate empty-outcome reporting. The full suite passed 877 fixture tests, followed by 27 targeted checks after final cleanup. Real cached processors admitted the 24 saved-context projections and a six-image/four-reference synthesis diagnostic without inference. The failed artistic run remains preserved. Actual model-selected inspection, hypothesis construction and rendered outcomes remain the next live evidence gate.

**Kusama is a working, instrumented, frozen-model artistic experiment system. It is at an early research stage, before demonstrated preference learning or a demonstrated advantage over strong artistic workflows.**

The first renderer experiment, controlled editing, a frozen visual director, reference-conditioned proposals, initial painting studies, durable records and human review are real. The newer source-connected studio adds bounded official collection ingestion, audited whole/detail retrieval, source-relationship proposals, an explicit selected-parent bridge, realized branches, decision-time experiment memory and gated dataset exports. These provide a foundation for testing the larger ambition. The central research loop—use critique and past outcomes to make better decisions, then prove that improvement on unfamiliar work—has not yet been demonstrated.

There is no defensible percentage of completion toward “superhuman artistic abilities.” The infrastructure is substantially further along than the artistic evidence. The source's first vertical slice has been executed; its E0 comparative experiment has not been completed; H1–H4 remain untested. Calling this a capable experimental prototype is justified. Calling it a self-improving artist is premature.

## Final overnight evidence

The inspected source-to-review-queue path produced 16 studio images at 512×640, including local and finishing variants. Three E0 development pools are complete; the fourth has a saved parent and a prepared but unexecuted technical repair. In total 38 new renderer outputs and 46 Qwen calls were recorded, using 81,834 reasoning tokens with no unsettled calls. All eight provisional heldout briefs and all matched workflow controls remain unexecuted. The final suite passed 706 software tests; this is not evidence of artistic preference.

The 13:48 review backup contains 9 collections, zero human quality/potential votes, zero pool choices and zero advancements. Seven fresh dataset snapshots have zero eligible rows and no model was trained. Original-brief/descendant grouping is implemented, with conservative quarantine of unresolved or unsupported recovery ancestry; broad source-family-transfer and differently worded brief lineage still require explicit provenance.

The source relationship sometimes survives as a visible configuration, but requested local effects were absent or incomplete, masks could exclude necessary objects, and model observations sometimes overstated success. No best artwork or learned advantage has been established. The next gate is a genuine human comparison and explicit parent choice followed by a controlled revision and follow-up review. Further inference requires an explicit follow-up resource contract and audited migration from the expired session.

## Phase 1 offline evaluation — 8 September 2026

At **2026-09-08T15:10:50.685902+00:00**, the [Phase 1 handoff](../operations/2026-09-08_phase1_handoff.md) adds a general evaluation CLI, a frozen 24-brief development suite, versioned metrics, brief-level reports and two-stage blinded review. Offline execution scored 12 real saved E0 revisions across three completed development briefs, retaining two missing pairs from the fourth brief; a separate 24-image fixture demonstration tests the harness. Both new annotation stores contain zero genuine judgments. No new Qwen/FLUX call or weight update was made by this milestone. Pinned CLIP descriptors and reference neighbors describe these saved outputs; they do not establish preference or artistic improvement.

The prospective compact Kusama-search versus source-matched best-of-four plans remain unexecuted because the frozen resource session has expired. The cached ordinary-critique versus inquiry-relationship diagnostic retains its original conditions and E0 render profile; it is not relabeled as the new benchmark. Engineering verification passed 725 tests in the shared full suite and 17 final focused Phase 1 tests. The remaining evidence gate is actual human review, explicit parent choice, a controlled revision and follow-up review, plus an independently authorized prospective comparison.

## Additional content diagnostic — 8 September 2026

A separately authorized [content capability diagnostic](../experiments/content_probe_20260908.md) completed four Qwen calls and eight FLUX initial generations under `flux_base4b_study_512_v1`, using a new bounded resource session. These are additional outputs, separate from the overnight totals above. No old job was resumed, no filter setting was changed and neither model was trained.

The assistant-authored suite compared direct prompts with Qwen-written prompts for a clothed control, its nonsexual adult nude counterpart, political criticism and religious criticism, using one seed per route. Qwen returned all four prompts without observed explicit refusal. Both nude outputs depict adult nudity without added covering; both routes also realize the supplied critical scenes. Assistant visual inspection found cropped lower legs/feet in the Qwen figure outputs, incorrect or absent mouth tape in the political images, and uneven painting/transaction detail. The saved images demonstrate this narrow content capability without training; they do not establish unrestricted generation, broad reliability, reference-image understanding, human artistic preference or learned advantage. All judgments remain explicitly assistant-attributed, with no human quality/potential votes or advancements written. The run's 109-check integrity audit passed with no unsettled invocations. Human inspection and any separately budgeted replication are the next evidence gates for this diagnostic; the broader selected-parent human-review gate remains unchanged.

## Bounded visual development software — 8 September 2026

The [bounded-development workflow](../workflows/bounded_development.md) implements the technical sequence requested with human judgment set aside: an edit-and-observe benchmark, stronger continuation controls, autonomous proposal/observation/selection, simpler workflow comparisons and evidence-based next-experiment reporting. Continuation now exposes explicit native-pixel crops and supplied geometric controls. New requests require full declared target/support coverage, respect external protected regions and retain uncertain target-existence evidence. Linked protected intentions need an attributed revisit reason before removal or change. Old frozen policies and inputs remain unchanged; native crops do not establish high-resolution upscaling or a quality/throughput gain.

The bounded loop lets frozen Qwen propose an intention map and intervention, observe actual same-parent seed alternatives, then select a child, retain the parent, recover or stop. The [common-parent comparison](../operations/2026-09-08_dev_panel.md) gives best-of-N, ordinary critique/edit and bounded development the same whole/masked/crop controls, source access, external requirements, seeds and named profile. Actual costs and differing decision opportunities remain disclosed. A separate external-memory ablation reuses the primary memory-off result; it is a correlated comparison, not another independent control run. Memory usefulness and a workflow advantage remain unestablished.

The [offline saved-image diagnostic](../../artifacts/edit_benchmark/20260908_saved_failures_final/report.md) evaluates **five saved children across two briefs**: two geometric resize results pass the current width/height replay, one masked resize fails, and two semantic connection outcomes remain uncertain. All five preserve their declared protected pixels. Both connection plans fail the new full-support scope check. These are measurements of existing artifacts and cached detector evidence, not new render results or artistic judgments. The stricter-versus-overlap rule comparison is explicitly counterfactual on familiar failures.

This software and offline-diagnostic milestone made no new Qwen/FLUX inference calls, wrote no human labels and trained no model. Its fixture checks establish execution and evidence boundaries. The next technical evidence gate is fresh, separately budgeted development execution with appropriate independent measurements, followed by the controlled workflow comparison; no artistic or learned advantage is established by the implementation. Earlier dated inference and review counts remain unchanged.

A separately bounded live follow-up finished under a new explicit contract; its [8 September development handoff](../operations/2026-09-08_bounded_development_handoff.md) preserves all versioned failures. The eight-condition diagnostic consumed 10 Qwen calls and two renders: seven conditions stopped before rendering, while one completed proposal/edit/observation/selection and retained the original parent after Qwen saw no connecting bar. Independent semantic fulfillment remains uncertain. Across the entire follow-up session, 20 Qwen calls, 65,473 tokens and four renders are settled; those renders contain only two unique child hashes. No figure-resize children were produced. The pre-run v5 code passed 836 software tests; the post-run fixes and concurrent additions present in an isolated shared-source snapshot passed 951 tests with matching TypeScript contracts. Concurrent research-task source edits caused one observed native-plan mismatch and prevent proof of the loaded source identity for later conditions. This is not a valid matched comparison and establishes no workflow or memory advantage. The next technical gate is valid executable proposals and independently measured edits under an isolated frozen source snapshot. These counts are separate from the expired overnight totals.

## Independent artistic research software — 8 September 2026

The [independent research workflow](../workflows/independent_research.md) implements the requested breadth and discovery controls. Qwen can choose searches, open whole/detail images and contextual passages, follow admitted links, reject leads, revise the inquiry, recall its notebook and select a variable reference set. A draft reopens the decisive evidence before a later commitment. Source claims, visual observations, proposed analogies and operator search policies remain distinct. Initial problem framing is still operator-supplied.

The source backend supports a standing metadata catalog, frozen per-episode catalog imports, search-driven acquisition from three official providers, existing indexed sources and configured contextual texts. Search modes expose lexical, contextual/historical, frozen-CLIP, typed-relationship, exploratory and counter-question routes with their actual limitations. Coverage policy targets are operator choices, not achieved archive completeness. Historical/contextual matching is currently lexical, and embedding distance is not an artistic-quality measure.

Committed hypotheses can be explicitly rendered through frozen FLUX and registered for manual review. The optional development loop can undertake fresh bounded research from its actual parent and preceding observations, then preserve adopted visual and textual evidence in a revision. The dashboard exposes questions, inspected sources, rejected leads, hypothesis contributions, calls and provenance. The older automatically paired workflow remains available as a baseline.

This implementation used offline fixtures and controlled model substitutes: **no live Qwen/FLUX call, external collection acquisition, human judgment or training was performed by this milestone**. Current verification is recorded in [implementation status](../operations/implementation_status.md). The next evidence gate is a bounded run from a substantial artistic problem, several realized hypotheses, human inspection, and development of an explicitly chosen saved parent. A trained evaluator is not a prerequisite. Independent discovery here describes decisions in the recorded tool loop; it does not establish originality relative to unknown pretraining, compelling art or an advantage over strong simpler workflows.

## What the ambition means

The source's distinctive ambition is to find and transform aesthetically and intellectually consequential reference relationships beyond human capability **on defined tasks**, then develop stronger work and sustained practices. It is broader than imitating an artist's appearance or making attractive images.

The potential advantage is a growing record of which connections, interventions and critiques actually helped: what was tried, what changed, what different viewers valued, what deserved further development, and whether those lessons transferred. More source images, model parameters or polished explanations cannot establish that advantage by themselves.

A defensible future superiority claim would identify the task, human comparison group, references/tools available, time and resource limits, presentation conditions, independent reviewers, held-out projects and uncertainty. It would establish a bounded result under that protocol. It would not establish universal beauty, historical importance or superiority to every artist.

**We are on a plausible path to testing this ambition, not on an already validated path to achieving it.** Staying on that path means making the next artistic comparison informative. Frozen models can improve as a system through better tools, context, retrieval and search; weight training is a separate possible improvement, not the definition of all progress.

## What the system does today

### Starting point for a full artistic run

Direction clarification, **8 September 2026**: a full artistic run starts with a sufficiently complex artistic problem statement, before proposing image directions. It must support consequential stakes, unresolved tensions, grounded inquiry and meaningfully different artistic responses. The [problem-statement guide](artistic_problem_statements.md) defines this planning criterion and preserves the user's primary Wangechi Mutu case-study question about reconstructing the Black female body as a source of power, self-definition and new possibilities for existence.

Operators supply and assess the problem today. A future model should adequately construct such statements; this is a separate upstream research goal, not an existing capability or training result. Existing brief/inquiry contracts do not validate conceptual adequacy. Historical technical studies retain their original scope, and this clarification makes no inference calls or changes to frozen runs. The immediate human-reviewed continuation gate remains in place.

### Existing execution paths

The earlier curated-reference workflow remains supported:

1. An operator supplies a brief, curated source records and local reference images. The runner validates and snapshots them.
2. Frozen Qwen sees the images and context and proposes concrete pictorial directions, source relationships, predicted effects and FLUX prompts.
3. Frozen FLUX receives each saved **text prompt** and makes two initial images using different seeds. These have no parent image. In this workflow, the source images are inputs to Qwen, not direct image conditioning for FLUX.
4. An optional frozen Qwen pass observes each sibling pair and predicts finished-quality and development-potential preferences, with a possible revision. These are model predictions.
5. Humans can compare saved images in the dashboard and explicitly choose a candidate to develop. Choices are recorded; they do not execute another image call.

The newer [source-connected studio](../workflows/source_connected_studio.md) adds a second executable path. Bounded official APIs supply rights-tagged source records and byte-preserved images. Frozen CLIP, lexical search and unvalidated typed relationships assemble a whole/detail shelf. Qwen sees selected source images and proposes source roles, transformations, predicted effects, whole/part relationships and a standalone rendering instruction. FLUX realizes saved instructions; a versioned archive retains descriptor coverage and protected alternatives. With no validated quality model or real human preference labels, selection uses an explicitly attributed seeded exploratory policy.

The [continuation bridge](../workflows/continuation.md) now carries an exact saved parent, inquiry, references, proposed revision, preservation constraints, two seeds and a bounded budget into the persistent studio. It accepts either an actual human advancement submission or a separately attributed operator/model selection. Qwen can propose a local intervention after seeing the actual parent. The bridge supports whole-parent edits, masked edits, explicit native-pixel crops and supplied geometric transforms with background repair; preserved outside pixels do not guarantee successful local semantics or edges. Actual children receive lineage, observations and statement-free follow-up review collections. Preparing a continuation or voting does not execute it; execution remains an explicit action.

The earlier 8 September intention-map change added optional parent-bound maps, preflight rejection of edits overlapping pixel-protected elements, linked snapshots with reasons for reopening resolved decisions, matching whole/detail gallery views, and attributed selection/stop records with separate local, whole and preservation assessments. That first milestone required explicitly supplied maps. The later bounded-development software now accepts autonomous Qwen map proposals through the same validated contract. Neither change establishes a learned attention policy, semantic segmentation or high-resolution upscaling. Fixture validation does not alter the dated inference/review counts below; actual desired-change measurements and the broader human-reviewed artistic-development gate remain separate.

Optional [experiment memory](../data/experiment_memory.md) retrieves attributed preceding actions, actual artifacts, observations and failures. Each new development decision freezes its eligible candidate set, cutoff and at most four admitted events; Qwen receives a bounded text excerpt and is told that the prior images were not shown. Resume reuses that exact snapshot. Sealed episodes are excluded from development memory; the heldout API permits only its own preceding non-review state. Memory usefulness remains unvalidated.

The overnight profiles are explicit and separate: `flux_base4b_1024_v1` at 1024 × 1024 and `flux_base4b_study_512_v1` at 512 × 640. Both declare 50 steps, guidance 4.0, BF16, model offload and VAE tiling. The E0-only alternative `flux_base4b_e0_384x512_v1` explicitly declares 384 × 512 with those settings; its smaller dimensions do not by themselves establish throughput or quality. Profile identity and actual execution remain in frozen inputs. See [session profiles](../../configs/overnight.yaml) and [E0 profile rules](../operations/2026-09-08_dev_panel.md).

Today we are primarily **testing the tools and collecting the ingredients of an artistic decision dataset**. The reviewed parent → revision → follow-up pathway exists in software, but no actual human advancement has exercised that evidence gate in the checked review stores. Saving feedback, changing a prompt and retrieving a history are different from training a selector or showing that experience improves unfamiliar work.

## Capability map

| Area | What exists | What remains unproven or absent |
| --- | --- | --- |
| Image execution | Real FLUX generation/editing; deterministic geometry and local repair; explicit continuation crop/geometry controls; saved hashes and lineage | Reliable semantic control for arbitrary edits; high-resolution upscaling; general improvement in artistic quality |
| Direction | Real frozen Qwen multimodal proposals; bounded autonomous map/proposal/observation/selection software with saved inputs and responses | Validated advantage of autonomous development; a learned action policy; independently discovered diversity in the scaffolded V3 run |
| References | Curated packets plus 200 audited official records, geometric crops, whole/detail lexical/CLIP retrieval and 54 typed machine relationships; new bounded catalog/acquisition and contextual-source software | Comprehensive cultural discovery, verified semantic localization, human retrieval usefulness and learned connection selection |
| Inquiry and development | Versioned inquiry, model-directed research with variable reference sets, explicit human/operator/model parent selection, realized branches and follow-up review registration | A genuinely reviewed continuation showing improvement; independent research producing stronger art; sustained developing practice |
| Observation | Frozen Qwen sibling predictions and actual parent/child visible-change observations, separately attributed | Reliable valid responses in every case; validated visual accuracy or independent artistic judgment |
| Human feedback | Balanced A/B queues, quality/potential and pool/none choices, readiness/context/exposure fields, explicit advancement and follow-up collections | Actual labels in the checked stores; validated review protocol and practice critique panels |
| Resources and recovery | Shared invocation reservations, protected exploratory allocations, immutable inputs, no-call recovery sidecars and cost journals | Benefit of the allocation policy; complete economic totals without sourced rates |
| Experiment memory | Append-only event projection, decision-time cutoff checks, candidate snapshots and optional bounded prompt integration | Demonstrated artistic benefit from memory; complete lineage/family resolution |
| Learning | Grouped immutable dataset exports, sealed-input guards, cheap-selector implementation | Adequate genuine human data, locally verified multimodal training, research adapters, DPO and evaluated learned releases |
| Comparative research | Historical diagnostic schedule, E0 planner/runner, saved edit benchmark, common-parent workflows with equal control access, separate external-memory ablation and explicit profile/resource gates | Prospective technical workflow advantage; completed human-reviewed matched comparisons, E0 outcome, human benchmark results, H1–H4 and transfer studies |

The earlier entry points remain [creative.py](../../src/kusama/creative.py), [director.py](../../src/kusama/director.py), [backend.py](../../src/kusama/backend.py), [references.py](../../src/kusama/references.py), [studio.py](../../src/kusama/studio.py), [reviews.py](../../src/kusama/reviews.py), [execution.py](../../src/kusama/execution.py), [ledger.py](../../src/kusama/ledger.py) and [costs.py](../../src/kusama/costs.py). The new path uses [corpus.py](../../src/kusama/corpus.py), [retrieval.py](../../src/kusama/retrieval.py), [inspiration.py](../../src/kusama/inspiration.py), [episodes.py](../../src/kusama/episodes.py), [continuation.py](../../src/kusama/continuation.py), [memory.py](../../src/kusama/memory.py), [datasets.py](../../src/kusama/datasets.py) and [learning.py](../../src/kusama/learning.py). These extend the existing package and ledgers rather than reproducing the specification's proposed folder tree.

## Evidence we can claim

### Earlier interventions

- **Original mountain experiment:** one parent and two independent whole-image edits executed. Neither edit fulfilled the full one-third-size intervention and preservation request. This established real inference and exposed a control weakness. See [execution evidence](../experiments/verification.md) and [visual review](../experiments/visual_review.md).
- **Qwen-directed mountain edit:** one real Qwen proposal and one FLUX edit completed. The human response about greater vastness was uncertain. This established the handoff and a useful negative/uncertain result, not an artistic preference win. See [director verification](../experiments/director_verification.md).
- **Controlled scale experiment:** code scaled the original figure, while FLUX repaired the exposed background. Measured height ratio was 0.3365 and changed protected pixels were zero. The user confirmed a change in the feeling of vastness. The question did not establish an explicit preference or finished-quality improvement. The compositional intervention was supplied by the operator; Qwen formulated it. See [scale experiment](../experiments/scale_experiment.md).

### Maya memory V3 snapshot

Local run: [`20260908T022716_creative_ee890a8f`](../../artifacts/20260908T022716_creative_ee890a8f/manifest.json). The manifest was last updated at `2026-09-08T04:17:11.653718+00:00` in the completed diagnostic pass.

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
| Final run verification | [Technical integrity report](../../artifacts/20260908T022716_creative_ee890a8f/verification.json) passed; it explicitly reports 7 valid/5 unresolved observer pairs, 43 usable outputs and 7 retained failed attempts |

The counts above describe that completed historical diagnostic, whose bounded recovery was closed. They do not describe active later jobs. Consult current manifests before resuming or reporting later progress. Partial valid observer coverage does not prevent humans from reviewing saved images; model judgments remain unreliable and must not substitute for human feedback.

The [24-image contact sheet](../../artifacts/20260908T022716_creative_ee890a8f/contact-sheet-24.png) shows different compositions, including figures, surfaces, landscapes and abstraction. My contact-sheet inspection is an assistant observation, not an independent human quality assessment. The visible breadth is partly explained by [operator composition scaffolding](../../references/maya_memory/diversity_scaffold.json); it cannot be credited entirely to Qwen's autonomous invention. The preceding V2 run retained six closely related directions before rejecting a duplicate title, with zero paintings generated.

The current 108 comparisons × 2 questions = **216 planned dimension responses**, not 216 collected judgments, not 216 independent trials, and not the source specification's E0. E0 proposes 12 briefs, 84 images, three reviewers, 216 parent/edit judgments and 36 pool selections. The current single-brief generation study does not satisfy those conditions. Historical human feedback about vastness exists separately; zero current quality/potential votes does not mean the project has never received any human feedback.

### Source-connected studio and recovery snapshot

The [05:56 UTC source audit](../../artifacts/overnight/source_audit_final/summary.json) verifies 200 records from four official providers, 400 geometric crops, 502 documented statements, 43 machine interpretation claims and 40 unvalidated device seeds. The frozen final index contains 199 whole views and 398 crops after excluding one faint specimen and its descendants. Every admitted view passed the recorded rights/hash checks, and all 54 typed relationships exactly matched their saved backing claims. A CPU-only 12-query replay reproduced the same source IDs and ordering. These checks establish reproducibility and grounding, not cultural completeness or human usefulness. [Coverage gaps](../data/source_audit.md) include imprecise dates, unresolved place/culture fields, sparse distant relationships and unestablished contemporary-art representation.

The `care_control_v2` run has six source-connected initial FLUX images under the named 512 × 640 study profile. Its [06:19 UTC saved-response replay](../../artifacts/overnight/recovery_audits/care_control_v2_no_call_replay.json) retained all six connections and image hashes, preserved the renderer's generation projection, and left the completed observation job unchanged. Source proposal accounting stayed at seven charged Qwen invocations. The replay used normal locking with explicit model-load, dispatch and reservation traps: it made zero new calls and loaded no model. This is a recovery check, not an artistic evaluation or a total count for all later branch work.

One failed flat-wire response was recovered through an explicit coordinator-metadata sidecar. It supplies only absent request ID, neutral title, fixed external brief and null source influence levels. Replaying the sidecar recomputes the result from exact original raw bytes; it cannot supply a missing rendering instruction or change a transformation. The original failed response, repair call and costs remain recorded. The later `connection-flat-wire-v3-medium` system prompt explicitly requests a two-dimensional collage or painting; this operator-authored change applies to future runs and does not retroactively change the v2 requests or demonstrate better outcomes.

The older V3 diversity challenges, the current source-pairing rule, archive niches, protected allocations and seeded fallback selections are operator scaffolding. Qwen's artistic proposals and parent-conditioned local revisions remain model-authored; assistant observations and geometric checks remain separately attributed. The selected-parent bridge has executable operator/model paths and a contract for real advancement IDs. It has not yet supplied evidence of improvement from actual human choice.

At **06:21:25 UTC**, read-only checks found **zero actual pairwise votes and zero advancements** in both the historical and overnight review databases; the overnight pool-vote table also held **zero selections**. No human quality/development labels were created by preparation, generation, observation, recovery or this assessment. Historical comments about vastness remain legitimate separate human observations. No trained research selector/adapter, learned artistic advantage or completed human benchmark victory is established. [Training and eligibility status](../models/model_and_training_status.md) records the historical validators and the subsequent removal of multimodal training code.

### Historical software and resource evidence — 04:17 UTC

The suite at the **04:17 UTC historical verification** passed **402 tests**, with two dependency deprecation warnings, in 15.40 seconds (`.venv/bin/python -m pytest -q`). This is not the current repository's test total. Those tests exercised software contracts using fixtures and controlled substitutes; they did not evaluate the artistic merit of the paintings. Later implementation and test evidence is recorded separately.

The V3 primary/recovery cost records total approximately **80.70 minutes of recorded operation runtime**, including model loads and failed work. This is not an all-project total or complete human-time measurement. Local API fees are recorded as zero; electricity and hardware totals remain unknown without rates. See [cost accounting](../operations/cost_accounting.md).

Generated evidence and SQLite databases are local and ignored by Git. Their links will not work in a fresh checkout without a separately preserved artifact root. The original documentation audit ran no model inference and added no reviews. The historical 04:17 update incorporated its separately executed final observer pass and [cost snapshot](../../artifacts/20260908T022716_creative_ee890a8f/cost-summary.json), with no new human votes or advancements. Its checkout then had no Git commits; that is a historical provenance note, not a statement about the current repository. The current capability assessment and saved-response replay also made no inference calls or human labels.

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

Exercise the implemented cheap selector against frozen baselines once sufficient varied human feedback exists. Then test a reviewer-conditioned judge; train a director only from reviewed actions and development outcomes. Judge/director training has no current executor; a future implementation would require an actual supported-device check of multimodal masks, intended gradients, frozen parameters and save/reload before a research run. Local inference fit does not establish training fit.

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

Read [AGENTS.md](../../AGENTS.md) for coding-agent working rules. Keep commands in the workflow guides: [source-connected studio](../workflows/source_connected_studio.md), [continuation](../workflows/continuation.md), [experiment memory](../data/experiment_memory.md), [reference loop](../workflows/reference_loop.md), [human review](../evaluation/human_review.md), [studio](../workflows/studio.md), [controlled editing](../workflows/controlled_editing.md) and [recovery](../operations/recovery.md).

When evidence changes, update the snapshot timestamp, run state, actual review counts, demonstrated outcome and next gate together. Preserve older experimental records. Cite actual artifacts or explicit human responses; do not promote an ambition, parser success, model prediction or planned budget into a measured result.
