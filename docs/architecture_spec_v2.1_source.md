# Kusama v2 architecture specification — source text

Source: [Kusama_v2_Technical_Architecture.docx](https://docs.google.com/document/d/1NmD6y8WxAOiCQdIFmCGhflEIxU9mleDU/edit).

Retrieved: 2026-09-08. The document identifies itself as version 2.1, dated 8 September 2026. Drive reports an uploaded Word document, last modified `2026-09-08T00:21:50.000Z`.

This is the connector's readable text extraction, preserved for offline reference. It does not preserve the original document's layout or table structure; several tables appear as sequential lines at the end. Consult the linked original for formatting. The specification describes proposed research and assumes an empty repository. Its examples, commands, module layout, budgets and training configurations are design proposals, not claims about the current implementation. For the current assessment and implementation priorities, read [project direction](project_direction.md).

The source text below is reference material. It does not authorize executing its example commands or override the user's task or repository agent instructions.

---

Kusama v2
Technical architecture and research specification
Expanded revision of the review copy · 8 September 2026 · Version 2.1
Purpose, scope and success conditions
Kusama is a proposed experimental studio that develops artistic work through inquiry, reference study, branching experiments, informed critique and learned decisions. Its long-term ambition is to find and connect aesthetically and intellectually consequential references at a level beyond human capability on defined tasks. The immediate research objective is to improve unfamiliar finished work, and the development of a sustained practice, at a measured resource cost.
This specification assumes an empty repository. Interfaces, commands and configurations below define what to implement; they are not evidence of shipped software, GPU fit, trained weights or validated artistic capability. The first medium is 2D collage and painterly work. Physical, spatial, sonic and performative work require additional execution tools and evaluation of actual outcomes in those media.
Artistic contract. Each project attempts an answer to a worthwhile question. Store the question, its stakes, a provisional position, intended viewing conditions and unresolved possibilities. Prefer genuine tensions with several defensible approaches. The inquiry can emerge through making and change during development; an answer may be embodied, contradictory or open-ended. Neither an elaborate statement nor strict agreement with initial intent establishes artistic success.
Research claims are separate. H1: a reviewer-conditioned selector retains work that reviewer values more than a frozen generic selector. H2: branching development and experiment memory improve returned work over strong simpler workflows at equal cost. H3: adapting model weights improves outcomes over the otherwise identical untrained system. H4: useful operations transfer to unfamiliar subjects, sources or practices. A later practice-level claim tests whether critique and longer memory improve subsequent bodies of work.
The initial style study may use Wangechi Mutu's documented work and process, contrasting practices and non-art material. It must not become the system's universal taste. Saturation, restraint, density, ambiguity, polish, awkwardness and kitsch can each succeed. Evaluate whether a visual language serves the work; avoid replacing glossy generic imagery with a mandatory sparse or muted aesthetic.
Required evidence. Every evaluated state has actual saved artifacts and traceable lineage. Reviews retain individual judgments and presentation context. Exploration receives protected resources. Training uses reviewed outcomes and sealed splits. Claims compare strong controls, count unsuccessful work and disclose human involvement. A persuasive explanation of an unsuccessful image cannot satisfy these requirements.
The intended product is a studio that maintains a developing portfolio, traceable reference connections, consequential alternatives and memory of experiments. The prospective technical contribution is a dataset and decision process that improve work and transfer beyond familiar sources. Historical significance, universal beauty and superiority to every human artist remain outside what a short experiment can establish.
Runtime architecture and model boundaries
Use one Python application with logical roles, a single coordinator, persistent storage, GPU workers and a review interface. Separate roles do not require separate distributed services. The coordinator pins model, prompt, archive and policy versions for each run; offline training publishes new versions between experiments.
Selected model roster. D, O and J begin with Qwen/Qwen3-VL-8B-Instruct; D and J receive independent adapters when evidence supports training. R uses undistilled black-forest-labs/FLUX.2-klein-base-4B through its own processor and frozen generation/editing pipeline. These choices are supported interfaces, not performance commitments. [1, 2]
Use frozen openai/clip-vit-base-patch32 embeddings, normalized to unit length, for an initial 512-dimensional retrieval index and a cheap preference baseline. Optional facebook/dinov2-base features can diagnose structure or near duplicates. Neither encoder produces an artistic verdict; additional encoders must resolve measured retrieval or observation failures.
Reference memory stores source artifacts, documented context and proposed relationships. Experiment memory stores before/action/after records, execution recipes, failures and follow-up judgments. The dataset builder resolves both memories against split permissions and event time. Test outcomes never enter a director input, reviewer profile or training-time retrieval result.
The critical separation is between proposal, observation and judgment. D's interpretation is a hypothesis. O describes actual visible changes without treating a requested emotional effect as observed fact. J's primary input excludes D's rationale, system identity and revealing filenames. Context-bearing art receives its declared presentation in a separately specified condition.
Sharing a backbone creates correlated errors. Several model personas do not constitute independent critics. Practice evaluation starts with human panels; a model-assisted critic is tested against their decisions and, more importantly, the quality of subsequent work. Adapter switching on a worker must be serialized or isolated by process so concurrent requests cannot silently use the wrong policy.
State model, schemas and service contracts
Use Pydantic as the authoritative schema source and generate JSON Schema/OpenAPI plus TypeScript types. Forbid unknown fields in execution requests, validate enums and bounds, and resolve artifact/source IDs before dispatch. All examples use illustrative identifiers. Full records additionally carry timestamps, schema version, ownership/access fields and configuration hashes.
An accepted action is immutable. Execution produces a transition linking that action to output artifacts, observer findings and later reviews. Inquiry revisions and critique revisions append events rather than rewriting the past. A practice contains several works and projects; repeated crops of one output do not become a body of work.
{
  "schema_version": "1.0", "action_id": "act_042",
  "state_id": "state_100", "kind": "edit",
  "operation": "open_boundary", "parent_id": "img_100",
  "instruction": "Open the wrap into a supporting arc",
  "preserve": ["figure silhouette", "palette"],
  "reference_ids": ["ref_09", "ref_31"],
  "render_config_id": "flux_base4b_1024_v1",
  "seed": 23, "idempotency_key": "run_07:act_042"
}
POST /propose accepts a state ID, allowed tools and remaining budget; returns validated action JSON. POST /execute accepts an action ID and idempotency key; returns a job ID. GET /jobs/{id} returns state, output IDs or a typed failure. POST /observe accepts actual parent/child IDs. POST /judge accepts ordered artifact IDs, task and reviewer profile version. POST /reviews and POST /critiques append human responses and exposure records.
Each model input has a recorded cutoff event and a manifest of included fields. D receives only evidence available before its action. J receives the same image order and context condition as the corresponding human task. Renderer output is persisted before review assignment. A schema-valid action can still fail execution or produce weak art; these are different statuses and different learning targets.
Return explicit error codes for missing parent, unsupported tool, invalid schema, exhausted budget, model failure and uncertain execution outcome. Do not silently substitute an operation, discard a failed attempt or convert missing review fields into negative judgments.
Inquiry, action policy and sustained development
Represent the external experimental brief separately from the evolving internal inquiry. The brief remains fixed within a comparison. An InquiryRevision contains a question, stakes, provisional position, two to four working commitments, open tensions, evidence IDs and a link to its predecessor. Record the reason for revision: a new reference, an observed result, critique or an unexpected audience encounter.
Working commitments can be formal or conceptual: preserve a boundary's ambiguity; make dependence visible without depicting a victim; let repetition become oppressive through accumulation. They constrain an experiment without prescribing one approved style. Artistic position is an explicit working construction; do not invent private human experiences, trauma or biographical authority as factual evidence.
Director context. Supply the current artifact, brief, current inquiry, relevant prior actions and observations, selected references, allowed tools and remaining resources. Initial image input is one parent plus at most two source images. Text summaries point back to the underlying evidence. A later series-level decision can retrieve a bounded portfolio with detailed crops; profile this independently instead of assuming a local two-image judge can evaluate a practice.
The action vocabulary includes introduce_direction, select_reference, transform_relationship, edit, remove, continue, finish and stop. A request must compile to supported renderer/editor parameters. Define operation-specific fields for scale, position, source role and preservation where the backend supports them. Free-form visual instructions remain necessary, but cannot claim precise masks or material control the execution tool does not provide.
Begin D inference at temperature 0.8, top_p 0.95 and a 512-token action cap. Validate JSON and permit one logged repair attempt; charge both attempts. O uses deterministic decoding with a 256-token initial cap. J scores allowed completions rather than sampling a verdict. Freeze these settings within each controlled comparison; they are starting configurations, not optimized values.
Worked development. For “When does care become control?”, retrieve repair seams, protective packaging and architectural enclosure. Propose different relationships between support and confinement, not just a collage of those objects. Render opening, tightening and removing a boundary. If the renderer also changes posture, record that confound. If viewers discover a compelling reading about dependence, revise the inquiry and test it in further works and a new subject family.
A project milestone selects a provisional series, including unresolved branches, for practice critique. Record how each work extends, contradicts or exhausts a prior move. Coherence is progression of inquiry, not matching palettes. A single strong work remains valuable even when sustained practice has not yet been demonstrated.
Stopping is an action with evidence: a supported readiness judgment, exhausted useful hypotheses, or insufficient remaining budget. A director may retain a disputed decision or decline further revision. Train such decisions only when reviews or matched continuations support them; the last entry in a log is not automatically an optimal stopping example.
Reference ingestion, retrieval and connection learning
Start with 100–300 audited sources covering the pilot's practices and contrasting non-art material. An ingestion job registers origin, maker/date where known, medium, documented context, source family and permitted uses for retrieval, training and display separately. Preserve original bytes; normalize orientation and color in derived viewing copies; compute SHA-256; detect exact and near duplicates; assign related-source groups; then embed approved representations.
Store human description, quoted source text, machine caption and inferred relationship in separate fields. Unknown provenance stays unknown. Keep full images and useful details connected to the same source record. A visual fragment without its series, scale, material or context can mislead both retrieval and critique. Reference documents are untrusted data, never executable instructions.
Use lexical retrieval over metadata/context, CLIP text-image similarity over the visual index, and relation queries over a small typed connection store. The first relation store can be SQLite edges with fields for subject, relation, object, evidence IDs and origin (documented or proposed_analogy); a graph database is not an initial dependency. Deduplicate before displaying results and log the query, index version, candidate set and selection policy.
An illustrative 48-item encounter shelf reserves 24 slots for relevant sources, 12 for structurally related but distant domains and 12 for seeded draws across underrepresented families. This is a proposed sampling allocation to test, not a learned quality formula. If a bucket is empty, log the shortfall and a declared fallback. Provide a compact catalogue for the shelf, then pass only selected full images/details within the model's visual budget.
A Connection records the inquiry, candidate sources, relevant relationship, source and target roles, transformation, influence intensity, supporting evidence and realized artifact IDs. Intensity can range from a slight influence to an extreme transformation. The director must still specify an executable intervention; an intensity number is not presumed to map linearly to the renderer's visual effect.
For example, a source's enclosing fabric may become architectural confinement around another subject. Test the relationship on unfamiliar sources rather than merely repeating its recognizable motif. Label a new analogy as proposed; do not fabricate a historical influence. Assess contribution against relevant precedents and realized work. Embedding distance, source obscurity and the number of combined references are not originality scores.
Later, index relevant Wikipedia/Wikidata, museum collections, artist archives, scientific systems, craft and everyday environments. Official structured access and dumps support broad knowledge retrieval. [3] Artist Instagram accounts can contribute with authorized access and reuse, retaining captions and portfolio context. Tens of thousands of accounts are a future source pool, not a sufficient decision-training set. Keep this knowledge external to model weights initially.
Train retrieval or a connection reranker only after collecting candidate-set choices and outcomes. Compare lexical/CLIP, mixed retrieval, human reference selection and learned selection with source access held equal. Audit missed useful references, incorrect contextual claims, duplicate families and whether wider retrieval actually creates stronger work. Larger archives are justified by measured coverage gaps.
Search, protected randomness and resource allocation
Separate generation of alternatives from selection among them. The system needs chance in encounters, combinations, operations and continuations, as well as renderer seeds. Increasing sampling temperature alone does not ensure worthwhile exploration. Preserve surprising results before forcing them into the initial explanation; intentional chance can itself be a consequential artistic procedure.
For a full episode, maintain eight active slots: four favored by the declared target reviewer/context, two different directions and two protected directions. The candidate archive also stores rejected work and its lineage. Descriptors may include source families, spatial relationships, material treatment and operative questions. They organize exploration without certifying novelty or artistic value. Diversity archives have a computational precedent in quality-diversity search. [4]
This proposed 128-attempt cap is a starting policy. The 32 protected attempts cannot be reassigned because J dislikes the candidates. Four protected continuations receive four edits each unless execution fails or the global resource cap ends. Retries consume the appropriate attempt and cost budgets; the number of successful images may be lower. Keep a frozen pre-preference-training director available for these trials.
For a pool of at most eight candidates, compare all pairs under one specified reviewer/context. Candidate A receives expected preference points p(A) + 0.5*p(T) against B; its mean points rank it for the four favored slots. This is a local allocation rule, not an intrinsic art score. Record cycles, ties and sensitivity to pool membership. Readiness and development predictions remain separate tasks.
For larger pools, compare against three fixed archive anchors, retain six favored candidates plus two protected/different candidates, then rerank the shortlist with all-pairs comparisons. Initial anchors follow a fixed seeded policy. Charge every judge call and audit excluded human favorites. Do not average all critics into a universal target; separate contexts can maintain different shortlists.
Reserve predicted renderer, reasoning and review cost before dispatch, then reconcile actual consumption. The ledger tracks image attempts, model tokens, GPU time, human minutes, retries and failed calls. Use both an attempt cap and a total-cost cap. Seeds and configs support tracing; cached artifact bytes are authoritative because cross-environment reruns need not be identical.
Evaluate the reservation itself: at equal total cost, does protected search retain branches that independent humans later develop or show? Track loss of source/practice breadth as search continues and adaptation accumulates. Do not reward disagreement, maximal distance or strangeness directly; they can be produced cheaply without artistic consequence.
Advanced critique: contribution, practice and consequence
At the highest ambition, evaluation concerns what a practice contributes and how informed encounters develop over time. Use a structured critique whose criteria remain contestable. Consensus is a possible result; dissent and changed criteria are also evidence. These concerns generate questions, comparisons and experiments, never six additive scores. A numerical training loss can model situated decisions without asserting a universal numerical value for art.
Contribution. Identify the conversation, relevant precedents and the work's particular move. Ask what changes in form, experience, interpretation or social context. Craft can be assumed or deliberately refused, but neither polish nor roughness proves intentionality. Resemblance is not an automatic veto: Sturtevant's repetitions made originality and authorship part of the work. [5] Require evidence for historical claims; an unfamiliar precedent is not necessarily absent.
Practice. Review real works, failures and changes of direction. Ask whether successive pieces deepen, challenge or redirect a sustained inquiry with stakes. Distinguish a coherent project from production of a recognizable appearance. A single exceptional piece can be valued while the evidence for a practice remains incomplete. Never fabricate an artist's biography or development history to fill this gap.
Necessity. Ask why this form, scale, material, duration and presentation matter. Selected deletion, substitution and scale experiments investigate what changes in experience; alternatives may produce different valid works. A model's ability to rationalize every choice is weak evidence. Chance and accident remain valid procedures; the test concerns their consequences and what is retained. Physical claims need physical tests, and renderer side effects limit causal interpretations.
Excess over the statement. Collect grounded readings and experiences that exceed initial intent, including what viewers cannot yet articulate precisely. Revisit selected works after time has passed. Allow the work to change its question. Do not optimize interpretation count, obscurity or verbal complexity. Context-dependent and text-based practices receive the presentation that belongs to them; an image-only baseline is an experimental condition, not a universal criterion of good art.
Risk. Separate readiness to show from the value of further development. An unresolved experiment may deserve another round when a polished alternative has exhausted its inquiry. Record what is at stake and what could be learned. Interesting failure can justify support without receiving a false readiness label. Technical failure, extremity and polarization are not inherently productive risk; some consequences become legible only across a series.
Institutional context. Gallery fit includes program, curatorial concerns, exhibition conditions and feasibility. Keep it distinct from artistic judgment, market outcomes and network recognition. Compare judgments with and without prestige cues where meaningful, retaining necessary cultural context. Expertise means relevant knowledge and defensible attention; institutional reputation is an imperfect proxy. Avoid training the critic to reproduce representation or sales as an automatic quality label.
Pluralism allows different approaches to succeed without making every work equally successful. Preserve strong minority judgments and disputes about applicable criteria. Select reviewers with complementary knowledge and perspectives; calibration concerns factual reliability, uncertainty and consistent use of a stated context, not compulsory agreement. Model critics supplement this process only after validation against real critique and subsequent work.
Human review protocol and critique records
Use frequent local reviews for studio decisions and less frequent practice reviews at project milestones. A proposed practice panel has three to five people with complementary expertise; add a specialist when the work requires one. Record consented reviewer information, relevant expertise and exposure history. Model-generated dialogue must be labeled synthetic and cannot stand in for independent human judgment.
Independent encounter. Present actual work or the series in a specified condition. Collect readings, felt responses, uncertainty and willingness to develop, revisit or show before panel discussion. Permit mixed or absent affect. Use a statement-free condition where meaningful; evaluate truthful context with fresh viewers or disclose repeated-view priming.
Context and debate. Supply relevant references, process evidence and intended presentation. Check factual claims, discuss objections and record which criteria each reviewer applies. Retain initial judgments, revisions, final judgments and dissent. Do not force consensus or favor persuasive language unsupported by the work.
Consequential experiments. Select revisions or continuations that address important uncertainties. Preserve more than one defensible approach when resources allow. The director may accept, reinterpret or reject advice. Review actual results before treating a recommendation as successful.
Follow-up judgment. Use independent reviewers and selected delayed encounters to assess the developed practice. Record what became stronger, weaker or newly possible. Distinguish development value, exhibition readiness and enduring interest. Compliance with a critic is not the outcome target.
{
  "review_id": "rev_800", "reviewer_id": "expert_01",
  "task": "whole_work", "artifact_ids": ["img_A", "img_B"],
  "presentation_id": "neutral_v1", "choice": "B",
  "readiness": {"img_A": "N", "img_B": "Y"},
  "would_develop": {"img_A": "U", "img_B": "Y"},
  "label_origin": "human", "exposure_id": "session_12",
  "evidence": ["The enclosure also functions as support"],
  "abstention_reason": null
}
Extend the schema for practice reviews with work IDs, applicable criteria, cited precedents, evidence regions/time spans, objections, proposed tests, decision rationale and links between pre- and post-discussion judgments. Store original text as well as structured fields. Keep a tie, incomparability, uncertainty, missing response and refusal to judge distinct; none silently becomes a negative label.
The interface must hide condition-revealing filenames, model identity and director rationale in the primary local comparison. Balance A/B order, log viewing time and prior exposure, and allow “none” in pool selection. Keep generated statements truthful and fixed for a context condition. Annotation time is part of the cost ledger.
The highest-value training record links a practice before critique, independent and collective judgments, selected experiments, resulting works and follow-up evaluation. Such records let the system learn which criticism and artistic decisions were useful. Archive images and final verdicts alone omit that developmental evidence.
Dataset assembly, provenance and split integrity
Maintain immutable dataset snapshots built from the event store. Each row includes dataset_version, split_group_id, source event IDs, task, model-input manifest, label provenance and applicable reviewer/context. Resolve image bytes by hash; never silently regenerate a training image. Keep a data card recording coverage, exclusions, label counts, duplicates, collection cost and known confounds.
The later judge budget may begin around 3,000 training judgments, plus separately collected development and test labels. These are planning quantities, not sufficiency claims. A practice dataset needs complete projects, contrasting approaches, rejected trials, reversals and delayed outcomes. The number of review sentences is not the number of independent artistic decisions.
Split first. For supervised work, allocate related groups approximately 60% training, 20% development and 20% sealed test. This does not replace E0's four/eight brief split or a power calculation. Keep a project, its closely related briefs, seeds, crops, alternate renderings and all descendants together. Perform grouping before augmentation, task expansion, balancing or nearest-neighbor indexing of experiment memory.
Define source access for each claim. Ordinary new-project transfer may use a fixed shared reference archive. A source-family or artist-family transfer test excludes that family from training and training-time retrieval. New-reviewer tests exclude the person's labels except a declared adaptation set. Time-based tests freeze the training cutoff. “Unseen by Kusama's training” does not mean unseen during foundation pretraining.
Build inputs at decision time. An intervention row resolves only fields available before the selected action. Its child image and follow-up reviews are targets or selection evidence. Reviewer profiles contain training-only examples. Sealed review labels never enter prompts, retrieval, memory promotion, checkpoint selection or hyperparameter tuning. Reject snapshots that violate these boundaries rather than merely logging warnings.
Preserve failed operations, rejected alternatives, uncertain judgments and disagreements. A failed action can be a negative example or a recovery context; it is not a positive SFT target merely because its trajectory later succeeded. Human, synthetic and automatically derived labels use separate fields and filtering rules. Do not count reversed A/B pairs or readiness rows from one encounter as new independent judgments.
Use active collection to cover uncertainty, rejected directions and random cases. A proposed mix is 50% uncertain, 25% rejected and 25% random, with reviewer/source balancing where feasible. Adapt the mix after measuring coverage. Track learning curves at 500, 1,000 and 3,000 judgments and complete-project counts; add data that addresses actual errors rather than accumulating familiar favorites.
Local judge: objective, calibration and selection
Train J as a conditional predictor of human responses, beginning with an independent Qwen3-VL language adapter. Inputs contain the presented artifacts, task and a reviewer/profile version. Start with up to four short examples from that reviewer's training data, selected by a fixed policy. Compare profile conditioning, ID-only conditioning and a frozen few-shot baseline. An unseen reviewer receives a generic prior until an explicit adaptation set exists.
Use separate task prompts and label sets: whole-work comparison {A,B,T}; readiness {Y,N,U}; willingness to develop {Y,N,U}. T means equal preference, not that both works are ready. U is an explicitly uncertain human response. Missing labels and abstention are excluded from these initial losses; incomparability remains a separate review state until it has its own validated task.
Implement response-only supervised fine-tuning (SFT) using a tested multimodal collator. For presented input x and human label completion y, minimize token cross-entropy on completion tokens only. Prompt, padding and image tokens carry ignore labels. TRL supports visual SFT, but its processor, template and masking path must pass the local integration checks. [9]
L_J = - sum_t log p_theta(y_t | x, y_<t)
s(label) = sum_t log p_theta(label_t | x, label_<t)
P(label | allowed labels, x) = softmax(s(label) / temperature)
At inference, teacher-force every allowed completion under the identical prefix and stopping convention; score the full sequence, not an assumed single label token. Report the normalized restricted-label distribution. Fit temperature using development predictions only and archive the calibration artifact. Separately measure invalid labels under unconstrained generation, reversal consistency and uncertainty on repeated presentations.
Balance reviewers and, where genuine labels exist, start sampling 60% comparison, 20% readiness and 20% development rows. Randomize A/B order and remap labels consistently. Keep truthful-context and image-only tasks distinct. A reviewer changing their mind is a phenomenon to record; cross-reviewer disagreement is not automatically annotation error.
Cheap baseline. Freeze normalized CLIP vectors eA/eB and concatenate their signed difference and absolute difference into 1,024 features. Fit per-reviewer L2-regularized multinomial logistic regression; standardize on training data only and choose C from 0.01, 0.1 and 1.0 on grouped development data. Add consistent reversed pairs. Train readiness/development on 512-dimensional single-image features. Report constant baselines for one-class data and no personalization claim when evidence is inadequate.
Evaluate per-reviewer accuracy, macro-F1, log loss, Brier score, calibration and A/B reversal. Also measure human-favorite retention in a fixed candidate pool, false readiness, rejected exceptional work and pool-choice agreement including “none.” Good average pairwise accuracy can miss the selection decisions that matter most.
Use development-calibrated error/margin thresholds to route uncertain decisions to review or preserve alternatives. Written confidence is not calibration. Never let J veto protected exploration, silently average distinct reviewers into one taste, or treat its probability as proof that an unexplored direction has no value.
Director and practice-critic training
Initialize D independently from J. Supervised inputs contain the prior image, fixed brief, current inquiry, selected sources, prior observations, allowed actions and remaining budget. Targets are concise executable actions supported by reviewed results. Include source selection and connection transformation when candidate sets and realized outcomes are available; do not train only the last editing step.
Preserve subtraction, restraint, successful recovery and stopping in the positive set when evidence supports them. Failed actions remain negatives or starting states for recovery. Short intent fields may describe a proposed effect, but eloquent explanation is not a success label. The learning target is a decision available at the time, not a fictional retrospective account of how a famous artist must have thought.
Optional offline preference optimization. After SFT improves artifacts, construct (x, a_preferred, a_rejected) from different actions at the same prior state. Match source access, renderer settings and resource budgets. Human preference is over actual outcomes. Exclude unresolved pairs and comparisons dominated by execution failure unless the declared task is tool reliability. Two seeds are diagnostic; uncertain rankings need more realizations or omission.
delta = log pi_D(a_preferred | x) - log pi_D(a_rejected | x)
        - log pi_ref(a_preferred | x)
        + log pi_ref(a_rejected | x)
L_DPO = -log sigmoid(beta * delta)
pi_ref = frozen, exact director SFT checkpoint
Direct preference optimization (DPO) changes D's action probabilities; R remains frozen and no gradient through image generation is assumed. [7] Cache reference completion log probabilities from the exact SFT adapter, with fixed images, tokenizer, template and masks. Check cached values against direct forwards. Disabling the adapter and using the original base is not the intended reference. Verify TRL's multimodal DPO path and cache compatibility before a full run. [10]
For delayed effects, compare continuations from the same parent with equal remaining budgets and evaluate terminal works or series. Keep local-transition and terminal-outcome datasets distinct. If later policies or resource allocations differ, preference over final portfolios cannot identify the value of the first action alone. Train and evaluate longer planning only when those comparisons are adequately controlled.
Practice critic C. Begin with real critique sessions and a frozen assisted workflow. Later train a separate task or adapter to ground claims in artifacts and verifiable context, find relevant precedents, propose informative tests and estimate reviewer/context-specific development choices. Include conflicting legitimate readings, revised criteria, abstention and cases where ignoring advice produced stronger work. Reviewer-conditioned learning has precedent outside fine art; this application needs independent evidence. [6]
Critic targets are concise observable claims, cited context, questions, actions and decisions; do not reward elaborate private-reasoning narratives. Verify visual claims against actual portfolio images and relevant details. A summary-only model cannot certify form, material or spatial experience. Evaluate whether its advice improves subsequent work, not merely whether it sounds like an art critic.
Keep SFT, DPO, improved retrieval and critic assistance as separately versioned changes. Online reinforcement learning is optional and justified only if simpler offline methods leave a demonstrated bottleneck. Independent human review remains external to training; synthetic self-approval cannot establish artistic progress.
Training configuration, preprocessing and GPU gates
The following are initial experiments for separate J and D runs, not measured optima. Prepare the model for k-bit training and enumerate fully qualified module names. Target language attention projections q_proj/k_proj/v_proj/o_proj and MLP gate_proj/up_proj/down_proj; freeze the vision encoder, visual merger, embeddings, output head and base parameters. Fail if the target set is empty or contains vision modules; blindly selecting all linear modules would violate this boundary. [8, 11]
Effective batch equals microbatch × accumulation × data-parallel workers. On one worker, accumulate 32 SFT microbatches or 16 DPO pairs. Record config revisions after out-of-memory recovery. A small development-only sweep can compare rank 16/32 and SFT LR 5e-5/1e-4; repeat selected settings instead of reporting the luckiest seed.
Resize initial local-task viewing copies to a 512-pixel long edge while preserving aspect ratio, then use the checkpoint's own processor. J receives two images; D receives a parent plus at most two references. Keep full-resolution originals for people. Enforce a total processed input-plus-output cap of 8,192 tokens, reserving up to 512 action-output tokens. Reduce text/references before batching and record the final image/token counts.
Set trainer max_length=None after this admission check so automatic truncation cannot remove image tokens. Inspect decoded examples and completion masks for every task; use a verified explicit collator if the chat template lacks usable assistant masks. Portfolio-level critique requires a separately profiled larger visual input policy. [9, 10]
Before a full run, execute forward, backward, optimizer update and save/reload on real batches. Confirm nonzero intended gradients and unchanged frozen weights. Record peak allocated/reserved VRAM, host RAM, token counts, step time and output equivalence after reload. Evaluate the largest admitted sample, not just a small text example.
Profile each available GPU separately. Begin with one active workload per GPU or serialized roles; additional cards can run independent render jobs. Data parallelism follows a successful single-worker profile; summing VRAM does not create one larger device. Measure CPU offload and communication costs. Pin the tested Python/PyTorch/CUDA environment, processor/model SHAs, dependency lock and container digest before scaling.
Greenfield repository, execution and observability
Create a Python 3.11 package with uv, pyproject.toml, uv.lock, .python-version, .gitignore, .env.example, config examples and an exact setup/run README. No reusable implementation or pre-ingested corpus is assumed. Resolve and test dependencies before committing a lockfile; verify imports for Qwen3VLForConditionalGeneration, Flux2KleinPipeline and the required multimodal trainers against pinned versions.
Use FastAPI/Uvicorn, Pydantic, SQLite through one metadata writer and filesystem/object storage for artifacts. Start with a simple leased job queue. GPU dependencies are PyTorch, Transformers, Diffusers, Accelerate, PEFT, TRL and bitsandbytes; data tooling includes datasets, PyArrow, Pillow, NumPy, scikit-learn, SciPy and FAISS CPU where useful. Install dependency groups as their stage becomes necessary. Generate UI types from the shared schema.
Environment configuration covers access tokens where required, model/cache directories, artifact/database paths, device, precision, budgets and seeds. Keep secrets out of code, manifests and logs. Store JSON/YAML run configs, append-only JSONL events, Parquet dataset snapshots and safetensors adapters. Binary artifacts and model caches are not Git source files.
Job lifecycle. Validate and reserve resources, commit a queued job, lease it to a worker, execute, persist bytes atomically, append artifact/events, then reconcile cost. Use unique idempotency keys and a retry lineage. A timed-out external request may have completed; mark it uncertain and reconcile before resubmitting. Recovery must not silently lose outputs or spend the same reservation twice. Final images may not be recreated from seeds as a substitute for missing bytes.
Renderer contract. Start FLUX base at 1,024 × 1,024, 50 steps, guidance 4.0 and BF16; use the saved parent for editing and declare any additional sources. These are documented example settings to profile. [2] Check renderer prompt length after processing. Prompt editing does not guarantee masks, layers or unchanged pixels; explicit composition/masking tools must advertise their actual controls and receive separate execution records.
Every run records software/model revisions, prompt/compiler versions, inputs, policy and adapter hashes, seeds, requested/observed changes, viewing conditions, worker inventory and costs. Separate schema validity, execution success, preservation, preference and readiness. Maintain typed errors and terminal job states; an operational success must not masquerade as artistic success.
Expose CLI commands for doctor, ingest, split, run, export, train, evaluate and release. Each accepts a pinned config and writes a manifest. Promote only immutable evaluated releases; worker restarts must resolve the same release rather than “latest.”
First implementation, experiments and release tests
First vertical slice. Build only the renderer adapter, config loader and evidence log. Generate a collage of a person beside a mountain with seed 17. Edit that saved parent independently with seeds 23 and 41 to reduce the person's height to roughly one third while retaining its ground anchor, mountain framing and palette. Save parent, both results, comparison sheet and lineage/config/hash manifest. Inspect actual change and preservation; record latency, memory and failure/resume behavior. If GPU execution is unavailable, complete CPU contract checks and mark the visual gate unverified.
E0: test execution and useful feedback. Use 12 original briefs: four for protocol development, eight sealed. Generate one parent each. Compare an inquiry-driven relationship edit, the strongest ordinary critique-and-edit action and an exploratory action, each with two realizations. Planned output is 12 + 12×3×2 = 84 images. Three reviewers assess 72 parent/edit pairs each: 216 judgments total, plus 36 best-candidate-or-none pool selections. Balance presentation order, blind system identity, retain ties and ask readiness separately. Budget failures and additional context/practice reviews explicitly.
Subsequent tests separate process, memory, weight updates and transfer. Compare direct best-of-N, strong generic critique-and-edit, Kusama before/after each trained component, and a defined human-directed workflow. For selectors, use the same fixed candidate pool. For generation, match source access, execution conditions and total resource budgets; record human involvement. Ablate connection selection, experiment memory, protected allocation and practice critique. Do not weaken a control or discard unfavorable test outputs.
At practice level, compare matched development opportunities and resulting bodies of work. Use independent reviewers with relevant expertise and record minority nominations, readiness, development value and decisions to revisit. Analyze whether critique advice helped the subsequent work. A single aggregated win rate must not erase major differences between practices or reviewers.
Report per-reviewer preference, tie/incomparability rates, readiness, human-favorite retention, false readiness, cost per ready work and quality-versus-budget curves. Use brief/project-level grouping and account for repeated reviewers; sibling images are not independent trials. Freeze the primary comparison, practical effect threshold, readiness tolerance and confirmatory sample size after estimating pilot variance. A proposed target is 55% tie-adjusted preference with a 95% interval excluding 50%; eight sealed E0 briefs cannot establish this by assumption.
Implementation gates cover schema/lineage integrity, idempotent recovery, split leakage, image order, completion masks, adapter gradients, frozen weights and save/reload. Research gates require useful feedback, a process advantage, an incremental learning advantage and then breadth. Neither decreasing training loss nor increasing internal reward releases a model without artifact evaluation.
The first iteration succeeds if it can execute meaningful alternatives, collect credible discriminating feedback and expose where a simple baseline wins. It need not already be a convincing autonomous fine artist. Use those failures to choose the next investment rather than immediately scaling ingestion or training.
Transfer, scaling and technical foundations
Expand first to new projects, then unfamiliar subject/source families, visual practices, reviewers and later contexts. Separate intent adherence from preference: correct Bauhaus cues are not proof of strong work, and restrained work must not fail because a training set favors dense collage. New-reviewer adaptation can be measured at 0, 10, 50 and 200 labels with costs reported. Recheck earlier practices for forgetting and convergence after every material update.
Scale in four stages: a verified narrow studio; multiple sustained practices with broader process data; a shared artistic action policy with medium-specific tools and real outcomes; then, only if justified, a larger or unified multimodal model. Demonstrate whether retrieval, perception, control or judgment is the bottleneck before replacing the backbone. More parameters and more images do not by themselves teach useful artistic decisions.
A future unified model would need interleaved source/context/decision/artifact sequences, diverse process trajectories, comparisons, critique-to-revision sessions and real material or spatial feedback. Begin from a suitable pretrained model where possible; adapt representations or generation only after controlled substitutions show that the modular interfaces limit results. Full foundation pretraining requires a separate data, compute and research case. The current GPU fleet is a platform to profile adapters and experiments, not evidence that competitive foundation training is feasible.
The potential advantage is accumulated evidence of consequential reference transformations and artistic development, with uncertainty and dissent retained. A benchmark can support bounded superiority under a declared protocol. It cannot certify future historical importance or superiority to all human practitioners.
Selected sources and implementation references
These support component methods and examples. The integrated architecture, sampling allocations, critique protocol and numerical targets are proposed research, not validated results.
[1] Qwen3-VL-8B-Instruct — official model card.
[2] FLUX.2-klein-base-4B — official model card.
[3] Wikidata — data access documentation.
[4] Mouret and Clune — Illuminating search spaces by mapping elites.
[5] MoMA — Sturtevant: Double Trouble.
[6] Poddar et al. — Variational Preference Learning.
[7] Rafailov et al. — Direct Preference Optimization.
[8] Dettmers et al. — QLoRA.
[9] Hugging Face TRL — SFTTrainer and visual datasets.
[10] Hugging Face TRL — DPOTrainer and visual preference data.
[11] Hugging Face PEFT — LoRA configuration and target modules.
Role
Input
Output
Initial learning
Director D
State, inquiry, sources, prior outcomes, budget
Executable action JSON
Separate language LoRA
Renderer R
Compiled create/edit request
Actual image bytes
Frozen
Observer O
Parent, result, requested changes
Visible changes and side effects
Frozen task
Local judge J
Ordered artifacts, task, reviewer context
Allowed-label probabilities
Separate language LoRA
Practice critic C
Series, context, critique history
Evidence, objections, next tests
Human-led; later separate task
Scheduler
Candidate archive, judgments, ledger
Jobs, retention, stopping
Ordinary code first
Record
Essential fields and relationships
Practice / Project
Practice ID; project IDs; external brief; inquiry revisions; commitments; intended context
Work / Artifact
Work ID; ordered artifact versions; parent/action IDs; bytes URI; SHA-256; dimensions; media type
Action / Run
Prior state; operation; parameters; sources; preservation targets; idempotency key; policy; budget
Reference / Connection
Source provenance and permitted uses; candidate relationships; role transfer; realized outcomes
Review / Critique
Reviewer; task; presented artifacts; condition; choice/abstention; evidence; uncertainty; exposure
Phase
Renderer-attempt allocation
Purpose
Initial realizations
24
Establish competing directions
Ordinary development
48
Revise promising candidates
Protected fresh directions
16
Reach underexplored alternatives
Protected continuations
16
Give uncertain directions time
Finishing
16
Develop selected final work
Intervention checks
8
Probe consequential choices
Dataset family
Required evidence
Initial collection role
References / connections
Sources, candidate sets, role transfer, realized work
100–300 audited sources; coverage pilot
Human reviews
Individual choices, context, uncertainty and exposure
E0: 216 diagnostic pair judgments
Interventions
Prior state, action, actual result, observations, reviews
Later: 1,000–3,000 reviewed transitions
Practice / continuations
Series before/after critique; alternative trajectories; cost
Complete development sessions, then learning curves
Setting
Judge/director SFT
Optional director DPO
Frozen base
4-bit NF4; double quantization; BF16 compute
Same base; start from SFT adapter
Adapter
Rank 32; alpha 64; dropout 0.05
Continue that adapter
Optimizer
AdamW; LR 1e-4; decay 0.01
AdamW; LR 5e-6; decay 0.01
Schedule
Linear decay; 3% warmup; at most 2 epochs
Same; at most 1 epoch; beta 0.1
Batch
Microbatch 1; effective batch 32
Microbatch 1 pair; effective batch 16 pairs
Stability
Gradient clip 1.0; checkpointing; no packing
Same; cache fixed reference scores
Repeats
Seed 17 for integration; then 23 and 41
Same; frozen pair manifest
Module
Responsibility
src/kusama/schemas
Pydantic contracts, JSON Schema, shared validation
src/kusama/models
Director, observer, judge, renderer and critic interfaces
src/kusama/runtime
Coordinator, queue, budgets, archive and retrieval
src/kusama/data
Ingestion, lineage, splitting, export and validation
src/kusama/train / evaluate
SFT/DPO, calibration, model tests and human comparisons
review-ui / configs
React/TypeScript review UI; immutable run and model manifests

