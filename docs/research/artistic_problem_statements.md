# Artistic problem statements

Direction established **8 September 2026** from the user's instruction. This defines the starting point for a full artistic run and a future research capability; it does not report a new run, software gate or trained model.

## A full artistic run begins with a problem

**A full artistic run starts with a sufficiently complex artistic problem statement.** The problem gives inquiry, reference research, alternative actions, critique and revision a shared purpose. A subject, preferred appearance or rendering instruction alone does not supply that purpose. Single-image generations and controlled edit diagnostics remain useful component experiments; executing them does not by itself constitute a full artistic run.

“Sufficiently complex” means that the problem has consequential stakes, unresolved tensions and room for meaningfully different artistic responses. Complexity comes from what is at issue, not sentence length or theoretical vocabulary. Problems may be perceptual, material, personal, social, historical or speculative; they need not all adopt the concerns of the primary case study.

Before starting a new full run, the operator should be able to explain:

- **Question and stakes:** what is being investigated, for whom or in what context it matters, and what remains uncertain.
- **Productive tension:** which forces, assumptions or desires resist an easy resolution.
- **Grounding:** what supports the framing, distinguishing documented source context from the operator's interpretation and the model's proposed analogies.
- **Artistic openness:** how substantially different approaches could investigate the question without prescribing one composition, medium, style or answer.
- **Development and critique:** what choices could be tried and revised, and what actual features of the resulting works would let viewers discuss whether the inquiry has become consequential.

This is a documented readiness criterion for planning, not an implemented automatic complexity score. An eloquent statement does not establish the quality of the work it produces.

## Primary case study: Wangechi Mutu

The user supplied this central question for Kusama's primary case study:

> How can the Black female body—historically subjected to objectification, colonial classification, and imposed beauty standards—be reconstructed as a source of power, self-definition, and new possibilities for existence?

This is the user's framing of the case study, not a quotation attributed to Mutu or a claim that it exhaustively describes her practice. Historical and artwork-specific claims used during a run still require source grounding. Mutu anchors this initial inquiry and process study; her appearance and concerns are not a universal standard for other projects.

The question connects imposed ways of seeing and classifying a body with the possibility of self-authored existence. It leaves unresolved how reconstruction can produce agency without repeating objectification or replacing one imposed ideal with another. That tension can sustain reference research, competing artistic actions and revisions beyond a single illustration of empowerment.

Possible subordinate questions below are assistant-authored elaborations, not supplied statements by Mutu or validated artistic solutions:

- How might fragmentation or recombination make self-definition visible while retaining the tension of bodily vulnerability?
- How could a work disrupt the viewer's authority to classify the figure rather than merely offer a new category?
- What relationships between body, material and environment could suggest possibilities for existence beyond a single idealized image of power?

These are optional inquiries, not mandatory motifs. Critique should examine actual works and competing interpretations: where does agency become visible, where might the work reproduce the objectification it investigates, and which unresolved possibility deserves development? Keep these discussions distinct from observable edit success, finished quality, development potential and readiness. A model's explanation cannot settle them.

## Carry the problem through the run

The intended sequence is problem statement → grounded inquiry and reference research → alternative artistic hypotheses/actions → realizations → critique and selection → selected-parent revisions and follow-up review. Research and critique may change the inquiry; preserve the original statement and record each revision with authorship and reasons. Do not rewrite earlier frozen inputs to make them appear to have begun with a later formulation.

The problem statement establishes purpose. The working brief bounds an episode, an artistic hypothesis proposes a way to investigate the problem, and a renderer prompt specifies a particular realization. Preserve their relationship so that a local edit can be assessed in the context of the larger inquiry. A full run may stop with unresolved questions; no requirement to “solve” the historical or social problem follows from completing an episode.

Current interfaces accept operator-supplied briefs and versioned inquiry state. Use the existing [studio inquiry contract](../workflows/studio.md) and [source-connected brief workflow](../workflows/source_connected_studio.md); neither validates conceptual adequacy. Historical fixtures, briefs, frozen runs and results retain their original scope. This documentation does not retroactively qualify them as full problem-led artistic runs.

## Discover through making before artistic commitment

The user's 9 September assessment recognizes a connection from the *Veranda Post* to agency and collective dependence, followed by a crown proposal, as conceptual exploration. It also identifies what remained missing: making alternatives whose actual appearance could change the artist's interpretation or intended direction. An interpretive move is a beginning, not evidence that the proposed work has been tested.

For new full runs, require several provisional studies with different relationships or artistic operations and an explicit reason for trying each. Inspect the actual images for visible relationships, unexpected effects and gaps between image and explanation. Then choose, combine, abandon or reformulate according to the artist's commitments. The user-supplied possibilities of a crown as burden, construction or collective structure are illustrations of this openness, not prescribed answers. A useful study may reveal that the crown is too obvious or that relationships between bodies deserve to become the subject.

Research commitment freezes a study for rendering; it is distinct from artistic commitment after inspection. The [9 September implementation record](../operations/2026-09-09_studies_before_commitment.md) describes the bounded software path and its unverified artistic meaning.

## Future model for constructing problem statements

The project intends to develop a model capable of adequately constructing these problem statements. This is an upstream capability, distinct from proposing image directions, writing renderer prompts or judging completed images. Architecture and training method remain open; there is no dedicated implemented or validated problem-statement model, dataset contract or training result established here.

The proposed capability should use permitted context, interests and sources to formulate candidate questions with stakes, tensions, grounding, open alternatives and remaining uncertainties. Preserve raw proposals, source context, model identity, operator edits and acceptance or rejection with reasons. Missing feedback is not rejection. For now, the operator supplies and assesses the starting problem.

Adequacy needs two separate evidence gates:

1. Independent human assessment of candidate statements for specificity, consequential stakes, grounding, unresolved tension, openness and usefulness for sustained artistic decisions. Retain disagreement and compare against human-authored statements and a frozen prompting baseline.
2. Downstream comparisons of the work and development those statements enable, with the same source access, rendering/development workflow and total budgets, including formulation costs and disclosed human effort. Better prose alone cannot establish better artistic inquiry or outcomes.

Before any learning, collect attributed examples and genuine assessments, group paraphrases and descendants of the same problem/project together, and reserve unfamiliar problems and relevant source families for the claimed transfer test. Later artworks and sealed reviews must not leak into decision-time formulation inputs. Evaluate frozen prompting improvements separately from weight updates under the [learning prerequisites](../models/model_and_training_status.md).

This future capability does not displace the current evidence gate: genuine human feedback on saved work, an explicit selected parent, a controlled revision and follow-up review. Its own open question is whether constructed problems sustain consequential artistic development across unfamiliar projects.
