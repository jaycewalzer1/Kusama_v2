# Research stagnation after acquisition — 9 September 2026

Read-only audit of the latest panel run, `studio-87f61ec22af04bf3978b`. It ran from 14:37:58 to 14:43:37 UTC (10:37:58–10:43:37 EDT), approximately 5 minutes 40 seconds. The original user question is preserved in the [manifest](../../artifacts/runs/studio-87f61ec22af04bf3978b/manifest.json); the frozen research request attributes it to `operator`. Searches and notes are frozen-Qwen outputs. Diagnoses below are assistant assessments, not human artistic judgments.

**The run acquired and displayed one reference, but Qwen never drafted a hypothesis. The new stagnation guard stopped it after 15 calls, before the separate synthesis reserve became active.** FLUX was never invoked. This establishes a real source-acquisition/visual-input path, not an artistic outcome or successful synthesis.

## Verified accounting

| Measure | Result |
| --- | --- |
| Research version / status | `independent-research-v7-research-progress` / `stagnated` |
| Panel status | `partial`, correctly explaining no committed direction |
| Qwen calls / actual combined tokens | 15 completed / 76,663 |
| Search decisions | 10: eight executed, two duplicate attempts blocked |
| Other decisions | Two opens, two notes, one acquisition |
| Executed search results | Six empty eligible-result sets, two nonempty sets |
| Metadata candidates / distinct objects | 16 records / 10 objects |
| Network | 11 completed requests; 281,255 charged bytes |
| Acquired references | One: Winslow Homer's *Flamborough Head, England* (`artic_16791`) |
| Actual image exposure | One distinct saved image supplied on decisions 10–15 |
| Drafts / commitments / FLUX calls / generated images | 0 / 0 / 0 / 0 |
| Unused research allowance | Nine of 24 actions |

There were no rejected JSON responses, inference failures, failed network requests or unsettled model calls. The 48-request, 12-acquisition and 120-minute ceilings were not exhausted. All 15 prompts admitted: 13 at projection stage 0, one at stage 2 and one at stage 4, with 2,927–6,356 input tokens against a 6,392-token allowance. Early FLUX instruction validation was never reached because there was no draft. Financial totals remain incomplete because local hardware/electricity rates are unknown.

## Trace and interpretation

Actions 1–4 issued closely related Met queries about Black women, self-representation, resistance and visual agency. They yielded no eligible candidates; some matched records lacked an explicitly eligible CC0 image URL. These bounded responses do not establish absence across the museum's holdings.

Action 5 searched the Art Institute and returned eight candidates. Action 6 opened Homer's candidate metadata, which explicitly reported `acquisition_required: true` and contained no image. Action 7's note discussed monumental composition, viewpoint and watercolor technique. Those details are present in the saved museum description: this was metadata-based interpretation, not verified pixel observation or evidence of hallucination merely because the image had not yet been acquired.

Action 8 added `history` to the Art Institute query and returned a partly overlapping set, including a Frederick Douglass photograph. Action 9 acquired Homer's work under its canonical ID. Decisions 10–15 actually received that image. The Douglass candidate also remained in `candidate_memory` throughout these decisions, so this case is not the previous failure where a returned lead disappeared from ordinary context. Whether Douglass would have been a useful complementary reference is untested.

Actions 10–15 were two empty searches, another opening of the same acquired work, two blocked duplicates and a note. None added a new source, passage, draft or commitment. The final note proposed further searches and non-museum sources; those additional source providers were not available under this frozen run's three-museum access policy. The model never used `draft`, `commit`, `revise_question`, `recall`, `read` or an explicit `stop` action.

## What worked and what remains wrong

**The previous safeguards worked as limits.** Online access, metadata retrieval, acquisition, saved visual inputs, candidate retention, prompt admission, duplicate blocking and truthful panel status all have live evidence in this run. The blocker is no longer a malformed action or a missing rendering dispatch after commitment.

**The controller's stagnation and synthesis policies are disconnected.** In [research.py](../../src/kusama/research.py), `_research_progress` switches into synthesis only when three or fewer total actions remain. The execution loop independently stops after six consecutive actions without new evidence or a draft/commit. Thus the acquisition at action 9 reset the counter; actions 10–15 filled it; execution ended with nine actions unused. Before the last decision, `phase` was still `research`, ten actions remained, and `allowed_actions` still offered the entire action set. A generic instruction to draft was present, but no earlier enforced synthesis/replanning checkpoint occurred. This is a design gap in the assistant's preceding patch, not evidence that the frozen guard malfunctioned.

**Search reformulation remained shallow.** The model rearranged similar thematic vocabulary and added a word, instead of developing distinct source inquiries. Exact duplicate detection blocks normalized provider/query/limit matches, but does not recognize those near-repeats. The model reasonably recognized that the Homer work did not directly address the question's specific histories; that does not justify claiming it should simply commit a hypothesis from this source. However, it did not turn that objection into a materially different executable inquiry or a grounded proposal.

**Image availability did not produce a hypothesis.** Supplying pixels is now verified; attention to them and an effective artistic transformation are not established. The absence of a draft cannot be blamed on hypothesis-schema or render-token rejection: neither path was attempted. Repeated exposure to the same reference does not count as six independent discoveries or quality evidence.

## Next evidence gate

For a new version, introduce a bounded synthesis/replanning checkpoint before stagnation termination when acquired evidence exists. The model should explicitly assess whether it can propose a grounded visual experiment, needs one materially different permitted inquiry, or should stop because the evidence is inadequate. Preserve that choice and its reasoning; do not fabricate a hypothesis or automatically substitute an operator-selected source. Tie this checkpoint to both inactivity and remaining total budget, rather than only the final three steps. Near-repeat search summaries and explicit acquired-versus-metadata-only state would make the next choice clearer.

A fixture should reproduce acquisition followed by empty/repeated searches and verify that the checkpoint occurs before the stop, without new budget or silent retries. A separately admitted live diagnostic must then establish whether Qwen actually proposes an adequate hypothesis. Rendering and human artistic assessment remain subsequent gates. Raising the cap alone is not supported by this trace.

## Integrity and scope

Authoritative read-only replay passed. Verified all 67 registered research artifact hashes, 11 source snapshot hashes and 11 cached request-byte hashes; the research implementation hashes match the frozen plan. The [research report](../../artifacts/runs/studio-87f61ec22af04bf3978b/research/report.json), [last admitted decision](../../artifacts/runs/studio-87f61ec22af04bf3978b/research/decision-014.json) and [last action](../../artifacts/runs/studio-87f61ec22af04bf3978b/research/step-014.json) retain the direct evidence. Artifact links require this local store and do not resolve in a fresh checkout.

This audit updates documentation only. It made no inference, network acquisition, review or training call and did not change or resume the run. The proposed controller change is not implemented by this audit.
