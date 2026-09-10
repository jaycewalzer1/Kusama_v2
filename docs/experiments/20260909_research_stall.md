# Research exhausted without a hypothesis — 9 September 2026

Read-only analysis of `studio-5ffc9255fbe94d5ab668`, the latest local panel run at inspection. Execution was 9 September 02:25:23–02:33:47 UTC (8 September 22:25–22:33 EDT), approximately 8 minutes 25 seconds. The question and authorship were supplied by `jjweezy`; searches and notes were produced by frozen Qwen. This report's diagnosis and proposed changes are assistant assessments, not human artistic judgments.

The run stalled in search and never reached reference inspection or hypothesis construction. FLUX was never dispatched. There was no hypothesis waiting for a selection: neither a draft nor a committed hypothesis existed.

## Verified execution

| Measure | Recorded result |
| --- | --- |
| Research status | `step_limit` |
| Qwen calls | 24 completed; 109,109 actual combined tokens |
| Actions | 20 remote searches, 4 notes |
| Source requests | 29 completed of a 48-request ceiling; 82,530 charged bytes |
| Candidates | 38 metadata records representing 28 distinct reference IDs |
| Acquisitions / opened references / viewed images | 0 / 0 / 0 |
| Drafts / commitments / FLUX calls / outputs | 0 / 0 / 0 / 0 |
| Configured output ceiling | 4 directions × 2 images, `flux_base4b_study_512_v1` |
| Time allowance | 120 minutes; the action cap ended execution much earlier |

All responses were accepted and all actions completed without a recorded tool error. The token-admission fix worked for these calls: 22 contexts used compact stage 0 and two used stage 1, which removes schema labels and audit metadata. Admitted inputs ranged from 2,486 to 6,358 tokens, below the 6,392-token allowance. There were no uncertain resource calls. Financial cost remains incomplete because local hardware and electricity rates are unknown; zero provider API cost is not zero total cost.

Evidence: [manifest](../../artifacts/runs/studio-5ffc9255fbe94d5ab668/manifest.json), [frozen research plan](../../artifacts/runs/studio-5ffc9255fbe94d5ab668/research/plan.json), [research report](../../artifacts/runs/studio-5ffc9255fbe94d5ab668/research/report.json), and the adjacent resource, job, source and cost SQLite ledgers. These artifact links require this local artifact store and will not resolve in a fresh checkout.

## Why it stalled

1. **Qwen searched for the desired conceptual answer instead of developing tractable source inquiries.** Queries grew into long phrases about contemporary collage, Black female subjectivity and appropriation. The saved results often matched individual words: clothing described as black, European portraits and decorated vessels. Met searches also excluded records without an explicitly eligible CC0 image URL. Access and retrieval limitations were real, but they did not establish an absence of useful sources.

2. **It overlooked a directly relevant lead.** Action 4, [saved step 003](../../artifacts/runs/studio-5ffc9255fbe94d5ab668/research/step-003.json), returned *Untitled ("I Sell the Shadow to Support the Substance")*, attributed in collection metadata to Sojourner Truth in collaboration with S. C. Wright. The full preserved description discusses Truth's use of photography, sale of portraits and control of her own image. The next admitted decision contained its title, maker and the first 500 description characters, including her activism and strategic use of photography. Qwen nevertheless searched Cleveland instead of acquiring it. This is evidence of a missed lead, not proof that the source would have produced a good artwork. Frederick Douglass also appeared as a possible adjacent lead.

3. **The notebook converted weak search outcomes into unsupported conclusions.** Action 6 claimed the collections yielded only a white woman's portrait and confirmed a lack of accessible representations of Black women as autonomous subjects. The earlier Truth result contradicts that summary. Qwen also described subjects without viewing any pixels. Actions 10 and 13 repeated the broad conclusion. These are model claims, not findings about the collections or the user's premise.

4. **Context retention favors notes over uninspected evidence.** The [research payload](../../src/kusama/research.py) retains the latest tool result, five action summaries and four notes. Earlier action summaries retain queries but no candidate titles or result summaries. After the next search, the Truth lead disappeared from ordinary decision context; a subsequent note also replaced the last tool result. The source remained on disk and could be recovered through `recall`, but Qwen never used it. Its inaccurate notes remained available. This design plausibly reinforced the loop; a controlled comparison would be needed to isolate its causal contribution.

5. **The controller allowed repetition through the final action.** One identical Art Institute query appeared seven times, and the same query at Cleveland three times: eight repeated provider/query actions beyond their first occurrences. URL caching reused prior responses, so these repeats did not each cause new network traffic, but each still consumed a Qwen call and research step. The final [decision 023](../../artifacts/runs/studio-5ffc9255fbe94d5ab668/research/decision-023.json) explicitly showed `remaining_steps: 1` and recent repetitions; Qwen chose another identical search. There is no enforced stagnation transition or reservation of steps for acquisition, drafting and commitment.

6. **Rendering correctly required a committed hypothesis, but completion reporting obscured the outcome.** The [panel worker](../../src/kusama/panel_runs.py) accepts research `step_limit` as a terminal state and marks an empty hypothesis list `completed`, then returns before preparing rendering. Its explanatory message is accurate, but `completed` conflates a finished worker with a realized artistic run. The manifest's initialized `model_calls: 0` also remains stale; the authoritative ledger records 24.

## Proposed next change and evidence gate

For new, versioned plans, retain a bounded candidate shortlist and search-outcome summaries separately from model notes; disclose repeated searches and enforce a bounded response to stagnation. Teach source inquiry through short entity/material/form queries without prescribing artistic conclusions. Make the remaining acquisition → draft → reinspection/commit sequence explicit while enough steps remain, with an honest unresolved stop when evidence is insufficient. Do not force a fabricated hypothesis simply to produce images.

Report `step_limit` with no hypotheses as an incomplete artistic outcome and derive call totals from the resource ledger. Increasing the step budget alone is not supported by this trace: it risks extending the same loop. Wider source access may eventually help, but this run already missed an available lead.

The next software gate is fixture coverage for repeated searches, candidate retention, budget-aware termination and truthful panel status. The next live gate, after a separately admitted new run, is acquisition and actual visual inspection leading to a grounded committed hypothesis and render, followed by human assessment. That would establish progression through the workflow, not artistic advantage.

## Audit boundary

Verified the authoritative read-only research replay, all 100 registered research artifact hashes, 20 source snapshot hashes and 29 cached request-byte hashes. Current research, context, source and panel implementations match this run's frozen hashes. No implementation or frozen run record was changed, no inference or network acquisition was launched, and no review or training was performed. Only this report and documentation references were added. Artistic quality remains unassessed because no artwork was generated.
