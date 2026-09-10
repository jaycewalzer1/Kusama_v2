# Image-led research and development cycles — 9 September 2026

The user requested that the panel's fixed research → render → one-follow-up sequence become a bounded process in which seeing the work can reopen inquiry and change the next experiment. New plans use `independent-research-v15-image-led-inquiry` and `research-render-v5-development-cycle`. This extends the [provisional-study protocol](2026-09-09_studies_before_commitment.md); its initial three-study gate remains.

## Behavior

After all planned initial images are realized and inspected, frozen Qwen chooses a next action from the actual saved branches. `continue` edits any one branch; `combine` uses the first selected branch as the exact parent and other selected images as references; `reformulate` can start fresh or use saved branches. Every new image is inspected before the next choice. Earlier branches remain selectable, including after later revisions.

`research` selects an actual branch and supplies a new question prompted by its observation. The existing independent research engine receives that exact parent, the original operator problem/context, the model-authored working question and an attributed observation excerpt. The full development context is preserved alongside the request. It searches within frozen source and call caps. Its saved findings and explicitly selected image prompts return to the next development choice; its hypothesis does not automatically trigger rendering. An empty completed search remains unresolved evidence. Rejected inference ends the path without an implicit retry.

`review` pauses for human assessment; `stop` or `abandon` ends the cycle. These actions can happen early. A final terminal choice follows the action budget; it can retain a saved branch or no selection. A review request records no human judgment, advancement or preference. Actual human feedback and any subsequent continuation still use separately attributed existing workflows.

Initial study planning and rendering remain a batch. The implemented change opens research and repeated development **after** initial image inspection. Execution remains serial under the shared workload lock; nonlinearity comes from revisiting branches and alternating research with making.

## Frozen limits and records

The panel exposes development actions (default 3, range 1–6) and the action allowance for one optional reopened research session (default 8, range 4–24). Let `I` be initial images, `S` initial research actions, `D` development actions and `R` reopened research actions. The conservative ceilings are:

- Qwen: `S + I + 2D + 1 + R` calls, each reserving at most 8,192 combined tokens.
- FLUX: `I + D` attempts at the selected unchanged render profile.
- Online sources: initial 48 requests/12 acquisitions plus reopened 16 requests/4 acquisitions.

Defaults therefore reserve **42 Qwen calls (344,064 tokens) and six FLUX attempts**, up to 48 total model calls. Research consumes a development action, so not all independent ceilings need be reached together. The default window remains 120 minutes with a five-minute reserve. `I + D` must be at most 12 so all artwork branches fit the bounded inspection set. At most two reference images accompany a whole-image render; exact parent and reference ordering are preserved.

Full-run HTTP submissions must explicitly include `development_steps` and `development_research_steps`. An old open form without these fields is rejected with a refresh instruction before any run directory or worker is created, so its previously displayed smaller budget cannot silently authorize the expanded defaults.

The explicit API/CLI `development_cycle` policy additionally supports zero to two research sessions within its action cap; omitted/null preserves the previous one-follow-up study path. `study_development: false` remains a rendering diagnostic. None of these settings migrate existing frozen runs.

`render/study_development/` stores immutable decision/response inputs, cycle transition receipts, research receipts, exact processor admissions and its SQLite job ledger. Reopened research is nested under `study_development/research/cycle-NN/`; new images and render ledgers are under `render/cycles/step-NN/`. Each image has a distinct branch ID and exact parent hash. The overview maps the repeating inquiry; its dedicated Development tab shows the action history, retained branches, research outcomes, observations and review request. Branch-parent links revisit exact saved images, and research decisions open their corresponding nested evidence. The image gallery and research views also discover the nested records. Cost scopes include every admitted cycle render/research ID.

The UI reads frozen cycle settings, including queued panel requests, and labels model selections separately from human assessment. Earlier runs retain their original stage display. Expanded decision details and unchanged branch thumbnails survive polling. Fixture runs are explicitly labeled on Overview and Development. The workflow routes navigate saved evidence; they do not dispatch new work.

Following the user's layout correction, the workflow map and branch views are confined to `/#runs`. New run keeps the plain question form without a workflow diagram.

Branch observation excerpts have disclosed field limits; full responses remain saved. Original operator constraints, branch images and executable rendering instructions are not silently shortened. Exact processor admission occurs before model loading. Resume reconstructs decisions from saved responses and verifies research, source and render evidence. Missing charged responses require reconciliation, without another dispatch. Stop checks apply before every new admitted action. Failures survive.

## Verification and next evidence gate

Fixture checks exercise research after image inspection, findings returning to a decision, revisiting older parents, combining branches with source images, fresh reformulation, early review, exhausted budgets, empty search, stopping between render and inspection, replay, and tampered/missing evidence. The main panel and reopened-research integrations use the current grounded research protocol with substitute models and source access.

Final combined verification: `KUSAMA_PROCESSOR_ACCEPTANCE=1 .venv/bin/python -m pytest -q` passed **1,038 tests**, with **five checks skipped** because they require separately supplied live-process or retained-incident evidence. The first ordinary suite exposed a malformed historical-plan display regression; it was corrected before this passing run. Exact cached-processor checks passed without model inference, including an additional stress probe with ten artwork branches plus four source images at 2,454 input tokens against a 6,392-token allowance. Generated TypeScript contracts, JavaScript syntax/behavior, and 177 non-artifact documentation links passed. This is not the separate live-process acceptance gate.

Both existing local dashboard servers (ports 8765 and 8878) were refreshed after checking that no panel worker was active. Their live API contracts expose the new defaults. A browser check of the main form verified the development/research controls, six-image ceiling, 344,064-token allowance and source limits. No Start run submission was made.

The UI follow-up passed 58 focused panel, dashboard, live-polling and contract checks. JavaScript checks cover branch ordering and parent links, exact reopened-research navigation, retained image elements and expanded observations during polling, explicit fixture/review attribution, queued/incomplete states and the historical stage fallback. A temporary browser preview of a clearly labeled engine fixture verified the loop layout, five retained branches and navigation from a combination to its saved parent and from a research decision to its source session. The preview used scripted responses and substitute images; no run was submitted through the panel. Six local links in the changed workflow/operation guides and the whitespace check passed.

These are software and input-routing results. This implementation performs no real Qwen/FLUX inference, external source acquisition, human review or training. The next artistic gate is a separately admitted run in which actual seeing changes a research question or artistic action, followed by independent human assessment of the selected revision and retained alternatives. More flexible orchestration has no established artistic advantage yet. Local generated evidence remains outside Git and requires its separately preserved artifact root in another checkout.
