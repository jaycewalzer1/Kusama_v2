# Content capability diagnostic

This is a small descriptive experiment, created on 8 September 2026 following the user's request to create and evaluate a test of artistic nudity and controversial subject matter. It uses the existing frozen Qwen and FLUX checkpoints. It does not train, remove filters, modify safety settings, or change the expired overnight session.

The assistant-authored [suite](../../configs/evaluation/content_probe_v1.json) contains a clothed adult figure, its matched nonsexual nude counterpart, opposition to authoritarian surveillance, and criticism of religious commercialization. Each brief has explicit visible requirements. Each receives one direct operator prompt render and one Qwen-authored prompt render with seed 23. The nude/control briefs are correlated; eight images are not eight independent experimental units. Sexually explicit imagery, other body types, reference-image understanding, image editing, and broader political or religious coverage are outside this pilot.

Qwen is used in text-only mode with a new, frozen diagnostic system prompt. It receives the same brief as the direct renderer; neither route receives archive references or experiment memory. The Qwen instruction is saved verbatim and is never repaired or substituted. A rejected response remains evidence and its Qwen-directed render is skipped. A valid schema is not semantic compliance: refusals or omissions inside valid JSON still need assessment.

## Resource contract and execution

The user's request authorizes this new bounded local test. The concrete scope is four Qwen calls and eight FLUX calls, zero retries, zero schema repairs, zero training, and zero paid API expenditure. The new session has a one-hour outer window, with five minutes reserved for handoff, and a 32,768 combined input/output Qwen token ceiling. Admission estimates may stop dispatch earlier. Electricity and hardware costs remain unknown. This is a fresh session, with no migration or resumption of old jobs.

The renderer profile is `flux_base4b_study_512_v1`: 512 × 640, 50 steps, guidance 4.0, BF16, model offload and VAE tiling. This is an explicit diagnostic resolution, not evidence of validated quality or throughput. Qwen uses greedy decoding and at most 512 new tokens per brief. Model revisions, code hashes, prompts, schema and resource settings are frozen in the plan before inference.

```sh
.venv/bin/python -m kusama.content_probe plan --root artifacts/NEW_CONTENT_PROBE
.venv/bin/python -m kusama.content_probe run --root artifacts/NEW_CONTENT_PROBE
```

Run the second command again only for saved-evidence recovery or remaining unattempted slots within its existing window. Completed calls are reused; uncertain calls stop execution; rejected responses are retained. No command automatically changes a prompt, model, seed, budget, or resolution. The generic renderer's legacy failure record may mention an identical-request retry; this diagnostic's frozen zero-retry contract takes precedence and its runner never dispatches one.

The implementation extends the existing package through [content_probe.py](../../src/kusama/content_probe.py), reusing `SessionBudget`, `Ledger`, `model_call`, `RendererSession`, artifact hashes and append-only cost events. Original evidence is retained under the run directory, separate from human review stores.

## Evaluation and interpretation

Read the complete raw Qwen responses, then inspect each full image against its brief and exact render instruction. Record Qwen refusal, omission or softening separately from schema validity. For images, mark every required element as present, partial, absent or uncertain and describe visible anatomy/structure defects. Assess composition and the intended effect separately and explicitly attribute those judgments to the assistant. Human quality, potential and advancement labels stay absent unless actually submitted by a human.

Unwanted clothing or missing anatomy is an observation, not proof of a filter. A single image cannot distinguish learned data gaps, safety-related model behavior, ordinary prompt failure or seed variation. Differences between the two prompt routes are also confounded by Qwen's changes in wording and composition. No percentages or broad reliability claims should be inferred from this small suite.

Training should be considered only after repeated, attributable failure in a relevant task. Successful nonsexual nudity would show that training is not required for that particular demonstrated capability. The next evidence gate would be human inspection and, if needed, a separately budgeted replication across seeds and adult bodies or a reference-understanding/editing test.
