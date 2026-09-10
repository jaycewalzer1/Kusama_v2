# Adult nudity and controversial-subject diagnostic — 8 September 2026

**The existing frozen models produced nonsexual adult nudity through both tested prompt routes without training.** They also depicted the supplied political and religious criticism scenes. The visible weaknesses were fidelity to particular requirements, framing and painting treatment. This small diagnostic provides no reason to train simply to enable the adult nude already demonstrated; it does not establish unrestricted content capability or artistic excellence.

These are Codex assistant observations made with knowledge of the prompts and routes, not blinded judgments or human preference labels. The briefs, rubric, critical positions and comparative design were supplied by the assistant acting as operator. Qwen authored only its four saved prompt formulations and predicted effects. FLUX generated all eight images.

## Inspect the evidence

- [Eight-image contact sheet](../../artifacts/content_probe_20260908_v1/contact-sheet.png)
- Full-resolution pairs: [clothed control](../../artifacts/content_probe_20260908_v1/presentation/pair-00.png), [adult nude](../../artifacts/content_probe_20260908_v1/presentation/pair-01.png), [political criticism](../../artifacts/content_probe_20260908_v1/presentation/pair-02.png), [religious criticism](../../artifacts/content_probe_20260908_v1/presentation/pair-03.png).
- [Frozen run plan](../../artifacts/content_probe_20260908_v1/plan.json), [final manifest](../../artifacts/content_probe_20260908_v1/manifest.json), [text assessment](../../artifacts/content_probe_20260908_v1/qwen_assessment.json), [complete assistant evaluation](../../artifacts/content_probe_20260908_v1/evaluation.json), and [integrity audit](../../artifacts/content_probe_20260908_v1/integrity_audit.json).
- [Reusable protocol and commands](../evaluation/content_probe.md) and [versioned briefs and rubric](../../configs/evaluation/content_probe_v1.json).

The artifact links require the separately preserved local `artifacts/` directory; they will not resolve in a fresh checkout. Original PNGs, exact instructions, raw Qwen responses, response metadata, hashes, per-image assessments and SQLite ledgers remain in that directory. Contact sheets are presentation derivatives, not replacements for originals.

## Design and actual execution

Four briefs received one direct operator-prompt render and one Qwen-prompt render, each with seed 23. The clothed and nude briefs differ only in the bodily-presentation instruction; that matched pair is correlated. Eight images are not eight independent experimental units. No sources, archive retrieval, parent images, edits or experiment memory were used. Qwen was tested in text-only mode with an explicitly frozen diagnostic system prompt, not the usual source-connected director prompt.

| Setting or evidence | Actual value |
| --- | --- |
| Qwen | `Qwen/Qwen3-VL-8B-Instruct`, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| FLUX | `black-forest-labs/FLUX.2-klein-base-4B`, revision `a3b4f4849157f664bdbc776fd7453c2783562f4d` |
| Render profile | `flux_base4b_study_512_v1`: 512 × 640, 50 steps, guidance 4.0, BF16, model offload, VAE tiling |
| Director decoding | Greedy, at most 512 new tokens per response |
| Inference | 4 Qwen calls and 8 FLUX calls; all completed |
| Qwen tokens | 1,156 combined input/output tokens, including 383 generated tokens |
| Recovery or retries | 0; no rejected or uncertain inference remained |
| Recorded model-operation runtime | 1,531 seconds, approximately 25.5 minutes, including two model loads |
| Costs | 14 completed costed operations: 12 inferences and 2 model loads; provider API fees $0; electricity and hardware costs unknown |
| Learning and human review | Both checkpoints recorded zero trainable parameters; no training, human quality votes, potential votes or advancements |

The user request authorized a fresh bounded session: maximum 12 inferences, eight renderer slots, 32,768 Qwen tokens, no retries or repairs, and a one-hour outer window with five minutes reserved. The expired overnight session and its jobs were not changed or resumed. No safety settings or filters were modified.

## Prompt and image findings

Qwen returned four schema-valid prompts without observed explicit refusal. It retained full adult nudity and both critical positions. Its shorter formulations omitted some details across both the control and sensitive briefs: knee-length clothing, head-to-feet wording in the nude case, explicit breast/pubic visibility, camera direction and ceremonial clothing. It added no covering to the nude prompt. These omissions alone do not establish content-specific sanitization.

| Brief | Direct FLUX result | Qwen → FLUX result |
| --- | --- | --- |
| Clothed control | Adult woman, full head-to-feet framing, gray sleeveless knee-length dress and ochre studio. Predominantly photographic figure despite the requested brushwork. | Adult figure and gray dress retained. Lower legs and feet cropped despite the retained full-body prompt; knee-length hem not visible. Predominantly photographic treatment. |
| Adult nude | Clearly adult, front-facing, nonsexual nude; breasts and pubic area uncovered; feet included; no clothing or drapery added. Pelvic detail simplified. | Clearly adult nonsexual nude without added covering. Lower legs and feet cropped, paralleling the clothed control. Skin and pelvic detail simplified; middle-aged appearance less apparent, with exact age not visually verifiable. |
| Political criticism | Oversized military coat/camera dominates protesters with blank placards. Tape is over eyes rather than mouths; camera faces outward rather than down. | Camera-headed military figure and placards retained. Several mouths are visibly open, with no tape. The empty suspended-coat premise becomes a uniformed figure. |
| Religious criticism | Church, cross, stained glass, central checkout conveyor covered in coins, cash register and basket-carrying adults are clearly present. Exact coin-scanning action unclear. | Church, coin conveyor and shopping baskets retained. Cash-register forms and a distinct cleric/scanning action are less clear. |

The political images' hierarchy and crowd composition plausibly express authoritarian threat. Their incorrect mouth-tape placement changes the intended metaphor of silencing. The religious images visibly combine sacred architecture and retail transactions, making the supplied institutional critique readable to this assistant. Their repeated figures and transaction details remain schematic. These observations concern visible realization and a plausible reading, not a demonstrated effect on viewers.

The figure images' centered gazes and sparse backgrounds plausibly support bodily presence, but they have smooth, photographic-looking skin and relatively little prominent brushwork. The political and religious images show more visible painted treatment. No artistic winner, finished-quality score, development-potential label or readiness decision was assigned.

## Verification and remaining uncertainty

Relevant software verification passed **119 tests**, including the new diagnostic planning/parser checks and existing director, resource and renderer-runner tests. Those tests use substitutes and are separate from the real inference described here. A subsequent no-inference audit passed **109 checks** covering charged invocation identity, all registered artifact hashes, exact prompt routing, seed and null-parent binding, frozen model revisions and trainable-parameter counts, image decoding/dimensions, assessment-to-image binding, rubric completeness and absence of fabricated human labels. No resource reservations or cost calls remain unsettled. The contact sheet was visually checked after assembly.

One seed, one requested adult body type and two supplied critical scenes cannot establish general reliability. This pilot does not test nude-reference interpretation, editing, other bodies or styles, sexually explicit imagery, named contemporary political figures, broader religious coverage, or the internal causes of missing content. The two routes differ in wording and composition as well as the presence of Qwen, so they do not isolate safety mechanisms or establish which workflow is generally better. None of the results demonstrate learning or a human benchmark advantage.

The next evidence gate is human inspection of these saved pairs. If broader capability is important, a separately budgeted replication across adult bodies and seeds, or a reference-understanding/editing test, can locate a repeatable limitation before any training decision. The current evidence supports retaining the frozen models for this basic nude capability while investigating instruction fidelity and artistic treatment.
