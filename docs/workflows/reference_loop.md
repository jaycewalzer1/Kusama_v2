# Frozen reference-informed creative experiments

`kusama.creative` connects a saved brief and a small reference packet to the pinned
`Qwen/Qwen3-VL-8B-Instruct` director, then to the pinned
`black-forest-labs/FLUX.2-klein-base-4B` generator. Both loaders explicitly freeze
all parameters. Nothing in this workflow trains either model.

The complete experiment has four stages:

1. Snapshot the brief, sourced reference text and one to six local reference
   images. Reference image files are preserved byte-for-byte, with SHA-256 hashes.
2. Qwen receives those images and their paired context. By default it proposes
   one direction per model call, for 1–12 distinct directions: a pictorial intervention, a self-contained
   FLUX prompt, a predicted effect and specific relationships to real source IDs.
   Each later call also receives previous titles and the first 300 characters of
   each core intervention, with this excerpt policy recorded in the request.
3. FLUX generates exactly two realizations of each direction, varying only the
   recorded seed. These are **initial generations** with no parent image. The
   reference images condition Qwen; FLUX receives the exact text prompt saved in
   the direction. They are not passed directly into FLUX as image references.
4. The optional frozen Qwen observer sees the two resulting images per direction.
   It records visible observations, separate finished-quality and
   development-potential preferences, uncertainty and one possible revision.
   These are model predictions, never human votes or advancement decisions.

New runs save their images and model observations for inspection in the [development panel](dev_panel.md). They no longer register A/B schedules. Existing review collections and their history remain preserved; see the [retirement record](../operations/2026-09-08_dev_panel.md).

## Reference packet

The packet is a small inference input, not a training corpus. Each reference is
an image with sourced context, or a text-only source. One to twelve references,
including one to six images, are supported. Local asset paths are resolved
relative to the packet JSON. The runner performs no web retrieval: research and
asset acquisition happen before it starts, with provenance and permitted uses
recorded in the packet.

```json
{
  "schema_version": 1,
  "id": "memory-study",
  "title": "Interrupted memory and living continuity",
  "brief": "An original painting about the emotional experience of interrupted cultural memory.",
  "creative_constraints": [
    "Distinguish an artist's formal inspiration from evidence about a different culture.",
    "Do not invent glyph meanings or represent a living culture as extinct."
  ],
  "references": [
    {
      "id": "source-01",
      "title": "A documented work or object",
      "url": "https://example.org/collection/object",
      "provenance": "Who published this source and how the image was obtained.",
      "permitted_uses": "The source's applicable image terms and the intended use.",
      "source_family": "Museum collection record",
      "context": "A sourced paraphrase; distinguish facts from interpretation.",
      "relationship_to_brief": "An explicitly proposed artistic relationship to the brief.",
      "image_path": "images/object.jpg",
      "maker": "Known maker, or omit",
      "date": "Documented date, or omit",
      "medium": "Documented medium, or omit"
    }
  ]
}
```

`image_path`, `maker`, `date` and `medium` are optional. Context is limited to
1,800 characters per reference, relevance to 1,000, brief to 5,000 and each
creative constraint to 800. Keep a real packet substantially below these maxima:
Qwen rejects an actual input over 8,192 tokens without truncating it. By default,
each image is converted to RGB with EXIF orientation and proportionally resized
to at most 262,144 pixels **for inference only**. Source bytes stay unchanged;
original and processed dimensions are saved in every Qwen response record.

An optional `composition_challenges` object supplies an operator-authored
composition scaffold, keyed by `d01` through `d12`, with up to 800 characters per
challenge. For example, a one-direction run can supply
`{"d01": "Make absence legible through a panoramic empty landscape, without a central figure."}`.
When any challenges are supplied, every direction requested by the run must have
an assignment; missing coverage is rejected before a job or model call is created.
For a 12-direction experiment supply all 12 IDs.

The complete challenge map remains in the immutable packet. Each Qwen request
receives only the current assignment immediately before its final instruction,
alongside explicit operator-authored provenance. With scaffolding, earlier
directions are represented only by IDs and titles to reduce compositional
anchoring. The full earlier descriptions still participate in duplicate checks.
Qwen formulates realizations of the assigned challenges; it does not receive
credit for inventing the operator's compositional diversity. Without scaffolding,
the existing title-plus-core-excerpt behavior is preserved. This mechanism guides
semantic diversity but does not certify it; inspect the proposed compositions
before committing to expensive generation.

References are treated as evidence, not executable instructions. The director
prompt distinguishes proposed artistic analogies from documented relationships
and prohibits invented cultural or historical facts. These prompting constraints
reduce avoidable mistakes but do not verify model claims or guarantee cultural
accuracy. Human source checking and artistic judgment remain necessary.

## Running and resuming

First save all Qwen directions without loading FLUX:

```bash
.venv/bin/python -m kusama.creative \
  --packet references/maya_memory/packet-v3.json \
  --directions 12 --batch-size 1 --realizations 2 --observe --stage direct \
  --width 512 --height 640 --steps 50 \
  --device mps --precision bf16 --offload model --local-files-only
```

Use the directory printed by that command for the remaining stages:

```bash
.venv/bin/python -m kusama.creative --resume artifacts/PRINTED_RUN_ID --stage render
.venv/bin/python -m kusama.creative --resume artifacts/PRINTED_RUN_ID --stage observe
```

`--stage all` runs all requested stages in order. `--directions 3` runs a smaller
3 × 2 experiment. Width/height/steps are recorded choices, not quality guarantees.
Larger final works can follow an explicit human selection.

The default `--batch-size 1` gives a 12 × 2 experiment with observer a budget of
48 invocations: 12 director calls, 24 FLUX calls and 12 observer calls. Batch sizes
2 and 3 remain configurable and are frozen in a new run's specification. A real
three-direction pilot returned extra directions and reached its output-token
limit, so one direction per call is the operational default. Count and ID checks
still reject extra output; there is no guarantee an unconstrained model always
follows the JSON contract. `--note` saves an explanation for a new experiment,
including a superseded pilot ID when appropriate. Failed pilot evidence remains
unchanged; changing batch size requires a new run.

Qwen is closed before loading FLUX; FLUX is closed before reloading the observer.
The models never deliberately coexist on shared GPU memory. No fallbacks,
automatic retries, model substitutions or prompt repairs occur.

`--resume` permits stage selection and explicit saved-response revalidation; all model inputs, seeds and settings
are frozen. It uses saved packet assets even if original source files disappear.
Every recorded immutable artifact is checked before work resumes. A valid saved
response or image is recovered without another model invocation, including a
crash between saving bytes and registering them. A reserved call without saved
bytes becomes **uncertain**, and a malformed saved Qwen response stays rejected.
Neither is automatically repeated. Start a separate experiment to change inputs
or explicitly try a different proposal; retain the failed run as evidence.

If a parser implementation incorrectly rejects otherwise complete model JSON,
`--resume RUN_DIRECTORY --stage direct --revalidate-saved` explicitly evaluates
the original raw text against the current validator. It saves a separate
`.revalidated.json` sidecar containing the original response hash, original error,
validator code hashes and validated output. The original rejected response stays
byte-identical and registered. This performs no model call, text repair or prompt
substitution. Ordinary subsequent resumes consume the linked sidecar after
checking its hash and binding. The prompt requests concise prose while validation
allows normal model verbosity: core interventions up to 4,000 characters,
expected effects and FLUX prompts up to 2,000, and each source relationship up to
1,000. The entire raw response remains capped at 16,384 characters; FLUX also
enforces its actual tokenizer limit without truncating the saved prompt.

Explicit revalidation also permits a complete single JSON object followed only
by surplus closing `]` / `}` delimiters and whitespace. The sidecar records the
exact ignored suffix and the versioned parser policy. Extra objects, data,
commentary, opening delimiters, prefixes and duplicate keys
remain rejected. Normal inference parsing remains strict. This narrowly scoped
recovery never changes the original response bytes or relaxes field, count, ID,
source or duplicate-direction validation.

One separate observer-specific syntax policy handles a missing `]` between two
complete `visible_observations` objects and the next top-level `finished_quality`
field. It parses the two entries structurally, inserts exactly one array closer
into the parsing input, and then requires strict whole-document JSON parsing and
the complete observer validator. The sidecar records the insertion text and its
Unicode-codepoint offset. It cannot supply a missing candidate, repair an
incomplete entry, match a pattern inside a string, or fix any other missing
delimiter. Original model bytes and all descriptive claims—including mistaken
claims—remain unchanged.

The observer similarly requests one or two short observations per image and
concise reasons. It permits up to four observations per image and 2,000 characters
per descriptive field, including revision text. Exactly two bound candidate IDs,
their order, the four preference values and one candidate-bound revision remain
strict. The global response cap and model output-token budget are unchanged.
An observer response rejected only by an earlier descriptive bound can be
revalidated with `--resume RUN_DIRECTORY --stage observe --revalidate-saved`;
saved images and the original observer response are reused without inference.

A response missing an entire candidate observation cannot be repaired by parser
revalidation. An explicit bounded retry can instead request new model content:

```bash
.venv/bin/python -m kusama.creative \
  --resume artifacts/PRINTED_RUN_ID --stage observe \
  --retry-rejected-observers 1
```

This operator flag authorizes at most one additional invocation for each pair
whose original observer response is saved, registered and schema-rejected.
Original responses, failed attempt records and costs stay unchanged. A reserved
invocation with missing output or an inference error is ineligible. A failed retry
is never repeated automatically; saved valid retry bytes recover after a crash.

Retries use a separate `RUN_ID_observer_recovery` job in `observer_recovery/`,
with a budget of one call per direction. Its specification binds the original
job, packet hash, candidate image hashes, frozen director settings and explicit
operator flag. The original run's specification and budget remain immutable.
Successful observations record their owning job/attempt, nested response path and
a `supersedes` link to the rejected original. The full observer prompt and its
two-candidate schema/example are retained in the new response metadata. No
missing observation is invented, and images are never regenerated by this path.

For a 12 × 2 run with one rejected original observer, completion uses 48 original
reservations (including that rejection) plus one recovery reservation: 49 actual
invocations. The explicitly authorized maximum is 60 across the two budgets;
unused recovery slots need not run. `model_calls` and `planned_calls` retain the
primary job counts, while `total_model_calls`, `authorized_maximum_model_calls`,
`observer_recovery` and `cost_accounting.run_ids` expose the combined scope.
The recovery budget and saved evidence persist across resumes. A successful
saved retry is reused without another invocation; starting a new retry requires
the retry flag on the current invocation. Omitting that flag never spends an
unused recovery slot.

Schema failures can also be the result of the diagnostic experiment. To finish
the original observer pass while preserving those failures, use:

```bash
.venv/bin/python -m kusama.creative \
  --resume artifacts/PRINTED_RUN_ID --stage observe \
  --revalidate-saved --continue-rejected-observers
```

This continues to the next pair after a retained, registered schema rejection,
including an exhausted failed retry. Existing valid retries are reused; no new
retry is started by this command. Missing responses, uncertain reservations,
inference failures and integrity failures still halt. The explicit revalidation
flag also applies to a newly saved response during this pass using only the
documented bounded syntax policies and full schema validation.

Only accepted model outputs appear in `observations.json` and the manifest's
`observations`. Every unresolved rejection, and every original rejection later
resolved by a valid retry, appears in `observer_rejections.json` and the
manifest's `observer_rejections`, with its owning job, attempt, response path,
hash and parse error. These are diagnostics, not substitute model observations.
`observer_status` records attempted, valid and rejected pair counts, unresolved
direction IDs, and the number of rejected responses. Its status is `complete`
when every pair has a valid output, or `partial` when all original slots were
attempted but one or more pairs remain unresolved. The run can then reach
`needs_human_review` with a partial observer pass, including zero valid model
observations. Human reviews and advancement remain independent decisions.
The explicit continuation flag is recorded in the session metadata. A completed
partial pass resumes by reusing its saved evidence without new inference.

Before rendering, the runner reconstructs the ordered direction list from saved,
reserved Qwen responses and rejects an aggregate that differs. This also recovers
a legitimate `directions.json` saved just before a registration crash. Candidate
output paths must resolve inside the run directory before a model is loaded.

## Saved evidence and cost

Each run saves `source_packet.json`, normalized `packet.json`, exact reference
assets, each Qwen request/prompt/raw response, `directions.json`, each FLUX PNG,
`candidates.json`, optional observer responses, model/configuration provenance,
seeds, runtime/memory measurements, a SQLite-backed event history and a manifest.
Candidate records link the direction, exact generation prompt, reference packet
hash and proposed source relationships. Initial candidates have null parent
image fields; later edits should link their explicitly selected parent.

The ledger reserves every Qwen and FLUX invocation before it can run. Cost events
record each model load and inference separately through `kusama.costs`. Local
provider API cost is zero; energy and hardware totals remain unknown until
explicit assumptions/rates are configured. The manifest gives the shared cost
journal location and run identifier, so the dashboard can show that distinction.

Qwen's distinctness contract rejects identical titles, core interventions and
FLUX prompts, including repeats across batches. Semantic distinctness, artistic
success, faithfulness to the brief and the caliber of the result still require
human review. The saved observer is useful for comparing model judgments to
those human reviews, without converting a model prediction into a label.
