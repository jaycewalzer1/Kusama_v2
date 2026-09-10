# Missing observation bracket — 9 September 2026

## Preserved incident

`studio-404d3b553ae24c5c8cb9` failed at 23:13:23 UTC during its first
caption-withheld image inspection. `response-004.json` omitted the `]` closing
`observations`, then attempted to close the surrounding object. The strict parser
rejected character 873. It received 9,330 input tokens and emitted 173 tokens,
well below the 1,800-token output cap and 16,384 combined limit. This was a
malformed completion, not context overflow or output truncation.

The authoritative ledgers record five Qwen charges (four completed, one failed),
44,063 measured input/output tokens, no unsettled calls, 13 completed source
requests, 18 metadata candidates and one acquired reference, Olowe of Ise's
*Veranda Post*. No hypothesis was committed and no FLUX study was rendered.
All 27 registered research artifacts matched their hashes. The rejected image
observations remain unverified model statements, not accepted inspection or human
judgment. The exact completion and its hash are retained in the portable
[regression fixture](../../tests/fixtures/research/missing_observation_bracket.json).

## Correction for new plans

New `independent-research-v20-json-generation` plans freeze a syntax-only JSON
generation contract. A pinned [LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer)
token filter is passed to Transformers' generation loop. It restricts next-token
choices to a single JSON object, including nested arrays and strings, and blocks
early EOS while that object is incomplete. The strict parser, duplicate-key
rejection, action validation and bounded tool feedback still apply afterward.
This does not repair a saved completion, supply an observation, choose an artistic
action or add an inference call.

The local [adapter](../../src/kusama/json_generation.py) uses the library's core
API because its bundled Transformers integration imports a removed tokenizer
location in the pinned Transformers 5 environment. Tokenizer preprocessing is
cached against actual tokenizer bytes; each response gets a fresh parser.
Environment-dependent parser defaults are overridden by the frozen contract.
Only the latest prefix state is retained for the single-sequence research call,
avoiding accumulation of full-vocabulary allowlists. Some tokens spanning a
string and its following field require splitting into smaller tokens; syntax
filtering changes the sampling distribution and does not establish identical
artistic decisions or model quality.

The original problem, context, source access, 16,384-token allowance, 1,800-token
output cap, call counts and render profile are unchanged. Historical callers
without the frozen response-format field use their original generation path.
The failed run is not migrated or dispatched again. Output-cap exhaustion,
inference errors and malformed responses remain terminal; syntax constraints are
not a guarantee of valid action arguments, grounded observations or completion.

## Verification and preparation

The tests exercise the exact malformed text through a byte-level tokenizer and
the pinned local Qwen tokenizer, preserving Unicode names and escaped quotes,
rejecting a second object and premature EOS. The Qwen test uses explicit
punctuation token boundaries to accommodate the filter's conservative token
segmentation. A separate test runs the real Transformers generation loop with
scripted logits preferring an invalid closing brace and premature EOS. No Qwen
weights or artistic inference are used. Historical strict parsing remains
covered, including rejection when a substitute backend ignores the filter.

`KUSAMA_PROCESSOR_ACCEPTANCE=1 .venv/bin/python -m pytest -q` passed **1,079 tests**
with **five skipped** in 109.92 seconds. The skips require an unmocked live process
or explicitly selected historical archives; the new portable incident/tokenizer
tests ran. Dependency synchronization and whitespace checks passed.

The unexecuted preparation is
`artifacts/preparations/studio-404d3b553ae24c5c8cb9-v20-json-generation/`.
It contains a validated `start-run.json` with a fresh submission ID, the unchanged
resource configuration as a preview, a frozen/preflighted standalone research
plan, full/admitted initial contexts and a provenance receipt. The initial prompt
uses **3,949 input tokens** against the **14,584-token input allowance**. Original
project/brief attribution and the original chance seed are retained in the
standalone plan. No resource session was admitted, no worker was dispatched, and
no Qwen, FLUX or collection-network call was made by preparation. The five original
charges, rejected status, saved files and 27 registered hashes were verified again.
These generated artifacts remain local and are unavailable in a fresh checkout.

Software tests cannot establish reliable live research or artistic improvement.
The next evidence gate is a separately admitted run reaching three grounded
provisional studies and actual-image inspection, followed by human assessment of
the selected-parent revision and its alternatives.
