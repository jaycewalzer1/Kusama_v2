# Repeated JSON failure and live follow-through — 9 September 2026

The first syntax-filter release did not establish live reliability. The user then
ran `studio-7f9e54f2551d4f7480a0`; its first response exhausted all 1,800 output
tokens inside a repetitive `reason` string. It used 3,952 input tokens, so the
16,384 combined allowance was not exceeded. The raw response, failed charge and
original plan remain preserved. Its exact text is retained in
[runaway_reason.json](../../tests/fixtures/research/runaway_reason.json).

The v20 generation filter constrained JSON syntax without enforcing the existing
1,000-character reason limit. Its freetext optimization also excluded some valid
tokens crossing from a string into another field, including `.\",\"`. The previous
tokenizer test accommodated this by splitting punctuation, which failed to test
the model's actual preferred token segmentation. Neither the fixture-only suite
nor the prepared replacement demonstrated a successful live completion.

New `independent-research-v21-bounded-json-generation` plans compile their frozen
action schema and current required-field display into generation constraints.
Required arguments precede the reason, whose existing maximum is enforced while
generating. The current allowed actions become schema alternatives; final
Pydantic and tool validation still apply. A supplementary token trie restores
legal string-boundary tokens omitted by the library shortcut. The regression
now uses the actual pinned Qwen tokenization of the original missing-bracket
completion, without punctuation splitting, and blocks its offending brace.

This is operator-authored serialization control. It adds no artistic content,
repairs no old response and changes no frozen budgets or weights. Exhausting the
output cap remains possible for large responses; malformed or incomplete output
is still preserved and rejected. Software validation does not establish useful
source choices, grounded interpretation or artistic quality.

The user explicitly authorized Codex to keep executing fresh bounded attempts and
monitoring until the failure is resolved. The first v21 attempt is
`studio-c90d2f52d80346cab283`, under the original 42-Qwen-call, six-FLUX-attempt,
120-minute contract and named 512 × 640 profile. The request records this chat
authorization instead of incorrectly claiming an operator pressed the UI button.
No original charged invocation was replayed. Live outcomes will be recorded after
the worker settles; an active worker is not a passed acceptance gate.

The first attempt settled at **00:14:49 UTC on 10 September** (9 September local
time) with eight valid JSON responses, zero acquisitions, zero hypotheses and no
images. It ended `stagnated` after repeating a search. Candidate reasons also
rejected historically objectifying representations for not already expressing
agency, confusing a source's usefulness to inquiry with the desired outcome.
This establishes the v21 serialization path across eight live calls, not a
successful artistic run.

New v22 plans retain the same limits and correct that operator guidance: useful
sources can contribute troubling evidence, contradictions or formal possibilities;
they need not already solve the problem. New payloads disclose unused prepared
queries, and generation cannot repeat a completed search. Nested query and image
inspection schemas expose requirements already checked by the tools. No source,
score, observation or artistic proposal is supplied by these changes.
The second explicitly authorized attempt is `studio-9b23642cceb241d495f2`, with
the original operator problem and per-run limits.

That second attempt failed at **00:23:14 UTC** after six charges. Its sixth
response tried to assess 17 candidates and exhausted the 1,800-token cap inside a
nested explanation. The source excerpt shown for *Veranda Post* also cut off
before the description of the senior wife's role. The rejected assessment is
retained as an unaccepted model completion, not scored evidence.

V23 caps assessment batches at six candidates and each candidate reason at 240
characters, in the frozen generation schema and bounded tool feedback. It also
explains the existing metadata `open`/`read` path so truncated cards do not have to
serve as complete source evidence. It does not select a preferred reference.
The third authorized attempt is `studio-bf964c1eddc14d2eb296`, under the same
per-run limits. Nineteen focused tokenizer/contract tests passed before dispatch.

The third attempt failed at **00:32:12 UTC** after five charges. Its assessment
list had five bounded entries, but it emitted 3,163 newlines before reaching the
output cap instead of continuing the object. V24 reduces consecutive formatting
whitespace to two characters and permits only EOS after a complete object.
Tokenizer replay of the exact saved completion blocks the whitespace loop at
token 449 of 1,800 while leaving the required comma available. The source policy
also makes the first search follow the model's own first prepared query, rather
than exhausting the budget on secondary queries before the primary subject.
Forty-one focused tests passed. The fourth authorized attempt is
`studio-f760acd764984553a2d1`, with unchanged per-run execution limits.

The full `KUSAMA_PROCESSOR_ACCEPTANCE=1 .venv/bin/python -m pytest -q` suite passed
**1,080 tests**, with **five skipped**, in 121.48 seconds. The skipped cases require
an unmocked acceptance run or explicitly selected historical archives. The new
Qwen-token and bounded-reason regressions ran. The first live v21 response supplied
the required analysis and queries in 218 tokens; this establishes one accepted
live action, not completion or artistic quality.

The fourth attempt ended `partial` after **11 valid Qwen calls**, with no acquired
references, hypotheses or studies. V24's full processor-enabled suite passed
1,084 tests with five skipped. Serialization alone did not resolve the run.

V25 (`independent-research-v25-rich-candidate-pages`) freezes a new context policy:
try smaller provider-interleaved candidate pages with complete descriptions before
historical excerpt projections. Duplicate candidate descriptions become explicit
pointers; deferred cards retain IDs and metadata-open instructions, and each page
retains its saved-result recall cursor. This adds no source ranking or artistic
choice. Historical projection stages and frozen requests remain unchanged.
Reconstructing the fourth run's decision 002 with this policy admits at **11,723
input tokens** within the unchanged 14,584 input allowance, retaining the complete
900-character *Veranda Post* description. The former admitted prompt cut it to
240 characters, excluding the senior wife's political role. This processor-only
check used saved evidence, no model call or new network acquisition.

V25's full processor-enabled suite passed **1,086 tests**, five skipped, in
129.46 seconds. The fifth authorized fresh attempt, `studio-c581d6c941cf409f950d`,
started at **02:08:33 UTC on 10 September** with the original problem, same per-run
limits and profile. It is recorded separately from all preserved earlier calls.

The fifth attempt ended `failed` at its frozen wall-time boundary after **eight
completed valid Qwen calls / 92,276 actual tokens**, with no unsettled charges,
acquisitions, hypotheses or studies. Local `pmset` logs show repeated sleep periods
during execution; one call spanned 02:31–03:52 UTC. Its clock was not extended and
none of its calls will be replayed. Across the five fresh troubleshooting attempts:
**38 Qwen calls (36 completed, two rejected), 376,370 actual tokens, zero FLUX
calls and zero studies**. V24 and V25 together supplied 19 consecutive valid
responses, but the full artistic-process gate remains unmet. The model's narrow
source interpretation remains unresolved alongside machine availability.

A read-only preservation audit verified **207 registered research artifacts** across
the two originally reported runs and five new attempts, with no hash mismatches.
[Accounting](../../artifacts/diagnostics/json-generation-live-v21/attempts-settled-accounting.json)
and [preservation](../../artifacts/diagnostics/json-generation-live-v21/preservation-final.json)
are local sidecars. Two separately attributed ARTIC diagnostic HTTP requests tested
query handling: an impossible term returned zero results; the second timed out and
was not retried. Neither dispatched model inference.

The [next request](../../artifacts/diagnostics/json-generation-live-v21/next-request.json)
preserves the original problem and limits, with a new submission ID. The active
`finish-kusama-run-recovery` task follow-up checks every 15 minutes. It waits for
external power and an awake/open machine before starting a fresh worker, prevents
idle sleep only for that worker's lifetime, and retains the same per-run limits.
It must inspect actual studies and settled ledgers before declaring completion;
new failures require concrete diagnosis and tests, not blind reruns. This follow-up
is continuation of the user's explicit monitoring authorization, not a successful
run or a human quality judgment. No weights, human votes or old frozen inputs changed.

At the 10 September 16:24 UTC follow-up, local power checks reported AC power and
an open lid, with no active studio worker. Both panel and research source identities
matched the tested v25 implementation. Fresh attempt six,
`studio-50f8cd3ff7564841adc7`, started with the prepared request and unchanged limits.
A verified `caffeinate -i -w` assertion belongs to that worker's lifetime only;
persistent power settings and lid-close behavior were not changed. The first real
analysis completed successfully; this is an in-flight record, not process acceptance.

Attempt six settled at **16:35:40 UTC** after seven valid completed Qwen calls /
73,793 actual tokens, with no acquisitions, hypotheses or studies. Qwen identified
a potentially useful source, but tried to draft twice from its search card before
acquiring or opening the image. Both drafts failed the existing grounding validator;
the bounded run ended `stagnated`. The preserved responses establish neither image
inspection nor accepted artistic instructions. Idle-sleep prevention ended with the
worker. The six attempts now total 45 Qwen calls / 450,163 actual tokens and zero FLUX.

V26 (`independent-research-v26-draft-evidence-gate`) freezes a prerequisite gate for
new plans: drafting is unavailable until source pixels have been acquired/opened
and inspected. It changes the available action contract, generation variants and
tool admission together. It supplies no source choice, description or hypothesis,
and leaves historical contracts and all charged outputs unchanged. The new regression
covers empty evidence, acquired-but-uninspected evidence, the inspection action and
subsequent availability of drafting. Full processor verification must pass before
another actual run.

The exact attempt-six pre-draft context, reconstructed in a disclosed sidecar,
now excludes draft generation and retains acquire/inspection actions. Final v26
processor-enabled verification passed **1,087 tests, five skipped**, in 100.85 seconds.
An earlier suite invocation hit the intended frozen-identity check while an initial
Next-message correction was being saved; it was rerun only after source edits settled.
The seventh authorized attempt, `studio-116416c64b4345d589f7`, is running with the
unchanged original request, a new submission ID, verified AC/open-lid conditions and
worker-scoped idle-sleep prevention. The follow-up now points to the latest numbered
attempt record instead of assuming the earlier timeout is still the newest run.

Attempt seven settled at **17:00:14 UTC** after six valid completed Qwen calls /
41,659 actual tokens, with no acquired reference images, accepted hypotheses or
studies. It opened metadata, then tried to acquire an unscored candidate with its
justification only in `reason`. The existing relevance gate requires the separately
attributed `relevance_override` field; rejection consumed its remaining stagnation
allowance. The seven attempts total **51 Qwen calls / 491,822 actual tokens**, zero
FLUX calls. All raw responses and settled charges remain preserved.

V27 (`independent-research-v27-acquisition-contract`) exposes current-question
eligibility and compiles the existing conditional acquisition requirement into the
generation schema. Eligible candidates can use ordinary acquisition; others require
a nonempty `relevance_override` of at most the existing 700 characters. No score or
justification is supplied. Exact pinned-Qwen tokenizer replay of the saved attempt
seven completion blocks its omitted override at token 40 (`reason`). Low/unscored,
empty-override and eligible-candidate tests exercise the actual token constraint;
question changes invalidate stale eligibility. An initial full suite caught a
reporting-path NameError in the patch; it was corrected before any new live run.

Final v27 processor-enabled verification passed **1,092 tests, five skipped**, in
102.22 seconds. Attempt eight, `studio-901a68c896cc482b8262`, started with unchanged
request settings, a new submission ID, verified external power/open lid and a
worker-scoped idle-sleep assertion. It is active; no completed-process claim follows
from the software gate. The scheduled follow-up reads the latest attempt and test
record rather than relying on a hard-coded old run/version snapshot.

Attempt eight settled at **17:27:33 UTC** after **seven completed Qwen calls /
43,062 actual tokens**, one acquired/inspected chance reference, three rejected
drafts and no committed hypotheses or studies. The eight attempts total **58 Qwen
calls / 534,884 actual tokens**, zero FLUX calls. Qwen's final claim of exhaustive
search is unsupported: no prepared-query search followed the chance bootstrap.
The user-reported attribution failures and the prospective v28 software correction
are recorded in the [grounded draft incident](2026-09-10_grounded_draft_contract.md).
That repair performs no additional inference and preserves the stopped run.
