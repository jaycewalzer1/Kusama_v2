# Expanded research context — 9 September 2026

The user requested a higher token limit after the [context-capacity repair](2026-09-09_context_capacity.md). New `independent-research-v19-expanded-context` plans default to **16,384 combined tokens per Qwen call**, twice the prior 8,192. With the unchanged **1,800-token output reservation**, **14,584 tokens** are available for input, including images.

The pinned local Qwen3-VL-8B-Instruct configuration at revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` declares `max_position_embeddings: 262144`. The selected 16,384 allowance is below that architectural maximum; it does not establish throughput or memory performance for a full model run on this machine.

## Frozen limits and resource accounting

- New research requests freeze `total_token_limit`; their director config records `max_input_tokens`. Exact processor admission, generation and global call reservations use that same value.
- The panel gets the allowance from its start-options API and submits it with the request. Research, actual-image inspection, development choices, geometric diagnostic assessments and reopened study research inherit the frozen value.
- An older full-run form that omits the explicit token allowance must refresh before starting; it cannot silently authorize a larger budget than it displayed.
- The default panel still permits 42 Qwen calls and six FLUX attempts, now with a 688,128-token Qwen allowance. Its largest permitted settings allow 91 Qwen calls / 1,490,944 tokens. The resource validator therefore permits at most 2,000,000 total reasoning tokens and 16,384 per call. The standalone overnight configuration defaults to 1,000,000 total tokens and 16,384 per call; this file does not itself admit or launch a session.
- Old director configurations without the new field retain the historical 8,192 allowance. Historical report verification reads the limit from each frozen plan. Saved call recovery checks the exact original reservation, and mismatched reservations cannot trigger another dispatch. The separate legacy development runner explicitly retains its 8,192 research contract.
- New-plan defaults do not modify existing resource sessions, frozen requests, raw responses, failures or charges. No costly inference, training or replay was launched for this change.

Candidate paging, complete saved-result recall and uncharged capacity checkpoints remain active. Increasing the allowance does not remove admission checks, shorten the operator's problem or turn omitted source text into a relevance judgment. Response format validation and the renderer's independent token limit remain unchanged.

## Verification and evidence boundary

`KUSAMA_PROCESSOR_ACCEPTANCE=1 .venv/bin/python -m pytest -q` passed **1,069 tests**, with **five skipped**, in 97.63 seconds. The skips require a live unmocked run or explicitly selected historical archives; the portable saved-overflow replay ran. Generated TypeScript contracts, changed documentation links and whitespace checks passed. The prepared plan's implementation identity still matched after verification.

The cached pinned processor admitted the saved 21-candidate incident context at **12,917 input tokens**, preserving all 21 candidate cards at projection stage 2. The same fixture still passes the historical 8,192-token path using candidate paging. Boundary tests admit exactly 14,584 input tokens, reject 14,585 before generation, preserve legacy director defaults and prevent a settled call from being recovered with a different reservation. A full scripted study/revision cycle checks that all 21 reasoning reservations use 16,384. The maximum panel configuration also passes admission.

The diagnostic receipt is in `artifacts/diagnostics/studio-b70366976547419586e8-expanded-context/`. These generated files are local and unavailable in a fresh checkout. Verification made no model or network calls. Tests use scripted generation and cached processors; they cannot establish improved model compliance, artistic quality or long-context runtime. The next artistic evidence gate remains a newly admitted run reaching distinct provisional studies, actual-image inspection and human assessment of the resulting revisions.

Read-only verification retained the incident's **14 registered research artifacts**, **two completed calls** and **8,083 measured tokens**, with the original 8,192 reservations. A replacement request, session configuration and v19 research plan were prepared under `artifacts/preparations/studio-b70366976547419586e8-v19-expanded-context/`; no resource session was admitted and no worker started. They preserve the original artistic problem, context, source scope, call counts and renderer profile. The replacement deliberately retains the original absence of chance leads; the current panel's general default remains two.

Both existing dashboard processes were refreshed after confirming no active studio worker. Read-only start-options and contracts requests verified the 16,384 limit. Open browser pages must refresh to load the new budget display.
