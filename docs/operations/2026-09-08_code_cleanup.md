# Code maintenance — 8 September 2026

This pass preserves the pre-existing uncommitted implementation and documentation work. It changes software maintenance and resource handling; it makes no model calls, trains no weights and writes no human judgments or generated run evidence.

## Changes

- Removed 33 unused imports across the Python source. Retained the controlled executor's `verify` and `repair_metrics` hooks as explicit re-exports because the execution adapter uses them.
- Replaced repeated CLI dispatch branches with one lazy command registry, also used by help. All 20 delegated command routes retain their original targets and argument semantics; doctor and export retain their separate handlers.
- Closed short-lived SQLite connections in retrieval, evidence export, dataset extraction, session snapshots and comparison accounting. Index writes still commit before file hashes are recorded. Read-only connections remain read-only.
- Reused the existing streaming SHA-256 implementation behind `evaluation.file_hash`, preserving its string/path interface and digest values. JSON serialization and immutable-write policies remain separate.
- Added regression coverage for connection cleanup on success and failure, lightweight CLI help, alias forwarding and unknown-command handling. The normal pytest run now includes the dashboard contract-generation check.

## Verification

The original source passed 951 tests. Seven new regression cases reproduced the open-connection and eager-import problems against a separate copy of that original source; the focused updated suite passed all 39 tests. The final `.venv/bin/python -m pytest -q` run passed **961 tests in 57.33 seconds**, with the same two dependency deprecation warnings as the baseline.

All 22 command help routes completed from an empty temporary directory without creating files. Seven local subprocess measurements per version gave median top-level help times of 74.3 ms before and 31.4 ms after. This is a local command-startup observation, not an inference throughput result.

Ruff's `F` checks passed for `src`, `scripts` and the new tests; `git diff --check` passed. Existing pytest fixture imports in older test files were not blindly removed as unused code. Dashboard declaration drift is checked by `tests/test_contract_types.py`; intentional schema changes can regenerate declarations with `uv run --locked python scripts/generate_contract_types.py`.

## Boundaries and next evidence gate

Source bytes changed, so frozen plans with implementation hashes may require their original source snapshot or a separately prepared new plan. This cleanup does not rewrite their identities, bypass preflight or authorize replay. Existing generated artifacts and model caches remain local and outside Git.

Artistic improvement remains unestablished. The next artistic evidence gate remains genuine human comparison, explicit parent choice, controlled revision and follow-up review; these software checks do not satisfy it.
