# Local inference cost accounting

New FLUX generation/edit and frozen Qwen director operations record append-only cost events in `costs.sqlite3`, alongside the applicable run database. A manifest's `cost_accounting.root` and `cost_accounting.run_id` identify its records. Each event binds a stable run ID, call ID, model, execution scope and snapshotted rate policy. Controlled attempts and original generation attempts also expose `cost_call_id`.

This machine executes pretrained models locally. Its **provider API fee is zero** for these operations. That does **not** establish a zero total cost. Electricity and hardware allocation remain **unknown** until the operator supplies rates. Monetary accounting is independent of the existing inference-call budgets.

## What is recorded

- An immutable start event exists before a tracked operation begins. A second event records its terminal outcome and measured duration; updates and deletes are blocked by database triggers.
- The existing synchronized inference/memory measurements are retained where available. Some wrappers measure the backend request or loading operation's wall time, explicitly named in `scope`; sampled memory is not a power measurement.
- Model loading and backend requests are separate, non-overlapping scopes. Summary `calls` counts costed operations, which can include model loading; it is not the image-call budget counter.
- Failed and interrupted operations retain consumed runtime and any resulting estimate. A hard process crash leaves an open call with unknown duration and total. No model is rerun to determine a price.
- Repeated append of identical event content is idempotent; conflicting content is rejected. The context wrapper refuses to execute an already recorded operation.
- Each currency is summed independently. `known_subtotal` contains only known components, while `total` is null if any component is unknown. No cost records means unknown historical coverage, not free execution.

A complete estimate covers the recorded local compute scopes only. It excludes labor, storage, research, networking and other overhead. It is not a provider invoice, measured electricity bill or full project production cost. Concurrent workloads can make a user-supplied average power assumption inaccurate.

## Supplying an estimate policy

Set `KUSAMA_COST_POLICY` to the absolute path of a JSON file. The default policy has no numeric tariff, power assumption or hardware rate. Its default reporting currency is USD; it makes no assumption about the user's actual tariff currency.

```json
{
  "currency": "USD",
  "average_power_watts": null,
  "electricity_per_kwh": null,
  "hardware_per_hour": null,
  "source": null
}
```

Replace nulls only with values supported by your own measurements, tariff or accounting assumptions. Any supplied numerical rate or power assumption requires a nonempty `source` explaining where it came from. Currency uses a three-letter code. Values must be finite and nonnegative. A zero rate must be explicitly supplied rather than inferred from missing data.

Electricity estimate = runtime hours × assumed average watts / 1000 × electricity tariff per kWh.

Hardware estimate = runtime hours × allocated hardware cost per hour.

The policy is copied into the start event. Later edits to the policy file cannot change existing events. This implementation does not read or charge an external billing account.

## Reading and reconciling

```sh
uv run python -m kusama.costs --root artifacts --run-id RUN_ID
```

For an executor that died without completing its event, explicitly close the open record as uncertain:

```sh
uv run python -m kusama.costs --root artifacts --run-id RUN_ID \
  --reconcile-call 'attempt:0' --evidence 'Process exited without a retained runtime measurement.'
```

Reconciliation preserves the original event and leaves missing runtime/cost unknown. It does not alter images, historical manifests, reviews or model-call reservations. Existing completed historical jobs are not silently backfilled or overwritten. A pre-accounting job can have resource measurements in its original manifest without having monetary coverage.

Python callers use `tracked_local_call(root, run_id, call_id, model, scope=...)` and `CostJournal(root).summary(run_id)`. Dashboard readers use `CostJournal(root, read_only=True)`. `Ledger.cost_summary(job_id)` aggregates registered descendants within the same ledger root without double counting their events. Nested directed-edit packets use separate accounting roots for the outer Qwen run and inner FLUX ledger; these must be inspected separately rather than mistaken for a single complete total.
