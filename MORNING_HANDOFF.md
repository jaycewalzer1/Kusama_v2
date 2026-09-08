# Overnight handoff — working checkpoint

**Still running as of 8 September 2026, 07:04 UTC; this is an interruption checkpoint.**

The first real source-to-image-to-local-revision path is working. Six whole-work
care/control images, eight local variants and two protected-child finishing
variants are saved. Ten actual observer calls are preserved, including five
empty-change reports accepted through explicit schema-revalidation sidecars.
E0 development is running at the separately named 384×512 profile: d01 has its
complete six-edit pool; d02 stopped before editing on a malformed preservation
list; d03/d04 continue. No human
votes or pool selections have been added and no model weights changed.

- Branch: `overnight/2026-09-08`; baseline `f04b6de`, latest code checkpoint `7d40bf4`; 628 fixture tests and TypeScript contract check passed.
- UI: [local review dashboard](http://127.0.0.1:8766/).
- Whole gallery: [care/control](artifacts/overnight/runs/care_control_v2/gallery.html).
- First local pair: [continuation](artifacts/overnight/continuations/care_control_v2_local_01/gallery.html).
- [Implementation status](docs/IMPLEMENTATION_STATUS.md), [source audit](docs/SOURCE_AUDIT.md), [workflow commands](docs/source_connected_studio.md).
- Authoritative usage: `artifacts/overnight/resources.sqlite3`; job/image evidence: `artifacts/overnight/jobs.sqlite3`; reviews: `artifacts/overnight/reviews.sqlite3`.

Do not launch a competing model job while the E0 worker holds the GPU lease.
Its command is `uv run --locked kusama e0 run --protocol artifacts/overnight/evaluation_v2/e0 --index artifacts/overnight/indexes/849a29aaaec9d1959002 --root artifacts/overnight --render-profile configs/e0_study_384.json`; log `artifacts/overnight/e0_development_remaining.log`. Completed calls replay without new reservations. The d02 schema failure needs a disclosed technical recovery before a complete development gate; its two valid actions and parent must be preserved. All heldout texts must remain outside the tuning path. [Completed first-episode evidence](artifacts/overnight/vertical_slice_evidence_final.json) records the actual 16-attempt chain; authoritative session usage remains in SQLite.

The current artistic uncertainty is whether local interventions make the proposed
source relationship visibly consequential. Catalog-like object rendering and
weak transformation in the chair direction were retained as findings. The next
evidence gate is inspection of real parent/child changes followed by genuine
human quality and development reviews. Technical completion cannot answer that.
