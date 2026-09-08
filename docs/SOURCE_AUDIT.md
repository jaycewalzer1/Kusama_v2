# Starter-source audit — 8 September 2026

The live starter corpus contains **200 distinct image-bearing source records, 400 geometric crops, 545 context claims (502 documented statements and 43 machine interpretations), and 40 machine-authored device seeds**. The machine claims comprise 39 explicit assistant visual observations and 4 short attributed reading notes. Thirty-nine device seeds have visual grounding in the saved source images. These observations and transfer hypotheses are **unvalidated**, and none is a human quality judgment. Source size does not establish artistic progress.

The local authoritative ledger is [`corpus.sqlite3`](../artifacts/overnight/corpus/corpus.sqlite3). Portable read snapshots are [`sources_manifest.jsonl`](../artifacts/overnight/corpus/sources_manifest.jsonl), [`regions.jsonl`](../artifacts/overnight/corpus/regions.jsonl), [`context_claims.jsonl`](../artifacts/overnight/corpus/context_claims.jsonl), and [`devices.jsonl`](../artifacts/overnight/corpus/devices.jsonl). The machine report has the current counts and storage bytes. These ignored, generated files are local evidence: their links do not resolve in a fresh checkout without the preserved artifact root.

## Executed coverage

| Institution | Intended initial quota | Accepted | Evidence and scope |
| --- | ---: | ---: | --- |
| Metropolitan Museum of Art | 60 | 40 | Live paginated v1.1 discovery and v1 object details; public-domain returned images |
| Art Institute of Chicago | 50 | 50 | Live artwork search, selected fields, public-domain filter, returned IIIF configuration |
| Cleveland Museum of Art | 40 | 60 | Live CC0 flag/image filter; structured descriptions and object details |
| Smithsonian | 50 | 50 | Bounded public S3 metadata ranges; 10 each from NASM, mineral sciences, botany, entomology and Cooper Hewitt |
| **Total** | **200** | **200** | Crops, context-only links and rejected records are excluded |

The Met pass encountered forbidden search responses and stopped at 40 eligible records. Its missing 20 were redistributed to Cleveland, with the revised allocation recorded in [`starter_sources.yaml`](../configs/starter_sources.yaml) and the ingestion event/config snapshots. No API key, complete collection clone, GPU, training, or paid inference was used by ingestion.

The six machine-assigned object families contain: painting/print 43; textile/pattern 24; sculpture/functional object 73; design/architecture 13; scientific/mechanical/everyday systems 17; natural form 30. There are 61 broadly functional/non-fine-art records, including 30 natural-history specimens and 10 aerospace/mechanism records. The remaining functional flags come from documented object types. These categories are a transparent retrieval convenience, not a curatorial distinction or an inference about community identity or ritual purpose.

The full documented place/culture and date strings are counted in [`corpus_report.json`](../artifacts/overnight/corpus/corpus_report.json). Missing metadata remains missing: maker 108, place/culture 21, date 22, medium 40, dimensions 43, description 72. Most natural specimens lack artist-style maker, medium and creation-date fields; collecting/locality metadata is preserved raw rather than repurposed into fabricated art metadata. European print and decorative traditions, museum cataloguing conventions and historical public-domain material remain strongly represented. This is not an adequate contemporary, global or originating-community-balanced corpus. Scientific sample images often include mounting sheets, labels, rulers or microscope presentation; these are not isolated transparent objects.

## Official source verification and rights rules

Current primary documentation was checked on 8 September 2026, and the adapter routes were exercised from this runtime.

- [Met API documentation](https://metmuseum.github.io/) announces `/public/collection/v1.1/search` with `offset` and `limit`, and the October 1 retirement of old v1 search. Object details remain `/public/collection/v1/objects/{id}`. The adapter retains `isPublicDomain`, returned image URL, object URL, maker/date/culture, medium, dimensions and credit line. Its [`Image and Data Resources` policy](https://www.metmuseum.org/policies/image-resources) distinguishes CC0 basic collection data and qualifying images from restricted images and other site content.
- [Cleveland API documentation](https://openaccess-api.clevelandart.org/) specifies the parameterless `cc0` flag and numeric `has_image=1`; each admitted record independently requires `share_license_status == "CC0"` and a returned web-image URL. The documentation releases dataset text/data as CC0 and limits image access by object rights status.
- [Art Institute API documentation](https://api.artic.edu/docs/#artworks) separates CC-BY-4.0 `description` from CC0 artwork-response data. The adapter requires `is_public_domain == true` and a returned `image_id`, and constructs the documented IIIF request using the returned `config.iiif_url`. Description attribution and the fact that HTML was removed are retained. Article/publication endpoints and unrestricted reuse of the general website are outside this rule.
- [Smithsonian Open Access repository](https://github.com/Smithsonian/OpenAccess) points to its public [S3 metadata index](https://smithsonian-open-access.s3-us-west-2.amazonaws.com/metadata/edan/index.txt). The adapter follows returned unit/file links, reads at most 512 KiB per selected stream using HTTP Range, preserves those exact bytes, skips partial final JSON lines, and retains line cursors. It independently requires CC0 metadata **and the selected media item’s CC0 usage field**. The [Smithsonian Open Access FAQ](https://www.si.edu/openaccess/faq) distinguishes CC0 metadata from restricted media. Screen-image URLs are returned resources, not invented asset IDs.

Each record separately records metadata, descriptive text, image retrieval/display, derivative and image-training eligibility, with license evidence, attribution and conditions. Eligible CC0 images may enter display, crop and image-training operations; this does **not** make CC-BY descriptions or arbitrary linked articles an unrestricted training dataset. AIC description use remains conditional on attribution and identification of changes. Unknown or restricted images fail admission while other sources continue. Rights eligibility is a recorded pipeline rule, not a report that training occurred.

Original bytes mean the exact image representation retrieved from the provider, often its bounded web/screen/IIIF derivative; they are not falsely described as the highest-resolution master. Viewing derivatives apply EXIF orientation, RGB/sRGB conversion where the embedded profile permits it, a bounded long edge and JPEG encoding. Transform lists disclose missing profiles or conversion limitations. Original, normalized and cropped bytes have separate hashes and dimensions. All raw API pages/stream ranges remain available by SHA-256.

## Context, devices and inspection

The 502 object-context statements retain source IDs, source URLs, raw response hashes and sentence locators. Their status means **the institution said it**, not that every interpretation is independently established historical truth. Four additional short assistant notes link to [Art21’s Mutu profile](https://art21.org/artist/wangechi-mutu/), its [studio-process account](https://art21.org/2021/07/21/weekly-watchlist-wangechi-mutu-returns-to-nature/), [MoMA’s Höch study](https://www.moma.org/collection/works/33945), and [The Met’s transnational Surrealism exhibition](https://www.metmuseum.org/ja/exhibitions/surrealism-beyond-borders). These are context-only links/paraphrases, with no presumed image or publication training rights. MoMA identifies the Höch study as watercolor, gouache and pencil; the artist’s broader photomontage practice does not change that object’s medium.

The assistant inspected the actual [200-source contact sheet](../artifacts/overnight/corpus/source_contact_sheet.jpg) and [40-device example sheet](../artifacts/overnight/corpus/device_examples_contact_sheet.jpg). Thirty-nine observations describe visible configurations—repeated bars, framed openings, shell ridges, material boundaries, vessel surfaces and specimen presentation—without claiming a universally caused emotion. Proposed transfers, alternate readings and contextual dependencies remain separately marked hypotheses. Original machine seed text is preserved in append-only `device_grounding_revision` events when its visual grounding is added.

One mounted botanical example, `smithsonian_nmnhbotany_2391073` (Codium cylindricum), is too faint in the bounded view to serve confidently as a biological-form example. The [visual-audit sidecar](../artifacts/overnight/corpus/source_visual_audit.json) excludes it from a director shortlist pending larger-view inspection. Its original bytes and failed usefulness observation remain retained. Other mounted specimens and tiny text can also become illegible at thumbnail size; use the full source view for context.

Every crop is explicitly labeled `geometric_fallback`. Boxes use normalized xyxy coordinates on the oriented viewing space, with minimum area/size/aspect checks, no silent out-of-bounds clamping, inherited rights and split grouping, parent links and crop hashes. Crops are not semantic segmentations. Forty device seeds link to real source examples, but their geometric region links require semantic review before claiming that a specific object or relationship has been localized.

## Failure, duplication and verification evidence

The corpus currently occupies approximately 165 MiB including raw responses, original retrieved representations, derivatives, crops, ledgers and audit exports, below the 2 GiB cap. The byte guard includes existing files and a conservative derivative reserve. The store admits one object identity and one exact original-image hash; duplicates do not add references or crops. Source families group supplied maker/culture/title or collection-parent evidence; exact-byte deduplication is not claimed as complete perceptual duplicate detection. Some maps and print-series sections are genuinely separate records but correlated sources.

The initial ingestion exposed a SQLite insert-placeholder bug after successful downloads. It was fixed, and nine already-saved validated JSON responses were reconciled into the cache without additional HTTP requests. All 25 affected request-attempt failures and retained bytes remain disclosed. Across the executed passes, the request ledger records 372 attempts and 304 normal completions; the 68 failed attempts comprise 25 local persistence failures, 37 HTTP 403 responses, 3 HTTP 404 responses and 3 URL preflight failures caused by a returned Met filename containing a space. These are attempt counts, not all failed network calls or distinct failed objects. URL encoding was corrected for future fetches; already-exhausted requests were not silently repeated. One source was excluded after an ICC conversion error, with a bounded conversion fallback now implemented. There are 392 rejected record encounters, mostly 388 without an explicitly eligible image URL. Rejected metadata and failed downloads are excluded from the 200.

The [live re-ingestion report](../artifacts/overnight/corpus/live_reingestion_verification.json) verifies that repeating the complete ingest preserves 200 references, 400 crops and all original image hashes, with no additional request attempts. The [byte verification](../artifacts/overnight/corpus/verification.json) checks every original/view asset, raw source response and crop hash plus rights/split inheritance. Fifteen focused fixture tests cover cache reuse and tampering, durable three-attempt ceilings, corrupt-image rejection, coordinate/rights/split validation, duplicate images/crops and byte limits. Fixture tests and successful HTTP execution are separate from assistant image inspection and human artistic evaluation; no human labels were collected here.

Exact commands from the project root:

```sh
.venv/bin/python -m kusama.corpus ingest --config configs/starter_sources.yaml
.venv/bin/python -m kusama.corpus report --config configs/starter_sources.yaml
.venv/bin/python -m kusama.corpus verify --config configs/starter_sources.yaml
.venv/bin/python -m kusama.corpus schema
.venv/bin/python -m pytest -q tests/test_corpus.py
```

The configuration uses JSON syntax, valid YAML 1.2, so parsing does not introduce an undeclared YAML dependency. The next source gate is human inspection of a small retrieved shelf and its selected details for contextual usefulness, including whether source-family grouping and rights distinctions survive the full creative/review path. More images are secondary to that evidence.

## Executed whole/detail retrieval follow-up

The final CPU-built index is [`849a29aaaec9d1959002`](../artifacts/overnight/indexes/849a29aaaec9d1959002/manifest.json), format `clip512-rrf60-v3`. It contains **199 eligible whole views and 398 crop views**, excluding the faint Codium view and both descendants from shortlisting while preserving all 200 original corpus records. Its content identity freezes the visual-audit hash/exclusions, source documents, claims, CLIP revision, and predicate query vocabulary. Prior indexes remain intact. Changing later source-audit findings requires building another index; the saved index’s shortlist does not silently change.

There are **54 unique typed source edges** extracted from the 39 saved assistant visual observations: encloses 13, repeats 12, contrasts 8, interrupts 8, fragments 5, conceals 2, connects 5, supports 1. These are explicit unvalidated interpretations of observed configurations. Generic inspection questions, source-transfer intentions, emotional predictions and arbitrary shared words in a device record cannot create an edge. Additional predicates may be supplied as registered edges only with resolvable source/evidence IDs; a machine note cannot be promoted to a documented institutional statement by changing its label. No substitute edge was invented to fill a vocabulary quota.

The relationship channel matches these typed predicates, with separately recorded brief, whole-context and local-need cues. Distant candidates must match a requested predicate outside the first two relevant items’ domains or explicitly preferred domains. A broad first shelf can therefore include several domains without automatically eliminating every distant slot. Sparse relations produce disclosed shortages and fallback assignments. Whole-source edges inherited by geometric details retain a caution: they do not prove that the exact crop contains the observed relationship.

The [12-query live CLIP audit with whole/detail context](../artifacts/overnight/indexes/849a29aaaec9d1959002/retrieval_audit_context_v2.json) resolved real source IDs and hashes for every query. Query images use actual frozen CLIP descriptors; lexical and typed relationship ranks plus reciprocal-rank fusion remain inspectable. Four queries filled both requested distant slots; others disclosed the lack of suitable typed relations and used the recorded fallback. This is mechanical grounding, not validated human usefulness or distractor rejection. CLIP token admission and any truncation are recorded; local queries send both their local need and whole context to the visual-text encoder.

The assistant inspected the [first query’s contact sheet](../artifacts/overnight/indexes/849a29aaaec9d1959002/retrieval_audit_contact_sheet.jpg). The grille and supported cup are clear structural examples, and the distant portrait/map have visible framing relationships. The high-ranked masks also expose the remaining ambiguity: a formal border around a face is not automatically evidence of supportive or restrictive social meaning. The next useful check is a human comparison of these references against the actual requested relationship, not a larger candidate count.

The source, resource, retrieval and inspiration focused suites together passed **50 fixture tests**, including explicit edge evidence, extension predicates, dominant-domain distant selection, frozen audit exclusion, complete local/whole encoder context, immutable index identity, resource concurrency and interrupted-call accounting. Final index image encoding took approximately 6.36 seconds on CPU with two Torch threads; this is an observed indexing time, not end-to-end research latency or artistic performance.

```sh
.venv/bin/python -m kusama.retrieval query \
  --index artifacts/overnight/indexes/849a29aaaec9d1959002 \
  --text 'An enclosure that can support and confine' --scope whole \
  --output artifacts/overnight/inspectable_source_query.json
.venv/bin/python -m pytest -q tests/test_corpus.py tests/test_resources.py tests/test_retrieval.py tests/test_inspiration.py
```

## Final cached audit — 8 September 2026, 05:56 UTC

The [compact final report](../artifacts/overnight/source_audit_final/summary.json) records a second cached-only re-ingest and complete byte/rights/edge checks. The re-ingest used **zero fetches and zero new request attempts**; all reference, crop, claim and device rows and image hashes stayed identical. The request ledger remains at 372 attempts. The corpus occupies 173,892,319 bytes at this check. The network entry point was disabled for re-ingestion, and no new sources, downloads or architecture were added.

All **313 cached raw response files** also match their stored response hashes, including the nine previously reconciled cache entries. The [raw-cache audit](../artifacts/overnight/source_audit_final/raw_response_cache_verification.json) additionally inventories retained request-failure files. Their original failure events named retained paths without independently recording payload hashes, so this inventory does not retroactively claim an earlier checksum guarantee.

All **597 frozen index views** passed asset-hash and operation-rights checks, including derivative permission and split/rights inheritance for all 398 crops. The 50 AIC whole records and their 100 crops still carry conditional descriptive-text rights; image eligibility does not erase those conditions. All **54 unique typed edges exactly match the text and source ID of a saved machine visual claim**, resolve their device evidence, and remain unvalidated machine interpretations. See the [per-edge and rights audit](../artifacts/overnight/source_audit_final/rights_and_typed_edges.json).

The [12-query replay](../artifacts/overnight/source_audit_final/retrieval_replay_verification.json) used frozen image vectors and the cached frozen CLIP text encoder on CPU with two Torch threads. Every query returned the **same eight IDs in the same order** as the earlier final-index audit, with no CLIP truncation and all selected hashes resolving. Four queries filled both distant slots, one filled one, and seven filled neither; recorded fallback still supplies the remaining shelf positions. Every file in the final index retained its pre-audit hash. Audit outputs were written outside the index. No MPS/GPU work or sealed-brief reads were performed.

The [metadata coverage audit](../artifacts/overnight/source_audit_final/metadata_coverage.json) retains every exact place/culture and date label with its record ID and the explicit counting rules. There are **169 source families: 150 singleton families and 19 multi-record families containing 50 records**. The largest two groups each contain five records. This remains heuristic grouping, not proof of independent sources or comprehensive leakage prevention.

| Explicit place-name cues in recorded culture/place/locality fields | Records |
| --- | ---: |
| Europe | 56 |
| Latin America / Caribbean | 25 |
| Africa | 20 |
| North America | 19 |
| East Asia | 15 |
| West / Central Asia | 9 |
| Southeast Asia | 7 |
| South Asia | 4 |
| Antarctica | 4 |
| Named ocean / marine locality | 1 |
| Missing field | 21 |
| Present label left unmapped | 19 |

These are literal metadata cues, mixing culture labels with object places and specimen localities. They do not establish nationality, community identity or a balanced account of origin. No explicit Oceania place cue matched the rules; culture-only labels were left unresolved instead of assigning them through outside assumptions. All four admitting institutions are US museum providers. Forty records therefore remain missing or unmapped by this geography check, and museum metadata does not supply originating-community interpretation.

Date metadata is absent for 22 records and explicitly undated for one more. Literal labels contain 32 BCE/cross-era cues, 35 pre-1500 CE cues, 35 cues from 1500–1799, 41 from 1800–1899, and 4 from 2000 onward; records with ranges or alteration dates can occur in multiple buckets. The four 2000+ dates and 11 further post-1949 numeric dates belong to natural specimens and may describe collection events. The sole other post-1949 numeric cue is Cleveland `143792`, *Fragments*, whose saved API fields all say **1420–1955**. That unusually broad range is preserved and does not establish a postwar artwork. Broad labels such as “1900s” also remain imprecise; numeric cues are not inferred creation periods. Contemporary-art coverage remains unestablished by these records.

The final audit adds evidence of cached reproducibility and explicit coverage gaps. It does not establish human contextual usefulness, artistic quality, reliable emotional transfer or a learned advantage.
