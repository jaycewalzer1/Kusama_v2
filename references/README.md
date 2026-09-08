# Initial reference library

Two small packets support the frozen-model experiment. The JSON files are executable inputs to `kusama-creative`; each source has provenance, a rights statement, a short contextual paraphrase, and a separately stated proposed relationship to the brief. `asset_receipts.json` records the downloaded web renditions, hashes and retrieval times. Original image bytes are retained. This is an inference-time library, not a training dataset.

## Original inspiration: Wangechi Mutu

[The process packet](mutu_process/packet.json) contains seven sources and two images. It covers collage structure, material transformation, Nairobi and diasporic context, Mutu's account of women and authority, and specific comparative sources. [Art21's artist profile](https://art21.org/artist/wangechi-mutu/) and [studio account](https://art21.org/2021/07/21/weekly-watchlist-wangechi-mutu-returns-to-nature/) provide the process foundation; [her artist statement](https://www.metmuseum.org/press-releases/wangechi-mutu-artist-statement-2019-exhibitions) anchors interpretation in her words.

The Luba/Hemba and Kenyan environmental references are explicitly distinguished. A thematic comparison does not establish an artistic influence. The packet is a starting selection, not a complete account of Mutu's life, Kenyan cultures or African art.

## Active brief: Maya memory

The user asked for a painting expressing the feeling of being Maya and that “our history is lost.” The exact message and interpretation are retained in [brief history](maya_memory/brief_history.json). [The active packet](maya_memory/packet-v3.json) selects twelve references with four images:

| Role | Sources in the packet |
|---|---|
| Artistic process and visual language | Mutu's *Family Tree* and Art21's Nairobi studio account |
| Living presence | NMAI's Maya voices and Rigoberta Menchú Tum's testimony |
| Manuscript survival, loss and displacement | Dresden Codex, SLUB's provenance account, INAH's Maní research |
| Oral and written transmission | Newberry's Kʼicheʼ *Popol Vuh* guide |
| Material knowledge | Maya weaving, a specifically documented Poqomam huipil from Palín, a Classic-period maize whistle |
| Fragile reconnection | The documented distribution of *Popol Vuh* facsimiles in Guatemala |

Loss is the user's emotional intention. The packet does not turn it into a historical claim that Maya peoples or all of their knowledge disappeared. Community-specific references remain identified. The directions may explore rupture, unreachable records, broken transmission, dignity and continued presence; no single direction is required to include every theme.

The [Library of Congress scan](https://www.loc.gov/item/2021667917/) shows a faded surviving manuscript page, not colonial burning. LOC and SLUB give different production dates, retained as an uncertainty. The maize whistle is an optional formal reference for a relation between body and vegetation; it does not license invented ritual explanations. The painting should not contain fabricated readable Maya writing.

## What is sent to the models

The selected packet and four images are copied into the experiment directory and hashed. Qwen receives bounded image renditions and the sourced context; its directions record which sources it proposes to transform. FLUX receives Qwen's exact painting prompts. The initial images have no image parent; later edits must identify their selected parent. No reference artwork is silently pasted into the outputs.

Copyright remains with the respective holders. The local research copies are not represented as open-licensed assets except where the source explicitly identifies them that way. Public display or redistribution of the reference images is not part of this experiment. Model outputs and human reviews are stored separately from the reference library.

## Extending the library

Add a new sourced packet or version rather than rewriting inputs for a running experiment. Prefer additional Maya scholars, artists, language workers and community-specific sources when a later brief needs more depth. Cultural review remains human work: a packet improves what Qwen can consult but does not establish that its interpretation or the resulting painting succeeds.

The initial `packet.json` is retained for the first pilot. `packet-v2.json` adds the source's Poqomam attribution; `packet_changes.json` records the change. Existing run snapshots remain unchanged.

The version-three packet retains those sources and adds twelve explicit compositional challenges after the second pilot repeated the same fragmented-body motif. These challenges were written by Codex as experimental scaffolding; Qwen still formulates the detailed direction, source transformations and executable prompt. The full map and its provenance are saved in `packet-v3.json` and [the diversity record](maya_memory/diversity_scaffold.json). Only the current challenge is supplied in each request, with prior titles for duplicate avoidance. This is an instruction change, with no weight update.
