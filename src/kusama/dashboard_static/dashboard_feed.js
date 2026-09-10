/* A read-only, newest-first projection of saved run evidence. */
let runFeedEntries = [];
let runFeedLimit = 40;

function feedTime(value) {
  const time = Date.parse(value || "");
  return Number.isFinite(time) ? time : 0;
}

function resetRunFeed() {
  runFeedEntries = []; runFeedLimit = 40;
  byId("run-update-search").value = "";
  byId("run-update-type").value = "";
  byId("run-update-count").textContent = "";
  byId("run-updates-more").hidden = true;
  for (const node of document.querySelectorAll(".run-disclosure")) node.open = false;
  byId("run-evidence").open = !window.matchMedia?.("(max-width:1100px)").matches;
}

function openRunSection(name) {
  const id = {overview: "run-context", research: "run-research-section", images: "run-images-section",
    development: "run-development-section", files: "run-files-section", activity: "run-updates"}[name];
  const section = byId(id || "run-updates");
  byId("run-evidence").open = true;
  section.open = true;
  section.scrollIntoView({behavior: "smooth", block: "start"});
}

function buildRunFeed(result) {
  const entries = [], represented = new Set();
  const files = new Map((result.files || []).map(file => [file.name, file]));
  const events = new Map((result.activity || []).map(item => [item.id, item]));
  const join = (directory, name) => !directory || directory === "." ? name : `${directory}/${name}`;
  for (const session of result.research || []) {
    for (const step of session.steps || []) {
      const path = join(session.directory, `step-${String(step.index).padStart(3, "0")}.json`);
      represented.add(path);
      const index = String(step.index).padStart(3, "0");
      const attachments = [path, ...[`decision-${index}.json`, `decision-${index}-source.json`, `response-${index}.json`].map(name => join(session.directory, name))]
        .map(name => files.get(name)).filter(Boolean);
      for (const file of attachments) represented.add(file.name);
      const action = step.action || {};
      entries.push({id: `research:${path}`, type: "research", at: events.get(path)?.at || files.get(path)?.modified_at,
        label: step.status === "response_rejected" ? "Model response rejected" : (action.action || action.kind || "Research step").replaceAll("_", " "),
        detail: action.reason || action.note || action.query || action.question || action.hypothesis?.study_reason || action.hypothesis?.question || "No explanation supplied.",
        origin: session.fixture ? "Fixture actor" : step.origin || action.origin || "Qwen research",
        status: step.status, step, transcript: (session.transcripts || []).find(item => item.index === step.index),
        directory: session.directory, fixture: session.fixture, files: attachments});
    }
  }
  for (const file of result.images || []) {
    represented.add(file.name);
    const sidecar = files.get(file.name.replace(/\.[^.]+$/, ".json"));
    if (sidecar) represented.add(sidecar.name);
    entries.push({id: `image:${file.name}`, type: "image", at: file.modified_at,
      label: file.label || file.name.split("/").at(-1), detail: file.name,
      origin: file.record?.origin || "Saved image", file, files: sidecar ? [sidecar] : []});
  }
  const workflow = result.development_cycle;
  for (const row of workflow?.development?.cycles || []) {
    const path = join(workflow.development_directory, `cycle-${String(row.step).padStart(2, "0")}.json`);
    represented.add(path);
    const choice = row.choice;
    entries.push({id: `development:${path}`, type: "development", at: files.get(path)?.modified_at,
      label: {research: "Research reopened", continue: "Revise a saved branch", combine: "Combine branches", reformulate: "Reformulate",
        review: "Human review requested", stop: "Development stopped", abandon: "Direction abandoned"}[choice.action] || choice.action,
      detail: choice.what_changed || choice.reason, origin: workflow.fixture ? "Fixture actor" : "Qwen decision", row, workflow});
  }
  for (const item of result.activity || []) {
    if (represented.has(item.id)) continue;
    if (item.file) represented.add(item.file.name);
    entries.push({...item, id: `activity:${item.origin}:${item.id}`, type: "activity"});
  }
  for (const file of result.files || []) {
    if (represented.has(file.name)) continue;
    entries.push({id: `file:${file.name}`, type: "file", at: file.modified_at,
      label: file.type === "log" ? "Log updated" : "File saved", detail: file.name, file});
  }
  entries.sort((a, b) => feedTime(b.at) - feedTime(a.at) || a.id.localeCompare(b.id));
  // A single action often saves several receipts within the same second.
  // Keep these accessible with the update instead of burying it in bookkeeping.
  const buckets = new Map();
  for (const entry of entries) {
    const key = feedTime(entry.at) ? Math.floor(feedTime(entry.at) / 1000) : entry.id;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(entry);
  }
  const condensed = [];
  for (const bucket of buckets.values()) {
    const technical = entry => entry.type === "file" || (entry.type === "activity" && entry.origin?.includes("execution ledger"));
    const primary = bucket.find(entry => !technical(entry));
    if (!primary) { condensed.push(...bucket); continue; }
    const related = bucket.filter(technical);
    primary.files = [...(primary.files || []), ...related.filter(entry => entry.type === "file").map(entry => entry.file)];
    primary.events = related.filter(entry => entry.type === "activity");
    primary.at = bucket[0].at;
    condensed.push(...bucket.filter(entry => !technical(entry)));
  }
  condensed.sort((a, b) => feedTime(b.at) - feedTime(a.at) || a.id.localeCompare(b.id));
  // Adjacent file saves share a compact card; no saved file is dropped.
  const grouped = [];
  for (const entry of condensed) {
    const prior = grouped.at(-1);
    if (entry.type === "file" && prior?.type === "file" && feedTime(prior.at) - feedTime(entry.at) < 60000) {
      prior.files.push(entry.file);
    } else grouped.push(entry.type === "file" ? {...entry, files: [entry.file]} : entry);
  }
  return grouped;
}

function feedDisclosure(holder, label, render) {
  const details = cycleElement("details", "feed-evidence");
  details.append(cycleElement("summary", "", label));
  let loaded = false;
  details.addEventListener("toggle", () => {
    if (!details.open || loaded) return;
    loaded = true; render(details);
  });
  holder.append(details);
  return details;
}

function feedFile(holder, file) {
  feedDisclosure(holder, file.name, body => {
    const controls = cycleElement("div", "feed-file-controls");
    const link = artifactLink(file); link.textContent = "Open full file ↗";
    const reload = cycleElement("button", "", "Reload file"); reload.type = "button";
    const preview = cycleElement("div", "feed-file-preview");
    const load = async () => {
      reload.disabled = true;
      if (file.type === "image") {
        preview.replaceChildren();
        const image = cycleElement("img", ""); image.src = scopedURL(file.url); image.alt = file.name;
        preview.append(image); reload.disabled = false; return;
      }
      const pre = cycleElement("pre", "", "Loading saved file…"); preview.replaceChildren(pre);
      if (file.bytes > 2 * 1024 * 1024) {
        pre.textContent = "This file is large. Open the full file to inspect it."; reload.disabled = false; return;
      }
      try {
        const response = await fetch(scopedURL(file.url), {cache: "no-store"});
        if (!response.ok) throw new Error(`Could not read file (HTTP ${response.status}).`);
        const value = await response.text();
        pre.textContent = value.slice(0, 160000) + (value.length > 160000 ? "\n… Open full file for the remaining content." : "");
      } catch (error) { pre.textContent = error.message; }
      finally { reload.disabled = false; }
    };
    reload.addEventListener("click", load);
    controls.append(reload, link); body.append(controls, preview); load();
  });
}

function renderFeedCard(card, entry) {
  const heading = cycleElement("div", "feed-card-heading");
  const category = {research: "Research", image: "Image", development: "Development", activity: "Execution", file: "Files & logs"}[entry.type];
  heading.append(cycleElement("span", "feed-category", category));
  const time = cycleElement("time", "feed-time", feedTime(entry.at) ? new Date(entry.at).toLocaleString(undefined,
    {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit"}) : "Time not recorded");
  if (feedTime(entry.at)) time.dateTime = entry.at;
  // File timestamps describe saves/updates, not inferred model execution times.
  if (["file", "image", "development"].includes(entry.type)) time.title = "Saved file modification time";
  heading.append(time);
  card.append(heading);
  const title = entry.type === "file" ? `${entry.files.length} saved file${entry.files.length === 1 ? "" : "s"} updated` : entry.label;
  card.append(cycleElement("h4", "feed-title", title));
  if (entry.origin || entry.status) card.append(cycleElement("p", "feed-origin", [entry.origin, entry.status?.replaceAll("_", " ")].filter(Boolean).join(" · ")));
  if (entry.type === "image") {
    card.append(imageCard(entry.file.url, entry.file.label || entry.file.name, entry.file.label || entry.file.name));
    feedDisclosure(card, "Prompt, image inputs & lineage", holder => renderImageTranscript(holder, entry.file.record));
  } else if (entry.type === "file") {
    for (const file of entry.files) feedFile(card, file);
  } else {
    if (entry.detail) card.append(cycleElement("p", "feed-description", entry.detail));
    if (entry.type === "research") {
      const action = entry.step.action || {};
      if (action.query && action.query !== entry.detail) card.append(cycleElement("blockquote", "cycle-question", action.query));
      if (entry.step.tool_error || entry.step.error) card.append(cycleElement("p", "error", text(entry.step.tool_error || entry.step.error)));
      feedDisclosure(card, `Step ${Number(entry.step.index) + 1} · prompt, response & evidence`, holder => {
        renderResearchTranscript(holder, entry.step, entry.transcript, entry.fixture);
        researchEvidence(holder, "Exact action and result", entry.step);
        const open = cycleElement("button", "", "Inspect this research session →");
        open.addEventListener("click", () => focusCycleResearch(entry.directory)); holder.append(open);
      });
    } else if (entry.type === "development") {
      const choice = entry.row.choice;
      if (choice.new_question) card.append(cycleElement("blockquote", "cycle-question", choice.new_question));
      if (choice.action === "review") card.append(cycleElement("p", "muted", "Requested by the model. No human judgment is recorded by this decision."));
      cycleBranchLinks(card, choice.selected_image_ids, cycleViewModel({development_cycle: entry.workflow, images: []}));
      feedDisclosure(card, "Reason, commitments & observations", holder => readableValue(holder, entry.row));
    } else if (entry.record) {
      feedDisclosure(card, "Exact saved event", holder => readableValue(holder, entry.record));
    }
  }
  if (entry.type !== "file" && (entry.files?.length || entry.events?.length)) {
    const count = entry.files?.length || 0;
    feedDisclosure(card, `${count ? `${count} saved file${count === 1 ? "" : "s"}` : "Execution details"}${entry.events?.length && count ? " & execution details" : ""}`, holder => {
      for (const file of entry.files || []) feedFile(holder, file);
      for (const event of entry.events || []) researchEvidence(holder, `${event.label} · ${event.status || "recorded"}`, event.record || event);
    });
  }
}

function renderRunFeed(result) {
  runFeedEntries = buildRunFeed(result);
  const metrics = byId("run-metrics"); metrics.replaceChildren();
  const run = selectedRunRecord;
  for (const label of [run && text(run.status).replaceAll("_", " "), `${(result.images || []).length} images saved`,
    `${(result.research || []).reduce((n, s) => n + (s.steps || []).length, 0)} research steps`,
    run && `${calls(run)} model calls`].filter(Boolean)) metrics.append(cycleElement("span", "", label));
  renderFeedEntries();
}

function renderFeedEntries() {
  const query = byId("run-update-search").value.trim().toLowerCase();
  const type = byId("run-update-type").value;
  const matches = runFeedEntries.filter(entry => (!type || entry.type === type || (type === "file" && entry.files?.length) || (type === "activity" && entry.events?.length)) && (!query || JSON.stringify(entry).toLowerCase().includes(query)));
  const visible = matches.slice(0, runFeedLimit);
  const holder = byId("run-updates");
  const existing = new Map([...holder.children].map(node => [node.dataset.entryId, node]));
  const wanted = new Set(visible.map(entry => entry.id));
  // Keep the reader's current card in place when newer evidence is prepended.
  const anchor = [...holder.children].find(node => node.getBoundingClientRect?.().top >= 0);
  const anchorTop = anchor?.getBoundingClientRect?.().top;
  const preserveAnchor = holder.getBoundingClientRect?.().top < 0;
  for (const node of [...holder.children]) if (!wanted.has(node.dataset.entryId)) node.remove();
  for (const [index, entry] of visible.entries()) {
    const card = existing.get(entry.id) || cycleElement("article", `feed-card feed-${entry.type}`);
    card.dataset.entryId = entry.id; card.dataset.at = entry.at || "";
    const signature = JSON.stringify(entry);
    if (card.dataset.signature !== signature) {
      const expanded = new Set([...card.querySelectorAll("details[open]")].map(node => node.querySelector("summary")?.textContent));
      card.replaceChildren(); renderFeedCard(card, entry);
      for (const detail of card.querySelectorAll("details")) if (expanded.has(detail.querySelector("summary")?.textContent)) detail.open = true;
      card.dataset.signature = signature;
    }
    if (holder.children[index] !== card) holder.insertBefore(card, holder.children[index] || null);
  }
  if (!visible.length) holder.append(cycleElement("p", "empty", runFeedEntries.length ? "No updates match this search." : "No updates saved yet. Research, images and files will appear here as the run progresses."));
  if (preserveAnchor && anchor?.isConnected) window.scrollBy(0, anchor.getBoundingClientRect().top - anchorTop);
  byId("run-update-count").textContent = `${visible.length} of ${matches.length} updates${type || query ? " matching your filters" : ""}`;
  byId("run-updates-more").hidden = visible.length >= matches.length;
}
