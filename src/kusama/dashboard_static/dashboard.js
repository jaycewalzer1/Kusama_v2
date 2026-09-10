"use strict";

const API_PREFIX = location.pathname?.match(/^\/workspaces\/[a-f0-9]{20}/)?.[0] || "";
function scopedURL(url) { return typeof url === "string" && url.startsWith("/api/") ? API_PREFIX + url : url; }
const byId = (id) => document.getElementById(id);
let selectedId = null;
let detailRequest = 0;
let lastRuns = [];
let liveTimer = null;
let liveRequest = 0;
let lastDetailJSON = null;
let lastInspectionJSON = null;
let inspectedFiles = [];
let selectedRunRecord = null;
const stopRequests = new Map();

let explorerRefreshPending = false;
const LIVE_INTERVAL_MS = 15000;
const BUSY_RUN_STATES = new Set(["pending", "queued", "proposing", "running", "directing", "rendering", "observing", "loading", "generating", "editing", "preparing", "recovering", "starting", "executing", "researching"]);

function text(value) {
  if (value === null || value === undefined || value === "") return "—";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

function verification(value) {
  if (typeof value === "boolean") return value ? "pass" : "fail";
  if (value && typeof value === "object") {
    if (value.scope === "technical_integrity") return `Technical integrity ${text(value.status)}`;
    if (value.status !== undefined) return text(value.status);
    if (typeof value.passed === "boolean") return value.passed ? "pass" : "fail";
    const counts = Object.entries(value).filter(([, count]) => typeof count === "number" && count > 0);
    if (counts.length) return counts.map(([label, count]) => `${count} ${label.replaceAll("_", " ")}`).join(" · ");
  }
  return text(value);
}

function kindName(kind) {
  return kind === "reference_creative_experiment" ? "Reference creative experiment" : text(kind).replaceAll("_", " ");
}

function calls(run) {
  return `${text(run.model_calls)} / ${text(run.max_calls)}`;
}

function runtime(value) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(1)} s` : "—";
}

async function getJSON(url) {
  const response = await fetch(scopedURL(url), { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load ${url} (HTTP ${response.status}).`);
  return response.json();
}

function showError(message) {
  byId("error").textContent = message;
  byId("error").hidden = !message;
}

function markSelected() {
  for (const row of byId("runs").children) {
    const selected = row.dataset.id === selectedId;
    row.classList.toggle("selected", selected);
    row.querySelector("button").setAttribute("aria-current", String(selected));
  }
}

function renderRuns(runs) {
  for (const run of runs) if (!BUSY_RUN_STATES.has(run.status)) stopRequests.delete(run.id);
  renderRecentInquiries(runs);
  byId("runs").replaceChildren();
  const query = byId("run-search").value.trim().toLowerCase();
  const status = byId("run-status-filter").value;
  const visible = runs.filter(run => (!query || [run.title, run.name, run.directory, run.kind, run.brief].join(" ").toLowerCase().includes(query)) && (!status || (status === "active" ? BUSY_RUN_STATES.has(run.status) : run.status === status)));
  byId("run-empty").hidden = visible.length > 0;
  for (const run of visible) {
    const row = document.createElement("tr");
    row.dataset.id = run.id;
    row.addEventListener("click", () => selectRun(run.id, run.name));
    const nameCell = document.createElement("td");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = run.title || run.name || run.id;
    button.title = run.directory || run.name || run.id;
    nameCell.append(button);
    if (run.directory && run.directory !== run.name) {
      const path = document.createElement("small"); path.className = "run-path"; path.textContent = run.directory; nameCell.append(path);
    }
    row.append(nameCell);
    for (const [index, value] of [kindName(run.kind), run.status, calls(run), verification(run.verification)].entries()) {
      const cell = document.createElement("td");
      if (index === 1 || (index === 3 && ["pass", "fail"].includes(value))) {
        const badge = document.createElement("span");
        badge.className = "status";
        badge.dataset.state = value;
        badge.textContent = text(value).replaceAll("_", " ");
        cell.append(badge);
      } else {
        cell.textContent = text(value);
      }
      row.append(cell);
    }
    const actions = document.createElement("td"); actions.className = "run-actions";
    if (canStopRun(run)) actions.append(createStopButton(run));
    row.append(actions);
    byId("runs").append(row);
  }
  markSelected();
}

function addFact(name, value) {
  const label = document.createElement("dt");
  label.textContent = name;
  const content = document.createElement("dd");
  content.textContent = text(value);
  byId("run-facts").append(label, content);
  return content;
}

function artifactLink(artifact) {
  const link = document.createElement("a");
  link.href = scopedURL(artifact.url);
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = artifact.name;
  return link;
}

function renderDetail(run) {
  selectedRunRecord = run;
  byId("run-progress").textContent = run.data?.progress_message || text(run.status).replaceAll("_", " ");
  if (run.data?.stop_requested && canStopRun(run)) stopRequests.set(run.id, "requested");
  configureStopButton(byId("stop-run"), run);
  syncStopButtons(run);
  const previousRawScroll = byId("raw").scrollTop;
  byId("run-facts").replaceChildren();
  byId("run-name").textContent = run.title || run.name || run.id;
  byId("run-brief").textContent = run.brief || "";
  byId("run-brief").hidden = !run.brief;
  byId("detail-status").textContent = "";
  addFact("Kind", kindName(run.kind));
  if (run.directory) addFact("Directory", run.directory);
  if (run.title) addFact("Run ID", run.name);
  if (run.planned_images !== null && run.planned_images !== undefined) addFact("Images saved", `${run.image_count} / ${run.planned_images}`);
  addFact("Status", run.status);
  if (run.observer_status) {
    const observer = run.observer_status;
    const status = text(observer.status).replaceAll("_", " ");
    addFact("Qwen observations", `${text(observer.valid_pairs)} / ${text(run.planned_observations)} valid responses · ${text(observer.rejected_pairs)} pairs rejected · ${status}`);
  }
  addFact("Model calls / limit", calls(run));
  addFact(run.runtime_scope === "recorded_operations" ? "Measured operation time" : "Runtime", `${runtime(run.runtime_seconds)}${run.runtime_scope === "recorded_operations" ? " · completed scopes only" : ""}`);
  const cost = run.monetary_cost;
  if (!cost || cost.coverage === "no_cost_records") {
    addFact("Monetary cost", "Not recorded");
  } else {
    for (const [currency, amount] of Object.entries(cost.currencies || {})) {
      addFact(`Cost (${currency})`, amount.complete ? `${amount.total.toFixed(4)}${amount.estimated_calls ? " (estimated)" : ""}` : `Total unknown · known subtotal ${amount.known_subtotal.toFixed(4)}`);
    }
    if (cost.open_calls) addFact("Cost records pending", cost.open_calls);
  }
  if (run.verification?.scope === "technical_integrity") {
    const fact = addFact("Technical integrity", `${text(run.verification.status)} · `);
    const report = document.createElement("a");
    report.href = API_PREFIX + `/api/runs/${encodeURIComponent(run.id)}/files/verification.json`;
    report.target = "_blank";
    report.rel = "noopener";
    report.textContent = "saved verification report";
    fact.append(report);
  } else {
    addFact("Verification", verification(run.verification));
  }
  byId("run-error").textContent = run.error ? text(run.error) : "";
  byId("run-error").hidden = !run.error;
  byId("raw").textContent = JSON.stringify(run.data ?? run, null, 2);
  byId("raw").scrollTop = previousRawScroll;
  lastDetailJSON = JSON.stringify(run);
  byId("raw-details").hidden = false;
  const metrics = byId("run-metrics");
  if (metrics.children.length) {
    metrics.children[0].textContent = text(run.status).replaceAll("_", " ");
    metrics.children[metrics.children.length - 1].textContent = `${calls(run)} model calls`;
  }
}

async function selectRun(id, name, navigate = true) {
  if (navigate) activatePanel("runs");
  lastInspectionJSON = null;
  ++activeFileRequest; activePreviewFile = null;
  byId("file-preview").hidden = true;
  byId("run-file-search").value = "";
  byId("stop-run").disabled = false;
  resetRunFeed();
  selectedId = id;
  const request = ++detailRequest;
  lastDetailJSON = null;
  markSelected();
  byId("detail").hidden = false;
  byId("run-name").textContent = name || id;
  byId("detail-status").textContent = "Loading run…";
  byId("run-brief").hidden = true;
  byId("run-facts").replaceChildren();
  byId("images").replaceChildren();
  byId("files").replaceChildren();
  for (const target of ["run-research", "run-updates", "run-metrics", "run-stage-list", "run-directions", "generations-without-images", "run-cycle-map", "run-cycle-summary", "development-status", "development-branches", "development-history"]) byId(target).replaceChildren();
  for (const target of ["run-cycle-map", "run-cycle-summary", "run-development-section"]) byId(target).hidden = true;
  byId("run-stage-list").hidden = false;
  inspectedFiles = []; selectedRunRecord = null;
  byId("stop-run").hidden = true;
  byId("run-progress").textContent = "";
  byId("run-error").hidden = true;
  byId("raw-details").hidden = true;
  byId("raw-details").open = false;
  try {
    const run = await getJSON(`/api/runs/${encodeURIComponent(id)}`);
    if (request === detailRequest) {
      renderDetail(run); await loadInspection(id, request);
      if (navigate && request === detailRequest) byId("detail").scrollIntoView({behavior:"auto", block:"start"});
    }
  } catch (error) {
    if (request === detailRequest) byId("detail-status").textContent = error.message;
  }
}

async function refresh() {
  ++liveRequest;
  clearTimeout(liveTimer);
  byId("refresh").disabled = true;
  showError("");
  byId("list-status").textContent = "Loading runs…";
  try {
    await Promise.allSettled([getJSON("/api/projects").then(result => renderProjects(result.projects)).catch(error => { byId("projects").textContent = error.message; }), refreshExplorer(), refreshTools()]);
    const result = await getJSON("/api/runs");
    const runs = result.runs;
    if (!Array.isArray(runs)) throw new Error("The server returned an invalid run list.");
    lastRuns = runs;
    renderRuns(runs);
    updateRunStatus();
    const selected = runs.find((run) => run.id === selectedId) || runs[0];
    if (selected) {
      if (selectedId === selected.id && selectedRunRecord) {
        const request = detailRequest;
        const run = await getJSON(`/api/runs/${encodeURIComponent(selected.id)}`);
        if (request === detailRequest) { renderDetail(run); await loadInspection(selected.id, request); }
      } else await selectRun(selected.id, selected.name, false);
    } else {
      selectedId = null;
      ++detailRequest;
      byId("detail").hidden = true;
    }
  } catch (error) {
    showError(error.message);
    byId("list-status").textContent = "Refresh failed.";
  } finally {
    byId("refresh").disabled = false;
    scheduleLiveUpdates();
  }
}

function updateRunStatus() {
  const busy = lastRuns.some((run) => BUSY_RUN_STATES.has(run.status));
  byId("run-count").textContent = lastRuns.length;
  byId("active-count").textContent = lastRuns.filter(run => BUSY_RUN_STATES.has(run.status)).length;
  byId("list-status").textContent = lastRuns.length
    ? `${lastRuns.length} run${lastRuns.length === 1 ? "" : "s"}${busy ? " · live updates every 15 s" : ""}`
    : "No saved runs found.";
}

function scheduleLiveUpdates(delay = LIVE_INTERVAL_MS, retry = false) {
  clearTimeout(liveTimer);
  if (retry || explorerRefreshPending || lastRuns.some((run) => BUSY_RUN_STATES.has(run.status))) liveTimer = setTimeout(pollRunProgress, delay);
}

async function pollRunProgress() {
  if (byId("refresh").disabled) return;
  const request = ++liveRequest;
  const selectedBefore = selectedId;
  const detailBefore = detailRequest;
  let delay = LIVE_INTERVAL_MS;
  let retry = false;
  try {
    const listing = await getJSON("/api/runs");
    if (request !== liveRequest) return;
    if (!Array.isArray(listing.runs)) throw new Error("Invalid run list");
    const wasBusy = new Set(lastRuns.filter((run) => BUSY_RUN_STATES.has(run.status)).map((run) => run.id));
    const settled = listing.runs.some((run) => wasBusy.has(run.id) && !BUSY_RUN_STATES.has(run.status));
    explorerRefreshPending ||= settled;
    lastRuns = listing.runs;
    const scroll = { x: window.scrollX, y: window.scrollY };
    const tableScroll = byId("runs").closest(".table-scroll");
    const tableTop = tableScroll.scrollTop;
    renderRuns(lastRuns);
    tableScroll.scrollTop = tableTop;
    updateRunStatus();
    window.scrollTo(scroll.x, scroll.y);
    if (selectedBefore && lastRuns.some((run) => run.id === selectedBefore)) {
      const run = await getJSON(`/api/runs/${encodeURIComponent(selectedBefore)}`);
      if (request === liveRequest && selectedId === selectedBefore && detailRequest === detailBefore && JSON.stringify(run) !== lastDetailJSON) {
        const currentScroll = { x: window.scrollX, y: window.scrollY };
        renderDetail(run);
        window.scrollTo(currentScroll.x, currentScroll.y);
      }
      if (request === liveRequest && selectedId === selectedBefore && detailRequest === detailBefore) await loadInspection(selectedBefore, detailBefore);
    }
    if (explorerRefreshPending && request === liveRequest) {
        await refreshExplorer();
      explorerRefreshPending = false;
    }
  } catch (_) {
    if (request === liveRequest) {
      byId("list-status").textContent = "Live updates temporarily unavailable; retrying in 60 s.";
      delay = 60000;
      retry = true;
    }
  } finally {
    if (request === liveRequest) scheduleLiveUpdates(delay, retry);
  }
}

function renderProjects(projects) {
  const container = byId("projects");
  container.replaceChildren();
  if (!projects.length) container.textContent = "No studio projects yet.";
  for (const project of projects) {
    const section = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `${project.name} · current: ${project.current_artifact_id}`;
    section.append(summary);
    const inquiry = document.createElement("p");
    inquiry.textContent = `Inquiry v${project.inquiry_versions.length}: ${project.inquiry_versions.at(-1).inquiry.question}`;
    section.append(inquiry);
    for (const episode of Object.values(project.episodes)) {
      const status = document.createElement("p");
      status.textContent = `${episode.id}: ${episode.status} · calls ${episode.model_calls}/${episode.spec.max_calls} · rounds ${episode.round}/${episode.spec.max_rounds} · actions ${episode.action_ids.length}/${episode.spec.max_actions}`;
      section.append(status);
    }
    const branches = document.createElement("ul");
    for (const action of Object.values(project.actions)) {
      const item = document.createElement("li");
      item.textContent = `${action.id} ← ${action.parent_artifact_id}: ${action.direction.intended_change} · ${action.status}`;
      for (const id of action.output_artifact_ids) {
        const link = document.createElement("a");
        link.href = API_PREFIX + `/api/projects/${encodeURIComponent(project.id)}/artifacts/${encodeURIComponent(id)}`;
        link.textContent = ` [${id}]`;
        link.target = "_blank";
        link.rel = "noopener";
        item.append(link);
      }
      branches.append(item);
    }
    section.append(branches);
    for (const episode of Object.values(project.episodes)) {
      if (["stopped", "completed"].includes(episode.status)) continue;
      const form = document.createElement("form");
      form.className = "studio-select";
      const title = document.createElement("label");
      title.textContent = `Advance in ${episode.id}`;
      const select = document.createElement("select");
      select.required = true;
      const blank = document.createElement("option");
      blank.value = "";
      blank.textContent = "Choose an artifact";
      select.append(blank);
      for (const artifact of Object.values(project.artifacts)) {
        const option = document.createElement("option");
        option.value = artifact.id;
        option.textContent = artifact.id;
        select.append(option);
      }
      title.append(select);
      const reasonLabel = document.createElement("label");
      reasonLabel.textContent = "Reason";
      const reason = document.createElement("input");
      reason.required = true;
      reason.maxLength = 8000;
      reasonLabel.append(reason);
      const button = document.createElement("button");
      button.type = "submit";
      button.textContent = "Select for development";
      const status = document.createElement("p");
      status.setAttribute("role", "status");
      form.append(title, reasonLabel, button, status);
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (!operatorName()) {
          status.textContent = "Enter an operator name above to attribute this selection.";
          byId("operator").focus();
          return;
        }
        button.disabled = true;
        try {
          await postJSON(`/api/projects/${encodeURIComponent(project.id)}/select`, { episode_id: episode.id, artifact_id: select.value, author: operatorName(), reason: reason.value });
          status.textContent = "Studio selection saved. Generation has not started.";
        } catch (error) {
          status.textContent = error.message;
          button.disabled = false;
        }
      });
      section.append(form);
    }
    for (const decision of project.decisions) {
      const note = document.createElement("p");
      note.textContent = `${decision.kind} ${decision.artifact_id}: ${decision.reason} (${decision.author})`;
      section.append(note);
    }
    const raw = document.createElement("details");
    const label = document.createElement("summary");
    label.textContent = "Project JSON";
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(project, null, 2);
    raw.append(label, pre);
    section.append(raw);
    container.append(section);
  }
}

byId("refresh").addEventListener("click", refresh);

async function postJSON(url, body) {
  const response = await fetch(scopedURL(url), { method: "POST", headers: { "Content-Type": "application/json", "X-Kusama-Review": "1" }, body: JSON.stringify(body) });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : Array.isArray(result.detail) ? result.detail.map(item => `${item.loc?.join(".")}: ${item.msg}`).join("; ") : `Could not save (HTTP ${response.status}).`);
  return result;
}

function externalLink(url, label) {
  const link = document.createElement("a");
  try {
    const parsed = new URL(url, location.origin);
    if (!["http:", "https:"].includes(parsed.protocol)) return document.createTextNode(label);
    link.href = scopedURL(url);
  } catch (_) { return document.createTextNode(label); }
  link.textContent = label; link.target = "_blank"; link.rel = "noopener noreferrer";
  return link;
}

function imageCard(url, caption, alt) {
  const figure = document.createElement("figure");
  const link = externalLink(url, "");
  const image = document.createElement("img"); image.src = scopedURL(url); image.alt = alt; image.loading = "lazy";
  link.append(image); const label = document.createElement("figcaption"); label.textContent = caption;
  figure.append(link, label); return figure;
}

async function showSource(id) {
  activatePanel("sources");
  byId("explorer-status").textContent = "Loading source details…";
  try {
    const result = await getJSON(`/api/sources/${encodeURIComponent(id)}`);
    const ref = result.reference;
    byId("source-select").value = id;
    const holder = byId("source-detail"); holder.replaceChildren();
    const title = document.createElement("h3"); title.textContent = ref.title;
    const credit = document.createElement("p"); credit.textContent = [ref.maker, ref.date, ref.medium, ref.attribution].filter(Boolean).join(" · ");
    const description = document.createElement("p"); description.textContent = ref.description || "No descriptive text supplied.";
    const grid = document.createElement("div"); grid.className = "explorer-images";
    grid.append(imageCard(`/api/sources/${encodeURIComponent(id)}/image`, "Whole source", ref.title));
    for (const region of result.regions) grid.append(imageCard(`/api/source-regions/${encodeURIComponent(region.id)}/image`, `${region.id} · ${region.localization} · ${region.box.join(", ")}`, `Detail of ${ref.title}`));
    holder.append(title, credit, externalLink(ref.url, "Original collection record"), description, grid);
    for (const claim of result.claims) {
      const paragraph = document.createElement("p"); paragraph.textContent = `${claim.status} (${claim.author}): ${claim.claim} `;
      paragraph.append(externalLink(claim.supporting_url, "Source evidence")); holder.append(paragraph);
    }
    const rights = document.createElement("details"); const summary = document.createElement("summary"); summary.textContent = "Rights, source provenance and exact hashes";
    const pre = document.createElement("pre"); pre.textContent = JSON.stringify({ rights: ref.rights, original: ref.original, view: ref.view, split_group: ref.split_group, field_provenance: ref.field_provenance }, null, 2);
    rights.append(summary, pre); holder.append(rights);
    byId("explorer-status").textContent = "Documented source context and proposed interpretations retain separate attribution.";
  } catch (error) { byId("explorer-status").textContent = error.message; }
}

async function refreshExplorer() {
  await refreshResearch();
  try {
    const [continuations, inspirations, sources] = await Promise.all([getJSON("/api/continuations"), getJSON("/api/inspirations"), getJSON("/api/sources")]);
    const gallery = byId("continuation-gallery"); gallery.replaceChildren();
    if (!continuations.continuations.length) gallery.textContent = "No selected-parent continuations registered yet.";
    for (const item of continuations.continuations) {
      const row = document.createElement("p");
      row.append(externalLink(`/api/continuations/${encodeURIComponent(item.id)}/gallery`, item.request.title), ` · ${item.status} · ${item.model_calls}/2 calls · selection by ${item.selection.author} (${item.selection.origin})`);
      const inspect = document.createElement("button"); inspect.textContent = "Open controls"; inspect.addEventListener("click", () => openContinuation(item.id)); row.append(inspect);
      gallery.append(row);
    }
    const proposals = byId("inspiration-list"); proposals.replaceChildren();
    if (!inspirations.inspirations.length) proposals.textContent = "No connection proposals saved yet.";
    for (const item of inspirations.inspirations) {
      const details = document.createElement("details"); const summary = document.createElement("summary"); summary.textContent = `${item.brief?.question || item.brief?.title || text(item.brief)} · ${item.status}`;
      details.append(summary);
      if (item.images?.length) {
        const grid = document.createElement("div"); grid.className = "explorer-images";
        for (const image of item.images) grid.append(imageCard(`/api/inspirations/${encodeURIComponent(item.directory_id)}/images/${encodeURIComponent(image.id)}`, `${image.id} · ${image.connection_id} · ${image.origin}`, `Realized work ${image.id}`));
        details.append(grid);
      }
      const shelfDetails = document.createElement("details"); const shelfSummary = document.createElement("summary"); shelfSummary.textContent = "Reference encounter shelf and candidate-set evidence";
      shelfDetails.append(shelfSummary); details.append(shelfDetails);
      let shelfLoaded = false;
      shelfDetails.addEventListener("toggle", async () => {
        if (!shelfDetails.open || shelfLoaded) return;
        shelfLoaded = true;
        try {
          const saved = await getJSON(`/api/inspirations/${encodeURIComponent(item.directory_id)}`);
          const shelf = saved.shelf_data || {};
          const evidence = document.createElement("p"); evidence.textContent = `${(shelf.selected || []).length}/${shelf.requested_count || "?"} references returned. Shortages: ${text(shelf.shortages)}. ${shelf.fusion || ""}`;
          shelfDetails.append(evidence);
          const grid = document.createElement("div"); grid.className = "explorer-images";
          for (const row of shelf.selected || []) {
            const url = row.region_id ? `/api/source-regions/${encodeURIComponent(row.region_id)}/image` : `/api/sources/${encodeURIComponent(row.reference_id)}/image`;
            const card = imageCard(url, `${row.id} · ${row.allocation} · channel ranks ${text(row.ranks)}`, `Reference ${row.id}`);
            const button = document.createElement("button"); button.type = "button"; button.textContent = "Source context and exact details"; button.addEventListener("click", () => showSource(row.reference_id)); card.append(button); grid.append(card);
          }
          shelfDetails.append(grid);
          const raw = document.createElement("details"); const label = document.createElement("summary"); label.textContent = "Full candidate channels and admitted-query record";
          const pre = document.createElement("pre"); pre.textContent = JSON.stringify(shelf, null, 2); raw.append(label, pre); shelfDetails.append(raw);
        } catch (error) { shelfLoaded = false; const message = document.createElement("p"); message.textContent = error.message; shelfDetails.append(message); }
      });
      for (const connection of item.connections || []) {
        const card = document.createElement("article"); const heading = document.createElement("h3"); heading.textContent = connection.title;
        card.append(heading);
        for (const [label, value] of [["Proposed relationship", connection.shared_relationship], ["Transformation", connection.transformation], ["Visible test", connection.visible_test], ["Uncertainty", connection.uncertainty]]) {
          const paragraph = document.createElement("p"); paragraph.textContent = `${label}: ${text(value)}`; card.append(paragraph);
        }
        const author = document.createElement("p"); author.textContent = `Proposal attribution: ${connection.authorship || "recorded model proposal"}. Artistic outcome is separate.`; card.append(author);
        for (const source of connection.sources || []) {
          const button = document.createElement("button"); button.type = "button";
          button.textContent = `${source.reference_id}${source.region_id ? ` · ${source.region_id}` : ""}: ${source.role}`;
          button.addEventListener("click", () => { showSource(source.reference_id); byId("source-detail").scrollIntoView({ behavior: "smooth" }); }); card.append(button);
        }
        details.append(card);
      }
      proposals.append(details);
    }
    const selected = byId("source-select").value; byId("source-select").replaceChildren();
    const blank = document.createElement("option"); blank.value = ""; blank.textContent = "Choose a source"; byId("source-select").append(blank);
    for (const source of sources.sources) {
      const option = document.createElement("option"); option.value = source.id; option.textContent = `${source.title} · ${source.maker || source.provider}`; byId("source-select").append(option);
    }
    byId("source-select").value = selected;
  } catch (error) { byId("explorer-status").textContent = error.message; }
}
byId("source-select").addEventListener("change", (event) => { if (event.target.value) showSource(event.target.value); });

function researchText(holder, label, value) {
  if (value === undefined || value === null || value === "") return;
  const paragraph = document.createElement("p");
  const prefix = document.createElement("strong"); prefix.textContent = `${label}: `;
  if (label) paragraph.append(prefix);
  paragraph.append(text(value)); holder.append(paragraph);
}

function researchEvidence(holder, label, value) {
  const details = document.createElement("details");
  const summary = document.createElement("summary"); summary.textContent = label;
  const pre = document.createElement("pre"); pre.textContent = JSON.stringify(value, null, 2);
  details.append(summary, pre); holder.append(details);
}

// Saved model/source text is untrusted. Present it as text, never HTML/Markdown.
function readableValue(holder, value) {
  if (value === null || value === undefined) {
    const empty = document.createElement("p"); empty.textContent = "(not captured)"; holder.append(empty);
  } else if (Array.isArray(value)) {
    if (!value.length) { const empty = document.createElement("p"); empty.textContent = "(empty)"; holder.append(empty); return; }
    const list = document.createElement("ol");
    for (const item of value) { const row = document.createElement("li"); readableValue(row, item); list.append(row); }
    holder.append(list);
  } else if (typeof value === "object") {
    if (!Object.keys(value).length) { readableValue(holder, "(empty)"); return; }
    const fields = document.createElement("dl"); fields.className = "transcript-fields";
    for (const [key, item] of Object.entries(value)) {
      const label = document.createElement("dt"); label.textContent = key.replaceAll("_", " ").replace(/^./, c => c.toUpperCase());
      const content = document.createElement("dd");
      if (key === "output_schema") readableSection(content, "Response format requirements", item, true);
      else readableValue(content, item);
      fields.append(label, content);
    }
    holder.append(fields);
  } else {
    const paragraph = document.createElement("p"); paragraph.className = "transcript-text";
    paragraph.textContent = value === "" ? "(empty)" : String(value); holder.append(paragraph);
  }
}

function readableSection(holder, label, value, collapsible = false) {
  const section = document.createElement(collapsible ? "details" : "section"); section.className = "transcript-section";
  const heading = document.createElement(collapsible ? "summary" : "h4"); heading.textContent = label;
  section.append(heading); readableValue(section, value); holder.append(section);
  return section;
}

function renderResearchTranscript(holder, step, transcript, fixture) {
  const actor = fixture ? "fixture actor" : "Qwen";
  const prompt = transcript?.prompt;
  if (prompt) {
    const section = document.createElement("details"); section.className = "transcript-section";
    const heading = document.createElement("summary"); heading.textContent = `Prompt sent to ${actor} · Step ${Number(step.index) + 1}`;
    section.append(heading);
    readableSection(section, "System instructions", prompt.system ?? "Not recorded", true);
    readableSection(section, "User prompt · saved admitted context", prompt.payload ?? "Not recorded");
    readableSection(section, "Image inputs", prompt.images, true);
    if (transcript?.response && "prompt" in transcript.response) readableSection(section, "Exact serialized prompt", transcript.response.prompt, true);
    if (prompt.context_admission) readableSection(section, "Context admission and omissions", prompt.context_admission, true);
    holder.append(section);
  } else researchText(holder, "Prompt", transcript?.prompt_error || "No saved prompt is available for this step.");
  if (transcript?.draft_diff) readableSection(holder, "Changes from previous draft", transcript.draft_diff, true);
  if (transcript?.tokens) readableSection(holder, "Step tokens and projection removals", transcript.tokens, true);
  const response = transcript?.response;
  if (response && "raw_response" in response && response.raw_response === "") readableSection(holder, "Original response text", "");
  if (response?.raw_response !== undefined && response.raw_response !== null && response.raw_response !== "") {
    let value = response.raw_response;
    try { value = JSON.parse(value); } catch (_) { /* Preserve malformed replies verbatim. */ }
    readableSection(holder, `Response from ${actor}${step.status === "response_rejected" ? " · rejected before execution" : ""}`, value);
    readableSection(holder, "Original response text", response.raw_response, true);
  } else if (response?.output || step.action) {
    readableSection(holder, "Saved structured action · original response text unavailable", response?.output || step.action);
  } else researchText(holder, "Response", transcript?.response_error || "No response text was saved.");
  if (response?.parse_error || response?.inference_error) researchText(holder, "Response error", response.parse_error || response.inference_error);
  if (step.chance_exposure) readableSection(holder, "Chance exposure · optional source leads", { ...step.chance_exposure, sources: step.tool_result?.sources });
  if (step.tool_result !== undefined) readableSection(holder, "Tool response · returned evidence", step.tool_result, true);
  if (step.tool_error) readableSection(holder, "Tool response · error", step.tool_error);
}

function renderImageTranscript(holder, record) {
  const action = record?.saved_action || {};
  const prompt = action.prompt ?? record?.prompt ?? record?.edit_instruction ?? record?.instruction;
  readableSection(holder, "Prompt sent to FLUX", prompt ?? "No generation prompt was saved for this image.");
  researchText(holder, "Response", "Saved image shown above.");
  if (!record) return;
  researchText(holder, "Output attribution", record.origin);
  researchText(holder, "Seed", record.seed ?? action.seed);
  researchText(holder, "Image size", record.size?.join(" × "));
  researchText(holder, "Parent image", record.parent_image_id);
  researchText(holder, "Selected sources", (action.source_ids || record.source_ids)?.join(", "));
  const conditioning = record.conditioning || action.conditioning;
  if (conditioning) {
    researchText(holder, "FLUX image inputs", conditioning.mode === "text_only" ? "Text only" :
      conditioning.images.map(image => `Image ${image.input_index}: ${image.role === "parent" ? "artwork being edited" : image.source_id + " · view " + image.image_index + " · " + image.purpose}`).join("\n"));
    researchEvidence(holder, "Exact input images and hashes", conditioning);
  }
  researchEvidence(holder, "Exact image record and lineage", record);
}

function renderResearch(holder, session) {
  researchText(holder, "Result", session.result || "FAIL · not evaluated");
  if (session.acceptance_criteria) readableSection(holder, "Acceptance criteria", session.acceptance_criteria, true);
  researchText(holder, "Decision input", session.input_mode);
  for (const label of session.commit_labels || []) researchText(holder, "Commit label", label);
  researchText(holder, "Starting problem", session.problem);
  researchText(holder, "Problem authorship", session.authorship?.problem || session.authorship);
  researchText(holder, "Current research question", session.question);
  researchText(holder, "Execution", `${session.status} · ${session.reasoning_calls ?? 0} reasoning calls${session.fixture ? " · fixture; no model inference" : ""}`);
  if (session.network) researchText(holder, "Source acquisition", `${session.network.requests ?? 0} network requests · ${session.network.charged_network_bytes ?? 0} recorded bytes · ${session.network.acquisitions ?? 0} acquisitions`);
  const plan = session.frozen_plan || {};
  if (plan.snapshots?.parent) {
    researchText(holder, "Development research", "This session begins from a saved parent image. Its questions and reference choices remain linked to that decision-time parent.");
    researchEvidence(holder, "Saved parent and research trigger", { parent: plan.snapshots.parent, prior_context: plan.request?.prior_context });
  }
  const heading = document.createElement("h3"); heading.textContent = "Questions, searches and decisions"; holder.append(heading);
  const timeline = document.createElement("ol"); timeline.className = "research-timeline";
  for (const step of session.steps || []) {
    const row = document.createElement("li");
    const action = step.action || {};
    researchText(row, `Step ${Number(step.index) + 1}`, action.kind || action.type || action.action || (step.status === "response_rejected" ? "Model response rejected before action execution" : "Recorded action"));
    for (const [label, field] of [["Question", "question"], ["Query", "query"], ["Reason", "reason"], ["Source", "source_id"], ["Decision", "decision"], ["Note", "note"]]) researchText(row, label, action[field]);
    if (step.action && action.reason == null) researchText(row, "Reason", session.fixture ? "Not supplied by fixture actor" : "Not supplied by model");
    researchText(row, "Choice attribution", session.fixture ? "fixture actor" : (step.origin || action.origin || "model proposal"));
    researchText(row, "Recorded failure", step.tool_error || step.error);
    renderResearchTranscript(row, step, (session.transcripts || []).find(item => item.index === step.index), session.fixture);
    researchEvidence(row, "Exact action, returned evidence and viewed images", step);
    timeline.append(row);
  }
  if (!timeline.children.length) researchText(holder, "Research history", "No research actions recorded yet.");
  holder.append(timeline);

  const proposals = document.createElement("h3"); proposals.textContent = "Artistic hypotheses and chosen reference sets"; holder.append(proposals);
  for (const hypothesis of session.hypotheses || []) {
    const card = document.createElement("article");
    const title = document.createElement("h3"); title.textContent = hypothesis.title || hypothesis.id; card.append(title);
    for (const [label, field] of [["Inquiry", "question"], ["Proposed relationship", "relationship"], ["Transformation", "transformation"], ["Composition", "composition"], ["Visible test", "visible_test"], ["Uncertainty", "uncertainty"]]) researchText(card, label, hypothesis[field]);
    readableSection(card, "Image generation prompt", hypothesis.render_instruction ?? "Not recorded");
    researchText(card, "Proposal attribution", session.fixture ? "fixture actor" : hypothesis.origin);
    for (const source of hypothesis.sources || []) {
      researchText(card, `Reference ${source.source_id}`, source.role);
      researchText(card, "Property to transfer", source.transferred_property);
      researchText(card, "Property to discard", source.discarded_property);
    }
    researchEvidence(card, "Rendering instruction and exact source snapshots", hypothesis);
    holder.append(card);
  }
  if (!(session.hypotheses || []).length) researchText(holder, "Hypotheses", session.status === "response_rejected" ? "Research stopped after a rejected model response; no artistic hypothesis was committed." : "No artistic hypothesis saved yet.");
  const images = document.createElement("div"); images.className = "explorer-images";
  for (const asset of session.evidence_images || []) {
    const source = (session.source_history || []).find(item => item.source_id === asset.source_id) || {};
    const ref = source.reference || {};
    const title = asset.scope === "parent" ? "Saved parent" : (source.title || ref.title || asset.source_id || "Viewed evidence");
    const caption = asset.scope === "parent" ? title : `${title} · ${asset.scope || "viewed evidence"}`;
    const card = imageCard(asset.url, caption, `Research evidence ${title}`);
    const credit = [ref.maker, ref.date, ref.culture, ref.medium].filter(Boolean).join(" · ");
    if (credit) researchText(card, "Collection metadata", credit);
    if (source.text) researchText(card, "Source excerpt", source.text.slice(0, 900) + (source.text.length > 900 ? "… (full saved context below)" : ""));
    if (ref.url || source.url) card.append(externalLink(ref.url || source.url, "Original source record"));
    for (const claim of (source.claims || []).slice(0, 5)) researchText(card, `${claim.status || "Attributed claim"} (${claim.author || "source"})`, claim.claim);
    researchEvidence(card, "Provenance, rights and exact image hash", asset);
    images.append(card);
  }
  holder.append(images);
  researchEvidence(holder, "Research notebook, saved encounters and rejected leads", session.notebook || []);
  researchEvidence(holder, "Source history and attributed context", session.source_history || []);
  researchEvidence(holder, "Final selected references", session.selected_sources || (session.hypotheses || []).map(hypothesis => ({ hypothesis_id: hypothesis.id, sources: hypothesis.sources })));
  researchEvidence(holder, "Frozen problem, available tools, resource limits and operator policies", plan);
  researchEvidence(holder, "Complete verified research report", session);
}

async function refreshResearch() {
  const holder = byId("research-list");
  try {
    const result = await getJSON("/api/research");
    if (!Array.isArray(result.research)) throw new Error("Invalid research session list");
    holder.replaceChildren();
    byId("research-status").textContent = result.research.length ? `${result.research.length} saved research sessions. Open one to inspect its evidence.` : "No research sessions saved yet.";
    for (const session of result.research) {
      const details = document.createElement("details"); details.className = "research-session";
      const summary = document.createElement("summary");
      summary.textContent = `${session.id || session.directory} · ${session.status} · ${session.steps ?? 0} steps · ${session.hypotheses ?? 0} hypotheses${session.fixture ? " · fixture" : ""}`;
      details.append(summary);
      const content = document.createElement("div"); details.append(content);
      let loaded = false;
      details.addEventListener("toggle", async () => {
        if (!details.open || loaded) return;
        loaded = true; content.textContent = "Loading verified research evidence…";
        try {
          const report = await getJSON(`/api/research/${encodeURIComponent(session.directory_id)}`);
          content.replaceChildren(); renderResearch(content, report);
        } catch (error) { loaded = false; content.textContent = error.message; }
      });
      holder.append(details);
    }
  } catch (error) { byId("research-status").textContent = error.message; }
}


function operatorName() { return byId("operator").value.trim(); }
const panelTitles = { new: "Studio", runs: "Runs", development: "Development", research: "Research", sources: "Sources", tools: "Tools & system" };
function activatePanel(name) {
  if (!panelTitles[name]) name = "new";
  window.scrollTo?.(0, 0);
  for (const panel of document.querySelectorAll("[data-view]")) panel.hidden = panel.dataset.view !== name;
  for (const link of document.querySelectorAll("[data-panel]")) {
    if (link.dataset.panel === name) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  }
  byId("page-title").textContent = panelTitles[name];
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
}
window.addEventListener("hashchange", () => activatePanel(location.hash.slice(1)));
for (const link of document.querySelectorAll("[data-panel]")) link.addEventListener("click", () => activatePanel(link.dataset.panel));
byId("run-search").addEventListener("input", () => renderRuns(lastRuns));
byId("run-status-filter").addEventListener("change", () => renderRuns(lastRuns));

let workflowSchemas = {};
let workspaceRoot = "";
async function refreshTools() {
  try {
    const [result, schemas] = await Promise.all([getJSON("/api/tools"), getJSON("/api/contracts")]);
    const holder = byId("workflow-tools"); holder.replaceChildren();
    for (const tool of result.tools) {
      const card = document.createElement("article"); card.className = "tool-card";
      const heading = document.createElement("h3"); heading.textContent = tool.title;
      const description = document.createElement("p"); description.textContent = tool.description;
      const command = document.createElement("pre"); command.textContent = tool.command;
      const copy = document.createElement("button"); copy.textContent = "Copy command";
      copy.addEventListener("click", async () => {
        try { await navigator.clipboard.writeText(tool.command); copy.textContent = "Copied"; }
        catch (_) { copy.textContent = "Select command to copy"; }
      });
      card.append(heading, description, command, copy);
      if (tool.guide) card.append(externalLink(tool.guide, "Guide ↗"));
      holder.append(card);
    }
    workflowSchemas = schemas;
    updatePlanSchema();
    byId("contracts").textContent = JSON.stringify(schemas, null, 2);
    byId("prepare-schema").textContent = JSON.stringify(schemas.ContinuationCreate, null, 2);
    byId("decision-schema").textContent = JSON.stringify(schemas.DevelopmentDecision, null, 2);
    const system = await getJSON("/api/system");
    workspaceRoot = system.artifact_root;
    updatePlanSchema();
    byId("system-status").replaceChildren();
    researchEvidence(byId("system-status"), "Workspace & resource sessions", system);
  } catch (error) { byId("system-status").textContent = error.message; }
}
let activeContinuation = null;
let continuationRequest = 0;
const dispatchIds = new Map();
async function openContinuation(id) {
  activeContinuation = id;
  const request = ++continuationRequest;
  byId("continuation-controls").hidden = false;
  byId("continuation-controls").scrollIntoView({ behavior: "smooth", block: "start" });
  byId("continuation-title").textContent = id;
  byId("continuation-status").textContent = "Loading continuation…";
  byId("execute-continuation").disabled = true;
  try {
    const result = await getJSON(`/api/continuations/${encodeURIComponent(id)}/status`);
    if (request !== continuationRequest) return;
    const item = result.continuation;
    byId("continuation-title").textContent = item.request.title || id;
    byId("continuation-status").textContent = `${item.status} · ${text(item.model_calls)} calls${result.dispatch ? ` · dispatch ${result.dispatch.state}` : ""}`;
    byId("continuation-raw").textContent = JSON.stringify(result, null, 2);
    byId("continuation-summary").replaceChildren();
    researchText(byId("continuation-summary"), "Instruction", item.request.revision?.instruction);
    researchText(byId("continuation-summary"), "Selection", `${item.selection.author} (${item.selection.origin}) · ${item.selection.reason}`);
    byId("continuation-gallery-link").href = API_PREFIX + `/api/continuations/${encodeURIComponent(id)}/gallery`;
    if (result.dispatch?.submission_id) dispatchIds.set(id, result.dispatch.submission_id);
    byId("execute-continuation").disabled = ["completed", "stopped"].includes(item.status) || item.model_calls >= 2 || ["queued", "running", "completed"].includes(result.dispatch?.state);
  } catch (error) { if (request === continuationRequest) byId("continuation-status").textContent = error.message; }
}
byId("continuation-refresh").addEventListener("click", () => { if (activeContinuation) openContinuation(activeContinuation); });
byId("prepare-toggle").addEventListener("click", () => { byId("prepare-form").hidden = !byId("prepare-form").hidden; });
byId("prepare-form").addEventListener("submit", async event => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]'); button.disabled = true;
  try {
    const body = JSON.parse(byId("prepare-json").value);
    await postJSON("/api/continuations", body);
    byId("prepare-status").textContent = "Continuation prepared.";
    await refreshExplorer(); await openContinuation(body.id);
  } catch (error) { byId("prepare-status").textContent = error.message; }
  finally { button.disabled = false; }
});
byId("execute-continuation").addEventListener("click", async () => {
  if (!activeContinuation) return;
  const id = activeContinuation;
  if (!dispatchIds.has(id)) dispatchIds.set(id, crypto.randomUUID());
  byId("execute-continuation").disabled = true;
  try {
    await postJSON(`/api/continuations/${encodeURIComponent(id)}/execute`, { submission_id: dispatchIds.get(id) });
    byId("continuation-action-status").textContent = "Execution request submitted.";
  } catch (error) { byId("continuation-action-status").textContent = error.message; }
  await openContinuation(id);
});
for (const [formId, action] of [["observation-form", "observations"], ["decision-form", "decisions"]]) {
  byId(formId).addEventListener("submit", async event => {
    event.preventDefault();
    if (!activeContinuation) return;
    const id = activeContinuation;
    const button = event.currentTarget.querySelector('button[type="submit"]'); button.disabled = true;
    try {
      let body;
      if (action === "observations") {
        if (!operatorName()) throw new Error("Enter an operator name to attribute this observation.");
        body = { ...Object.fromEntries(new FormData(event.currentTarget)), author: operatorName() };
      } else body = JSON.parse(byId("decision-json").value);
      await postJSON(`/api/continuations/${encodeURIComponent(id)}/${action}`, body);
      byId("continuation-action-status").textContent = action === "observations" ? "Observation saved." : "Decision saved.";
      await openContinuation(id);
    } catch (error) { byId("continuation-action-status").textContent = error.message; }
    finally { button.disabled = false; }
  });
}
activatePanel(location.hash.slice(1));
async function initialize() {
  try {
    const result = await getJSON("/api/workspaces");
    const select = byId("workspace"); select.replaceChildren();
    for (const item of result.workspaces) {
      const option = document.createElement("option"); option.value = item.url; option.textContent = item.label; select.append(option);
    }
    select.value = (API_PREFIX || "") + "/";
  } catch (error) { showError(error.message); }
  await Promise.all([refresh(), loadStartOptions()]);
}
byId("workspace").addEventListener("change", () => { location.href = byId("workspace").value + location.hash; });

function updatePlanSchema() {
  const names = { research: "ResearchRequest", development: "LoopRequest", research_render: "ResearchRenderRequest" };
  const workflow = byId("plan-workflow").value;
  byId("workflow-plan-schema").textContent = JSON.stringify(workflowSchemas[names[workflow]] || {}, null, 2);
  byId("plan-directory").textContent = `Output: ${workspaceRoot}/${workflow}/${byId("plan-id").value || "<new plan ID>"}. Preparation freezes inputs; execution remains a separate command.`;
}
byId("plan-workflow").addEventListener("change", updatePlanSchema);
byId("plan-id").addEventListener("input", updatePlanSchema);
byId("workflow-plan-form").addEventListener("submit", async event => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]'); button.disabled = true;
  try {
    const result = await postJSON("/api/plans", { workflow: byId("plan-workflow").value, id: byId("plan-id").value, request: JSON.parse(byId("workflow-plan-json").value) });
    byId("workflow-plan-status").textContent = `Prepared ${result.directory}`;
  } catch (error) { byId("workflow-plan-status").textContent = error.message; }
  finally { button.disabled = false; }
});

let startSubmission = null;
let startTokenLimit = null;
let startOutputTokens = null;
let activeFileRequest = 0;
let activePreviewFile = null;
async function loadStartOptions() {
  const button = byId("start-run"); button.disabled = true;
  try {
    const result = await getJSON("/api/start-options");
    startTokenLimit = result.total_token_limit;
    startOutputTokens = result.reserved_output_tokens;
    byId("new-run-profile").replaceChildren();
    for (const profile of result.profiles) byId("new-run-profile").append(new Option(`${profile.width} × ${profile.height} · ${profile.id}`, profile.id));
    byId("new-run-profile").value = result.default_profile;
    updateStartSummary();
    button.disabled = false;
  } catch (error) { byId("start-run-status").textContent = error.message; }
}
function updateStartSummary() {
  if (startTokenLimit === null) return;
  const steps = Number(byId("new-run-steps").value);
  const images = Number(byId("new-run-directions").value) * Number(byId("new-run-images").value);
  const development = Number(byId("new-run-development-steps").value);
  const research = Number(byId("new-run-development-research").value);
  const chance = Number(byId("new-run-chance").value);
  byId("new-run-development-steps").setCustomValidity(images + development > 12 ? "Initial images plus development actions must total at most 12." : "");
  byId("run-settings-summary").textContent = ` · ${images} study images · up to ${development} development actions · ${byId("new-run-minutes").value} min`;
  byId("run-authorization").textContent = `Start authorizes up to ${steps} initial research calls, ${images} initial inspections, ${development + 1} development choices, ${development} new-image inspections and one optional research session of ${research} calls (${((steps + images + 2 * development + 1 + research) * startTokenLimit).toLocaleString()} Qwen tokens). Each Qwen call allows ${startTokenLimit.toLocaleString()} tokens: up to ${(startTokenLimit - startOutputTokens).toLocaleString()} input and ${startOutputTokens.toLocaleString()} output. FLUX: up to ${images + development} images. Online collections: up to 64 requests and 16 acquisitions across initial and reopened research. Development actions choose either research or rendering; the final choice pauses or stops. Seed variants do not count as distinct studies.`;
  byId("run-authorization").textContent += chance ? ` Each research session also offers up to ${chance} random source leads within those limits. The model may discard them.` : " Chance leads are off.";
}
byId("start-run-form").addEventListener("input", () => { startSubmission = null; updateStartSummary(); });
byId("start-run-form").addEventListener("submit", async event => {
  event.preventDefault();
  const button = byId("start-run"); button.disabled = true;
  byId("start-run-status").textContent = "Starting your run…";
  const body = { submission_id: startSubmission ||= crypto.randomUUID(), question: byId("run-question").value.trim(),
    title: byId("new-run-title").value.trim(), context: byId("new-run-context").value.trim(),
    render_profile: byId("new-run-profile").value,
    seed_artists: byId("new-run-seed-artists").value.split(";").map(s => s.trim()).filter(Boolean),
    seed_works: byId("new-run-seed-works").value.split(";").map(s => s.trim()).filter(Boolean),
    medium_constraint: byId("new-run-medium").value.trim(),
    research_steps: Number(byId("new-run-steps").value), directions: Number(byId("new-run-directions").value),
    images_per_direction: Number(byId("new-run-images").value), minutes: Number(byId("new-run-minutes").value),
    development_steps: Number(byId("new-run-development-steps").value),
    development_research_steps: Number(byId("new-run-development-research").value),
    random_exposure_count: Number(byId("new-run-chance").value),
    total_token_limit: startTokenLimit, allow_network: true, study_development: true };
  try {
    const result = await postJSON("/api/start-run", body);
    byId("start-run-status").textContent = "Run created. Follow its progress in the run workspace.";
    const listing = await getJSON("/api/runs"); lastRuns = listing.runs; renderRuns(lastRuns); updateRunStatus();
    await selectRun(result.id, result.title); scheduleLiveUpdates();
  } catch (error) { byId("start-run-status").textContent = error.message; }
  finally { button.disabled = false; }
});
function renderRecentInquiries(runs) {
  const holder = byId("recent-inquiries"); holder.replaceChildren();
  const recent = runs.filter(run => run.kind === "artistic_studio_run" || run.title).slice(0, 4);
  if (!recent.length) { holder.textContent = "Your inquiries will appear here."; return; }
  for (const run of recent) {
    const card = document.createElement("div"); card.className = "recent-inquiry-card";
    const button = document.createElement("button"); button.className = "recent-inquiry";
    const heading = document.createElement("strong"); heading.textContent = run.title || run.name;
    const status = document.createElement("span"); status.textContent = text(run.status).replaceAll("_", " ");
    button.append(heading, status); button.addEventListener("click", () => selectRun(run.id, run.name));
    card.append(button);
    if (canStopRun(run)) card.append(createStopButton(run));
    holder.append(card);
  }
}
async function loadInspection(id, request) {
  try {
    const inspection = await getJSON(`/api/runs/${encodeURIComponent(id)}/inspect`);
    if (selectedId !== id || detailRequest !== request) return;
    const serialized = JSON.stringify(inspection);
    if (lastInspectionJSON === serialized) return;
    renderInspection(inspection);
    lastInspectionJSON = serialized;
  } catch (error) { if (selectedId === id && detailRequest === request) byId("detail-status").textContent = `Some run details are unavailable: ${error.message}`; }
}
function renderInspection(result) {
  const images = result.images || [];
  const research = [...(result.research || [])].sort((a, b) => Number(Boolean(a.frozen_plan?.request?.initial_question)) - Number(Boolean(b.frozen_plan?.request?.initial_question)));
  inspectedFiles = result.files || [];
  byId("run-image-count").textContent = images.length || "";
  byId("research-step-count").textContent = research.reduce((n, session) => n + (session.steps || []).length, 0) || "";
  for (const target of ["images"]) {
    const holder = byId(target);
    const existing = new Map([...holder.children].map(node => [node.dataset.url, node]));
    const visible = [...images].sort((a, b) => feedTime(b.modified_at) - feedTime(a.modified_at));
    const wanted = new Set(visible.map(file => file.url));
    for (const node of [...holder.children]) if (!wanted.has(node.dataset.url)) node.remove();
    for (const file of visible) {
      const card = existing.get(file.url) || imageCard(file.url, file.label || file.name, file.label || file.name);
      card.dataset.url = file.url;
      card.querySelector("figcaption").textContent = file.label || file.name;
      const recordJSON = JSON.stringify(file.record || null);
      if (target === "images" && card.dataset.record !== recordJSON) {
        const prior = card.querySelector("div"); const open = prior?.querySelector("details")?.open; prior?.remove();
        const transcript = document.createElement("div"); transcript.className = "image-transcript";
        renderImageTranscript(transcript, file.record); card.append(transcript);
        if (transcript.querySelector("details")) transcript.querySelector("details").open = Boolean(open);
        card.dataset.record = recordJSON;
      }
      if (!existing.has(file.url)) holder.append(card);
    }
    if (!visible.length && !holder.children.length) {
      const empty = document.createElement("p"); empty.className = "empty"; empty.textContent = "Images will appear here as they are saved."; holder.append(empty);
    }
  }
  const unsaved = byId("generations-without-images"); unsaved.replaceChildren();
  for (const generation of result.generations_without_images || []) {
    const card = document.createElement("article");
    const title = document.createElement("h3"); title.textContent = generation.label; card.append(title);
    readableSection(card, "Saved FLUX prompt · planned generation", generation.action.prompt);
    researchText(card, "Response", generation.error ? "Generation failed; no image was saved." : "No saved image response yet; execution is not confirmed by this plan.");
    researchText(card, "Recorded failure", generation.error);
    researchText(card, "Seed", generation.action.seed);
    researchText(card, "Directory", generation.directory); unsaved.append(card);
  }
  const cycleWorkflow = renderCycleWorkspace(result);
  const stages = byId("run-stage-list"); stages.replaceChildren();
  const pendingStage = BUSY_RUN_STATES.has(selectedRunRecord?.status) ? "Pending" : "Not reached";
  const stageData = cycleWorkflow ? [] : selectedRunRecord?.kind === "artistic_studio_run"
    ? [["Question", "Saved", "overview"], ["Research", research[0]?.status || (selectedRunRecord?.status === "researching" ? "In progress" : pendingStage), "research"], ["Images", images.length ? `${images.length} saved` : selectedRunRecord?.status === "rendering" ? "In progress" : pendingStage, "images"]]
    : (result.stages || []).slice(0, 8).map(stage => [stage.directory, stage.status, "files"]);
  if (!cycleWorkflow && result.study_development?.length) stageData.push(["Study inspection & choice", result.study_development.at(-1).status, "overview"]);
  for (const [label, status, tabName] of stageData) {
    const button = document.createElement("button");
    const title = document.createElement("strong"); title.textContent = label;
    const state = document.createElement("span"); state.textContent = text(status).replaceAll("_", " "); button.append(title, state);
    button.addEventListener("click", () => openRunSection(tabName)); stages.append(button);
  }
  const directions = byId("run-directions"); directions.replaceChildren();
  for (const session of research) for (const hypothesis of session.hypotheses || []) {
    const card = document.createElement("details");
    const summary = document.createElement("summary"); summary.textContent = hypothesis.title || hypothesis.id; card.append(summary);
    researchText(card, "Question", hypothesis.question);
    researchText(card, "Provisional operation", hypothesis.artistic_operation || hypothesis.transformation);
    researchText(card, "What making it could reveal", hypothesis.study_reason);
    researchText(card, "Difference from other studies", hypothesis.distinct_from_previous);
    researchText(card, "Proposed relationship", hypothesis.relationship);
    researchText(card, "Image instruction", hypothesis.render_instruction);
    researchText(card, "Attribution", hypothesis.origin || "Model proposal"); directions.append(card);
  }
  for (const development of result.study_development || []) {
    if (cycleWorkflow) continue;
    const card = document.createElement("article");
    researchText(card, "Study development · model interpretation", development.reason);
    for (const observation of development.observations || []) {
      const detail = document.createElement("details");
      const title = document.createElement("summary"); title.textContent = `Inspection · ${observation.image_id}`; detail.append(title);
      for (const [key, label] of [["visible_relationships", "Visible relationships"], ["unexpected_effects", "Unexpected effects"], ["explanation_gaps", "Gaps in the explanation"], ["uncertainty", "Uncertainty"], ["possible_next_question", "Next question"]]) researchText(detail, label, Array.isArray(observation[key]) ? observation[key].join("\n") || "None reported" : observation[key]);
      card.append(detail);
    }
    if (development.choice) {
      researchText(card, "Artist's choice", development.choice.action);
      researchText(card, "Commitments", development.choice.commitments);
      researchText(card, "What seeing the images changed", development.choice.what_changed);
      researchText(card, "Reason", development.choice.reason);
      researchText(card, "Selected images", development.choice.selected_image_ids.join(", ") || "None");
      researchText(card, "Revised question", development.choice.new_question);
    }
    if (development.followup_observation) readableSection(card, "Follow-up inspection · model interpretation", development.followup_observation);
    for (const cycle of development.cycles || []) {
      const detail = document.createElement("details");
      const title = document.createElement("summary");
      title.textContent = `Development ${cycle.step + 1} · ${cycle.choice.action}`; detail.append(title);
      researchText(detail, "What seeing changed", cycle.choice.what_changed);
      researchText(detail, "Uncertainty and next experiment", cycle.choice.reason);
      researchText(detail, "Selected branches", cycle.choice.selected_image_ids.join(", ") || "None");
      researchText(detail, "New question", cycle.choice.new_question);
      if (cycle.research) researchText(detail, "Source research", `${cycle.research.status} · ${cycle.research.reasoning_calls} calls`);
      if (cycle.observation) readableSection(detail, "Actual image inspection · model interpretation", cycle.observation);
      card.append(detail);
    }
    if (development.branches) researchText(card, "Retained branches", development.branches.map(b => `${b.id}${b.parent_image_id ? ` ← ${b.parent_image_id}` : " (initial or fresh study)"}`).join("\n"));
    if (development.human_review_requested) researchText(card, "Human review", "Requested by the model. No human judgment has been recorded by this process.");
    directions.append(card);
  }
  const researchHolder = byId("run-research");
  const expanded = new Set([...researchHolder.querySelectorAll("details[open]")].map(node => node.querySelector("summary")?.textContent));
  researchHolder.replaceChildren();
  if (!research.length) researchHolder.textContent = "No independent research has been recorded for this run yet. Saved reasoning and files also appear in the update stream.";
  for (const session of research) {
    const holder = document.createElement("article"); holder.className = "research-session";
    holder.dataset.directory = session.directory;
    if (cycleWorkflow) {
      const question = session.frozen_plan?.request?.initial_question;
      holder.append(cycleElement("p", "eyebrow", question ? "REOPENED FROM A SAVED IMAGE" : "INITIAL INQUIRY"));
      if (question) holder.append(cycleElement("blockquote", "cycle-question", question));
    }
    renderResearch(holder, session); researchHolder.append(holder);
  }
  for (const detail of researchHolder.querySelectorAll("details")) if (expanded.has(detail.querySelector("summary")?.textContent)) detail.open = true;
  renderRunFeed(result);
  if (result.errors?.length) byId("detail-status").textContent = result.errors.join(" · ");
  renderRunFiles();
  byId("run-file-count").textContent = inspectedFiles.length;
}
function renderRunFiles() {
  const holder = byId("files"); holder.replaceChildren();
  const query = byId("run-file-search").value.trim().toLowerCase();
  for (const file of inspectedFiles.filter(item => item.name.toLowerCase().includes(query))) {
    const item = document.createElement("li");
    const button = document.createElement("button"); button.textContent = file.name; button.title = `${file.bytes?.toLocaleString() || "?"} bytes`;
    button.addEventListener("click", () => previewFile(file)); item.append(button); holder.append(item);
  }
  if (!holder.children.length) holder.textContent = query ? "No matching files." : "No files saved yet.";
}
async function previewFile(file) {
  activePreviewFile = file;
  const request = ++activeFileRequest;
  byId("file-preview").hidden = false;
  byId("file-preview-name").textContent = file.name;
  byId("file-preview-link").href = scopedURL(file.url);
  const content = byId("file-preview-content"); content.textContent = "Loading…";
  if (file.type === "image") {
    content.replaceChildren(); const image = document.createElement("img"); image.src = scopedURL(file.url); image.alt = file.name; content.append(image); return;
  }
  if (file.bytes > 2 * 1024 * 1024) { content.textContent = "This file is large. Open the full file to inspect it."; return; }
  try {
    const response = await fetch(scopedURL(file.url), {cache:"no-store"});
    if (!response.ok) throw new Error(`Could not read file (HTTP ${response.status}).`);
    const value = await response.text();
    if (request === activeFileRequest) content.textContent = value.slice(0, 160000) + (value.length > 160000 ? "\n… Open full file for the remaining content." : "");
  } catch (error) { if (request === activeFileRequest) content.textContent = error.message; }
}
byId("run-file-search").addEventListener("input", renderRunFiles);
byId("run-update-search").addEventListener("input", () => { runFeedLimit = 40; renderFeedEntries(); });
byId("run-update-type").addEventListener("change", () => { runFeedLimit = 40; renderFeedEntries(); });
byId("run-updates-more").addEventListener("click", () => { runFeedLimit += 40; renderFeedEntries(); });
window.matchMedia?.("(max-width:1100px)").addEventListener("change", event => { byId("run-evidence").open = !event.matches; });
byId("file-preview-refresh").addEventListener("click", () => { if (activePreviewFile) previewFile(activePreviewFile); });
function canStopRun(run) {
  return run?.kind === "artistic_studio_run" && BUSY_RUN_STATES.has(run.status);
}

function configureStopButton(button, run) {
  const phase = stopRequests.get(run.id) || (run.stop_requested || run.data?.stop_requested ? "requested" : null);
  button.dataset.stopRun = run.id;
  button.className = "stop-run-button";
  button.hidden = !canStopRun(run);
  button.disabled = Boolean(phase);
  button.textContent = phase === "sending" ? "Requesting stop…" : phase ? "Stopping…" : "Stop run";
  button.title = "Finish the current action, save its output, and stop before the next action.";
  button.setAttribute("aria-label", `${button.textContent} ${run.title || run.name || run.id}`);
}

function createStopButton(run) {
  const button = document.createElement("button"); button.type = "button";
  configureStopButton(button, run);
  button.addEventListener("click", event => { event.stopPropagation(); return requestRunStop(run); });
  return button;
}

function syncStopButtons(run) {
  for (const button of document.querySelectorAll("[data-stop-run]")) {
    if (button.dataset.stopRun === run.id) configureStopButton(button, run);
  }
}

async function requestRunStop(run) {
  if (!canStopRun(run) || stopRequests.has(run.id) || run.stop_requested || run.data?.stop_requested) return;
  const name = run.title || run.name || run.id;
  stopRequests.set(run.id, "sending"); syncStopButtons(run);
  try {
    const result = await postJSON(`/api/runs/${encodeURIComponent(run.id)}/stop`, {});
    const updated = {...run, status: result.status, data: {...run.data, ...result}};
    if (canStopRun(updated) && result.stop_requested) stopRequests.set(run.id, "requested");
    else stopRequests.delete(run.id);
    byId("stop-status").textContent = canStopRun(updated)
      ? `Stopping “${name}” after the current action. Saved work will be kept.`
      : `“${name}” is ${text(result.status).replaceAll("_", " ")}. Saved work is kept.`;
    syncStopButtons(updated);
    if (selectedRunRecord?.id === run.id) {
      renderDetail(updated);
      byId("run-progress").textContent = byId("stop-status").textContent;
    }
  } catch (error) {
    stopRequests.delete(run.id);
    syncStopButtons(lastRuns.find(item => item.id === run.id) || run);
    byId("stop-status").textContent = `Could not stop “${name}”: ${error.message} You can try again.`;
  } finally { scheduleLiveUpdates(1000, true); }
}
byId("stop-run").addEventListener("click", () => requestRunStop(selectedRunRecord));

initialize();
