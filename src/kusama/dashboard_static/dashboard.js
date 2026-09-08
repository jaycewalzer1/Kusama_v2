"use strict";

const byId = (id) => document.getElementById(id);
let selectedId = null;
let detailRequest = 0;
let lastRuns = [];
let liveTimer = null;
let liveRequest = 0;
let lastDetailJSON = null;
let reviewCollectionRefreshPending = false;
const LIVE_INTERVAL_MS = 15000;
const BUSY_RUN_STATES = new Set(["pending", "queued", "proposing", "running", "directing", "rendering", "observing", "loading", "generating", "editing", "preparing", "recovering", "starting", "executing"]);

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
  const response = await fetch(url, { cache: "no-store" });
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
  byId("runs").replaceChildren();
  for (const run of runs) {
    const row = document.createElement("tr");
    row.dataset.id = run.id;
    const nameCell = document.createElement("td");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = run.title || run.name || run.id;
    button.title = run.name || run.id;
    button.addEventListener("click", () => selectRun(run.id, run.name));
    nameCell.append(button);
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
  link.href = artifact.url;
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = artifact.name;
  return link;
}

function renderDetail(run) {
  // Keep loaded images and open disclosures in place during progress updates.
  const previousRawScroll = byId("raw").scrollTop;
  const existingImages = new Map([...byId("images").children].map((node) => [node.dataset.url, node]));
  const existingFiles = new Map([...byId("files").children].map((node) => [node.dataset.url, node]));
  const seenImages = new Set();
  const seenFiles = new Set();
  byId("run-facts").replaceChildren();
  byId("run-name").textContent = run.title || run.name || run.id;
  byId("run-brief").textContent = run.brief || "";
  byId("run-brief").hidden = !run.brief;
  byId("detail-status").textContent = "";
  addFact("Kind", kindName(run.kind));
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
    report.href = `/api/runs/${encodeURIComponent(run.id)}/files/verification.json`;
    report.target = "_blank";
    report.rel = "noopener";
    report.textContent = "saved verification report";
    fact.append(report);
  } else {
    addFact("Verification", verification(run.verification));
  }
  byId("run-error").textContent = run.error ? text(run.error) : "";
  byId("run-error").hidden = !run.error;
  const artifacts = [...(run.artifacts || [])].sort((a, b) => Number(b.name === "parent.png") - Number(a.name === "parent.png"));
  const hasOutputs = artifacts.some((artifact) => artifact.name === "parent.png");
  for (const artifact of artifacts) {
    const preview = artifact.preview || (hasOutputs
      ? /^(parent|attempt_\d+_result|edit_seed_\d+)\.png$/.test(artifact.name)
      : artifact.name === "comparison.png");
    if (artifact.type === "image" && preview) {
      seenImages.add(artifact.url);
      if (existingImages.has(artifact.url)) continue;
      const figure = document.createElement("figure");
      const caption = document.createElement("figcaption");
      figure.dataset.url = artifact.url;
      const labeledArtifact = { ...artifact, name: artifact.label || artifact.name };
      caption.append(artifactLink(labeledArtifact));
      const imageLink = artifactLink(artifact);
      imageLink.textContent = "";
      const image = document.createElement("img");
      image.src = artifact.url;
      image.alt = artifact.label || artifact.name;
      image.loading = "lazy";
      const record = run.data?.images?.find((candidate) => candidate.path === artifact.name);
      if (Array.isArray(record?.size) && record.size.length === 2) {
        image.width = record.size[0];
        image.height = record.size[1];
      }
      imageLink.append(image);
      figure.append(caption, imageLink);
      byId("images").append(figure);
    } else {
      seenFiles.add(artifact.url);
      if (existingFiles.has(artifact.url)) continue;
      const item = document.createElement("li");
      item.dataset.url = artifact.url;
      item.append(artifactLink(artifact));
      byId("files").append(item);
    }
  }
  for (const [url, node] of existingImages) if (!seenImages.has(url)) node.remove();
  for (const [url, node] of existingFiles) if (!seenFiles.has(url)) node.remove();
  byId("raw").textContent = JSON.stringify(run.data ?? run, null, 2);
  byId("raw").scrollTop = previousRawScroll;
  lastDetailJSON = JSON.stringify(run);
  byId("raw-details").hidden = false;
}

async function selectRun(id, name) {
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
  byId("run-error").hidden = true;
  byId("raw-details").hidden = true;
  byId("raw-details").open = false;
  try {
    const run = await getJSON(`/api/runs/${encodeURIComponent(id)}`);
    if (request === detailRequest) renderDetail(run);
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
    const projects = await getJSON("/api/projects");
    renderProjects(projects.projects);
    await refreshExplorer();
    await refreshReviews();
    const result = await getJSON("/api/runs");
    const runs = result.runs;
    if (!Array.isArray(runs)) throw new Error("The server returned an invalid run list.");
    lastRuns = runs;
    renderRuns(runs);
    updateRunStatus();
    const selected = runs.find((run) => run.id === selectedId) || runs[0];
    if (selected) {
      await selectRun(selected.id, selected.name);
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
  byId("list-status").textContent = lastRuns.length
    ? `${lastRuns.length} run${lastRuns.length === 1 ? "" : "s"}${busy ? " · live updates every 15 s" : ""}`
    : "No saved runs found.";
}

function scheduleLiveUpdates(delay = LIVE_INTERVAL_MS, retry = false) {
  clearTimeout(liveTimer);
  if (retry || reviewCollectionRefreshPending || lastRuns.some((run) => BUSY_RUN_STATES.has(run.status))) liveTimer = setTimeout(pollRunProgress, delay);
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
    reviewCollectionRefreshPending ||= settled;
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
    }
    // Only update the collection selector when an experiment settles. This never
    // calls openComparison or touches radios, notes, reviewer identity or drafts.
    if (reviewCollectionRefreshPending && request === liveRequest) {
      await refreshReviews();
      await refreshExplorer();
      reviewCollectionRefreshPending = false;
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
        link.href = `/api/projects/${encodeURIComponent(project.id)}/artifacts/${encodeURIComponent(id)}`;
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
        if (!reviewerName()) {
          status.textContent = "Enter your reviewer name in Human review first.";
          byId("reviewer").focus();
          return;
        }
        button.disabled = true;
        try {
          await postJSON(`/api/projects/${encodeURIComponent(project.id)}/select`, { episode_id: episode.id, artifact_id: select.value, author: reviewerName(), reason: reason.value });
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
refresh();

let reviewCollection = null;
let activeComparison = null;
let reviewRequest = 0;
let voteSubmissionId = null;
let voteSubmissionBody = null;
let advanceSubmissionId = null;
let comparisonImagesReady = false;
let comparisonStartedAt = null;
let activePool = null;
let poolSubmissionId = null;
let poolSubmissionBody = null;
let poolImagesReady = false;
const exposureByCollection = new Map();

function currentExposure() {
  const key = `${reviewerName()}\n${reviewCollection?.id || ""}`;
  if (!exposureByCollection.has(key)) exposureByCollection.set(key, { context_revealed: false, overview_opened: false });
  return exposureByCollection.get(key);
}

const reviewBase = (id) => `/api/reviews/collections/${encodeURIComponent(id)}`;

async function postJSON(url, body) {
  const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json", "X-Kusama-Review": "1" }, body: JSON.stringify(body) });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : `Could not save (HTTP ${response.status}).`);
  return result;
}

function reviewerName() {
  return byId("reviewer").value.trim();
}

function newSubmissionId() {
  return crypto.randomUUID();
}

async function refreshReviews() {
  const reviewer = reviewerName();
  const result = await getJSON(`/api/reviews/collections?reviewer=${encodeURIComponent(reviewer)}`);
  const select = byId("review-collection");
  const previous = select.value;
  select.replaceChildren();
  for (const collection of result.collections) {
    const option = document.createElement("option");
    option.value = collection.id;
    option.textContent = `${collection.title} · ${collection.candidate_count} images`;
    select.append(option);
  }
  if (!result.collections.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No comparisons registered yet";
    select.append(option);
  } else if (result.collections.some((collection) => collection.id === previous)) {
    select.value = previous;
  }
  const selected = result.collections.find((collection) => collection.id === select.value);
  byId("review-progress").textContent = selected ? `${selected.reviewed_pair_ids.length} of ${selected.comparison_count} comparisons reviewed${reviewer ? ` by ${reviewer}` : ""}` : "";
}

function reviewHistory(comparison) {
  const holder = byId("review-history");
  holder.replaceChildren();
  if (!comparison.reviews.length) holder.textContent = "You have not reviewed this pair yet.";
  for (const review of comparison.reviews) {
    const paragraph = document.createElement("p");
    paragraph.textContent = `${new Date(review.created_at).toLocaleString()}: finished work ${review.finished_quality.choice}; development potential ${review.development_potential.choice}${review.notes ? ` — ${review.notes}` : ""}`;
    holder.append(paragraph);
  }
}

async function openReviewCollection() {
  const collectionId = byId("review-collection").value;
  if (!collectionId) return;
  if (!reviewerName()) {
    byId("review-status").textContent = "Enter a reviewer name so your judgments are attributed to you.";
    byId("reviewer").focus();
    return;
  }
  const request = ++reviewRequest;
  byId("review-status").textContent = "Loading comparisons…";
  byId("review-workspace").hidden = true;
  activeComparison = null;
  try {
    try { localStorage.setItem("kusama-reviewer", reviewerName()); } catch (_) { /* Review still works without browser storage. */ }
    const collection = await getJSON(reviewBase(collectionId));
    if (request !== reviewRequest) return;
    reviewCollection = collection;
    activePool = null;
    byId("pool-details").open = false;
    byId("pool-form").hidden = true;
    byId("pool-images").replaceChildren();
    byId("pool-status").textContent = "";
    byId("review-brief").textContent = collection.brief;
    byId("collection-images").replaceChildren();
    byId("collection-overview").open = false;
    for (const candidate of collection.candidates) {
      const figure = document.createElement("figure");
      const caption = document.createElement("figcaption");
      caption.textContent = candidate.label;
      const link = document.createElement("a");
      link.href = `${reviewBase(collectionId)}/candidates/${encodeURIComponent(candidate.id)}`;
      link.target = "_blank";
      link.rel = "noopener";
      const image = document.createElement("img");
      image.src = link.href;
      image.alt = candidate.label;
      image.loading = "lazy";
      link.append(image);
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "Choose for development";
      button.addEventListener("click", () => {
        byId("advance-candidate").value = candidate.id;
        advanceSubmissionId = null;
        byId("advance-details").open = true;
        byId("advance-status").textContent = "Add your reason and record the choice below.";
        byId("advance-details").scrollIntoView({ behavior: "smooth", block: "center" });
      });
      figure.append(caption, link, button);
      byId("collection-images").append(figure);
    }
    byId("review-pair").replaceChildren();
    for (const [index, pair] of collection.pairs.entries()) {
      const option = document.createElement("option");
      option.value = pair.id;
      option.textContent = `${index + 1} of ${collection.pairs.length} · ${pair.id}`;
      byId("review-pair").append(option);
    }
    const history = await getJSON(`${reviewBase(collectionId)}/history`);
    if (request !== reviewRequest) return;
    const reviewed = new Set(history.votes.filter((vote) => vote.reviewer === reviewerName()).map((vote) => vote.pair_id));
    const next = collection.pairs.find((pair) => !reviewed.has(pair.id));
    if (next) byId("review-pair").value = next.id;
    await openComparison();
  } catch (error) {
    if (request === reviewRequest) byId("review-status").textContent = error.message;
  }
}

async function openComparison() {
  if (!reviewCollection || !reviewerName()) return;
  const request = ++reviewRequest;
  activeComparison = null;
  byId("review-workspace").hidden = true;
  byId("review-status").textContent = "Checking images and loading your comparison…";
  try {
    const comparison = await getJSON(`${reviewBase(reviewCollection.id)}/pairs/${encodeURIComponent(byId("review-pair").value)}?reviewer=${encodeURIComponent(reviewerName())}`);
    if (request !== reviewRequest) return;
    activeComparison = comparison;
    voteSubmissionId = null;
    voteSubmissionBody = null;
    advanceSubmissionId = null;
    byId("vote-form").reset();
    byId("advance-form").reset();
    comparisonImagesReady = false;
    comparisonStartedAt = null;
    byId("review-provenance-details").open = false;
    let loadedImages = 0;
    byId("save-review").disabled = true;
    byId("vote-status").textContent = "";
    byId("advance-status").textContent = "";
    byId("advance-details").open = false;
    byId("review-images").replaceChildren();
    byId("advance-candidate").replaceChildren();
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "Choose an image";
    byId("advance-candidate").append(blank);
    for (const [position, candidate] of Object.entries(comparison.candidates)) {
      const url = `${reviewBase(reviewCollection.id)}/pairs/${encodeURIComponent(comparison.pair_id)}/images/${position}?reviewer=${encodeURIComponent(reviewerName())}`;
      const figure = document.createElement("figure");
      const caption = document.createElement("figcaption");
      caption.textContent = position;
      const full = document.createElement("a");
      full.href = url;
      full.target = "_blank";
      full.rel = "noopener";
      full.textContent = "Open full size";
      caption.append(full);
      const imageLink = full.cloneNode(false);
      imageLink.textContent = "";
      const image = document.createElement("img");
      image.src = url;
      image.alt = `Candidate ${position}`;
      image.addEventListener("load", () => {
        if (activeComparison?.assignment_id === comparison.assignment_id && ++loadedImages === 2) {
          comparisonImagesReady = true;
          comparisonStartedAt = Date.now();
          byId("save-review").disabled = false;
        }
      });
      image.addEventListener("error", () => {
        if (activeComparison?.assignment_id === comparison.assignment_id) {
          comparisonImagesReady = false;
          byId("save-review").disabled = true;
          byId("vote-status").textContent = "An image could not be loaded. Reload the comparison before judging.";
        }
      });
      imageLink.append(image);
      figure.append(caption, imageLink);
      byId("review-images").append(figure);
      const option = document.createElement("option");
      option.value = candidate.id;
      option.textContent = `Image ${position}`;
      byId("advance-candidate").append(option);
    }
    const pairedIds = new Set(Object.values(comparison.candidates).map((candidate) => candidate.id));
    for (const candidate of reviewCollection.candidates) {
      if (pairedIds.has(candidate.id)) continue;
      const option = document.createElement("option");
      option.value = candidate.id;
      option.textContent = candidate.label;
      byId("advance-candidate").append(option);
    }
    byId("comparison-kind").textContent = comparison.kind === "before_after" ? "Before / after comparison. Image positions are randomized; provenance is available below." : "Alternative realizations. Image positions stay fixed for you across reloads.";
    byId("review-provenance").textContent = JSON.stringify(comparison.candidates, null, 2);
    const presentation = comparison.presentation;
    byId("review-brief").textContent = presentation?.condition === "statement_free" ? "Statement-free visual comparison. Context is available separately." : presentation?.supplied_context || reviewCollection.brief;
    reviewHistory(comparison);
    const index = reviewCollection.pairs.findIndex((pair) => pair.id === comparison.pair_id);
    byId("previous-pair").disabled = index <= 0;
    byId("next-pair").disabled = index === reviewCollection.pairs.length - 1;
    byId("review-status").textContent = comparison.reviews.length ? "Your earlier judgments are preserved below. Saving again adds a new review." : "Inspect both images, then answer each question independently.";
    byId("review-workspace").hidden = false;
    await refreshReviews();
  } catch (error) {
    if (request === reviewRequest) byId("review-status").textContent = error.message;
  }
}

byId("vote-form").addEventListener("input", () => { voteSubmissionId = null; voteSubmissionBody = null; byId("save-review").disabled = !comparisonImagesReady; });
byId("advance-form").addEventListener("input", () => { advanceSubmissionId = null; });
byId("vote-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!activeComparison || activeComparison.reviewer !== reviewerName()) return;
  if (!comparisonImagesReady) return;
  const snapshot = activeComparison;
  const form = new FormData(event.currentTarget);
  const answer = (dimension) => ({ choice: form.get(dimension), confidence: form.get(`${dimension}_confidence`) || null });
  const body = voteSubmissionBody ||= { reviewer: reviewerName(), assignment_id: snapshot.assignment_id,
    submission_id: voteSubmissionId ||= newSubmissionId(), finished_quality: answer("finished_quality"),
    development_potential: answer("development_potential"), notes: form.get("notes"),
    open_reading: form.get("open_reading") || null, felt_response: form.get("felt_response") || null,
    readiness: { A: form.get("readiness_A") || null, B: form.get("readiness_B") || null },
    development_willingness: { A: form.get("development_willingness_A") || null, B: form.get("development_willingness_B") || null },
    exposure: { ...currentExposure(), prior_exposure: form.get("prior_exposure"),
      context_revealed: currentExposure().context_revealed || snapshot.presentation?.condition === "supplied_context",
      duration_seconds: comparisonStartedAt === null ? null : Math.min(86400, (Date.now() - comparisonStartedAt) / 1000),
      duration_source: comparisonStartedAt === null ? "unrecorded" : "browser_elapsed" } };
  byId("save-review").disabled = true;
  byId("vote-status").textContent = "Saving…";
  try {
    const review = await postJSON(`${reviewBase(snapshot.collection_id)}/pairs/${encodeURIComponent(snapshot.pair_id)}/votes`, body);
    if (activeComparison?.assignment_id !== snapshot.assignment_id) return;
    if (!activeComparison.reviews.some((existing) => existing.submission_id === review.submission_id)) activeComparison.reviews.push(review);
    reviewHistory(activeComparison);
    byId("vote-status").textContent = "Both judgments saved. Use Next to continue.";
    await refreshReviews();
  } catch (error) {
    if (activeComparison?.assignment_id === snapshot.assignment_id) {
      byId("vote-status").textContent = error.message;
      byId("save-review").disabled = false;
    }
  }
});

byId("advance-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!activeComparison || activeComparison.reviewer !== reviewerName()) return;
  const collectionId = activeComparison.collection_id;
  const form = new FormData(event.currentTarget);
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  byId("advance-status").textContent = "Saving your development choice…";
  try {
    await postJSON(`${reviewBase(collectionId)}/advance`, { reviewer: reviewerName(), candidate_id: form.get("candidate_id"), reason: form.get("reason"), submission_id: advanceSubmissionId ||= newSubmissionId() });
    byId("advance-status").textContent = "Development choice recorded. Generation has not started.";
  } catch (error) {
    byId("advance-status").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

byId("start-review").addEventListener("click", openReviewCollection);
byId("review-collection").addEventListener("change", openReviewCollection);
byId("review-pair").addEventListener("change", openComparison);
byId("reviewer").addEventListener("input", () => {
  ++reviewRequest;
  activeComparison = null;
  byId("review-workspace").hidden = true;
  byId("review-status").textContent = "Open a comparison to review under this name.";
});
for (const [id, change] of [["previous-pair", -1], ["next-pair", 1]]) {
  byId(id).addEventListener("click", () => {
    const select = byId("review-pair");
    select.selectedIndex = Math.min(select.options.length - 1, Math.max(0, select.selectedIndex + change));
    openComparison();
  });
}
try { byId("reviewer").value = localStorage.getItem("kusama-reviewer") || ""; } catch (_) { /* Optional convenience only. */ }
refreshReviews().catch((error) => { byId("review-status").textContent = error.message; });

byId("collection-overview").addEventListener("toggle", () => {
  if (byId("collection-overview").open && reviewCollection) currentExposure().overview_opened = true;
});
byId("review-provenance-details").addEventListener("toggle", () => {
  if (byId("review-provenance-details").open && reviewCollection) currentExposure().context_revealed = true;
});

function externalLink(url, label) {
  const link = document.createElement("a");
  try {
    const parsed = new URL(url, location.origin);
    if (!["http:", "https:"].includes(parsed.protocol)) return document.createTextNode(label);
    link.href = parsed.href;
  } catch (_) { return document.createTextNode(label); }
  link.textContent = label; link.target = "_blank"; link.rel = "noopener noreferrer";
  return link;
}

function imageCard(url, caption, alt) {
  const figure = document.createElement("figure");
  const link = externalLink(url, "");
  const image = document.createElement("img"); image.src = url; image.alt = alt; image.loading = "lazy";
  link.append(image); const label = document.createElement("figcaption"); label.textContent = caption;
  figure.append(link, label); return figure;
}

async function showSource(id) {
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
  try {
    const [continuations, inspirations, sources] = await Promise.all([getJSON("/api/continuations"), getJSON("/api/inspirations"), getJSON("/api/sources")]);
    const gallery = byId("continuation-gallery"); gallery.replaceChildren();
    if (!continuations.continuations.length) gallery.textContent = "No selected-parent continuations registered yet.";
    for (const item of continuations.continuations) {
      const row = document.createElement("p");
      row.append(externalLink(`/api/continuations/${encodeURIComponent(item.id)}/gallery`, item.request.title), ` · ${item.status} · ${item.model_calls}/2 calls · ${item.reviews.votes} human reviews · selection by ${item.selection.author} (${item.selection.origin})`);
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

byId("load-pool").addEventListener("click", async () => {
  if (!reviewCollection || !reviewerName()) return;
  const collectionId = reviewCollection.id;
  const reviewer = reviewerName();
  byId("pool-status").textContent = "Loading registered pool…";
  try {
    const assignment = await getJSON(`${reviewBase(collectionId)}/pool?reviewer=${encodeURIComponent(reviewer)}`);
    if (reviewCollection?.id !== collectionId || reviewerName() !== reviewer) return;
    activePool = assignment; poolSubmissionId = null; poolSubmissionBody = null; poolImagesReady = false;
    byId("pool-form").reset(); byId("pool-images").replaceChildren(); byId("pool-choice").replaceChildren();
    const submit = byId("pool-form").querySelector('button[type="submit"]'); submit.disabled = true;
    let loaded = 0;
    const blank = document.createElement("option"); blank.value = ""; blank.textContent = "Choose an image or response"; byId("pool-choice").append(blank);
    for (const [position, candidate] of Object.entries(assignment.candidates)) {
      const card = imageCard(`${reviewBase(collectionId)}/pool/images/${position}?reviewer=${encodeURIComponent(reviewer)}`, `Image ${position}`, `Pool image ${position}`);
      card.querySelector("img").addEventListener("load", () => {
        if (activePool?.assignment_id === assignment.assignment_id && ++loaded === Object.keys(assignment.candidates).length) {
          poolImagesReady = true; submit.disabled = !assignment.complete;
        }
      });
      card.querySelector("img").addEventListener("error", () => {
        if (activePool?.assignment_id === assignment.assignment_id) {
          poolImagesReady = false; submit.disabled = true;
          byId("pool-status").textContent = "A pool image failed to load. Reload the pool before judging.";
        }
      });
      byId("pool-images").append(card);
      const option = document.createElement("option"); option.value = `candidate:${candidate.id}`; option.textContent = `Image ${position}`; byId("pool-choice").append(option);
    }
    for (const [value, label] of [["none", "None"], ["unsure", "Unsure"], ["decline", "Decline to judge"]]) {
      const option = document.createElement("option"); option.value = value; option.textContent = label; byId("pool-choice").append(option);
    }
    byId("pool-form").hidden = !assignment.complete;
    byId("pool-status").textContent = assignment.complete ? "Inspect the entire pool before selecting." : `Incomplete pool: ${Object.keys(assignment.candidates).length}/${assignment.expected_count} images. Excluded from standard pool selections.`;
    currentExposure().overview_opened = true;
  } catch (error) { byId("pool-status").textContent = error.message; }
});
byId("pool-form").addEventListener("input", () => { poolSubmissionId = null; poolSubmissionBody = null; });
byId("pool-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!activePool || !activePool.complete || !poolImagesReady || activePool.reviewer !== reviewerName()) return;
  const form = new FormData(event.currentTarget); const selected = form.get("choice");
  const body = poolSubmissionBody ||= { reviewer: activePool.reviewer, assignment_id: activePool.assignment_id,
    submission_id: poolSubmissionId ||= newSubmissionId(), choice: selected.startsWith("candidate:") ? "candidate" : selected,
    candidate_id: selected.startsWith("candidate:") ? selected.slice(10) : null,
    criterion: form.get("criterion"), notes: form.get("notes"),
    exposure: { ...currentExposure(), prior_exposure: "unknown" } };
  try {
    await postJSON(`${reviewBase(activePool.collection_id)}/pool/votes`, body);
    byId("pool-status").textContent = "Pool selection saved separately from pairwise judgments and advancement.";
  } catch (error) { byId("pool-status").textContent = error.message; }
});
