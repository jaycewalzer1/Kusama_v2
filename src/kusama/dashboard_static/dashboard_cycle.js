/* Read-only views of frozen workflow settings and saved development evidence. */
function cycleElement(tag, className, content) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (content !== undefined) element.textContent = content;
  return element;
}

function renderCycleMap(holder, workflow, counts) {
  holder.replaceChildren();
  const heading = cycleElement("div", "workflow-heading");
  heading.append(cycleElement("span", "eyebrow", "THE INQUIRY STAYS OPEN"),
    cycleElement("span", "workflow-limit", `Up to ${workflow.policy.max_steps} development actions`));
  const layout = cycleElement("div", "workflow-layout");
  const initial = cycleElement("button", "workflow-initial");
  initial.append(cycleElement("span", "workflow-number", "01 / OPEN POSSIBILITIES"),
    cycleElement("strong", "", "Research → studies → inspect"),
    cycleElement("span", "", `${counts.initial || 0} / ${workflow.initial_image_count} initial images saved`));
  initial.addEventListener("click", () => openRunSection("images"));
  const loop = cycleElement("div", "workflow-loop");
  const choose = cycleElement("div", "workflow-choice");
  choose.append(cycleElement("span", "workflow-number", "02 / CHOOSE THE NEXT EXPERIMENT"),
    cycleElement("strong", "", "What did seeing the work change?"));
  const routes = cycleElement("div", "workflow-routes");
  for (const [label, detail, route, tab] of [
    ["Research again", "Ask a new question from a saved image", "↩ Findings return to the next choice", "research"],
    ["Make or revise", "Develop, revisit or combine saved branches", "↩ Inspect the image, then choose again", "development"],
    ["Review or stop", "Pause for a person, or end the cycle", "All branches remain available", "development"]]) {
    const node = cycleElement("button", "workflow-route");
    node.append(cycleElement("strong", "", label), cycleElement("span", "", detail), cycleElement("small", "", route));
    node.addEventListener("click", () => openRunSection(tab));
    routes.append(node);
  }
  loop.append(choose, routes);
  layout.append(initial, loop);
  holder.append(heading, layout);
  holder.append(cycleElement("p", "workflow-navigation", "Open a route to inspect its saved evidence."));
}

function cycleViewModel(result) {
  const workflow = result.development_cycle;
  if (!workflow) return null;
  const report = workflow.development || {};
  const cycles = report.cycles || [];
  const initialIDs = workflow.initial_image_ids || [];
  const order = initialIDs.concat(cycles.flatMap(row => (row.render?.images || []).map(image => image.id)));
  const position = id => order.includes(id) ? order.indexOf(id) : Number.MAX_SAFE_INTEGER;
  const files = (result.images || []).filter(file => file.record?.id).sort((a, b) => position(a.record.id) - position(b.record.id) || a.name.localeCompare(b.name));
  const observations = new Map((report.observations || []).map(item => [item.image_id, item]));
  for (const row of cycles) if (row.observation && row.render?.images?.[0]) observations.set(row.render.images[0].id, row.observation);
  const labels = new Map(); let nextBranch = 0;
  for (const file of files) {
    const initial = initialIDs.indexOf(file.record.id);
    labels.set(file.record.id, initial >= 0 ? `Initial ${initial + 1}` : `Branch ${++nextBranch}`);
  }
  return {workflow, report, cycles, files, observations, labels,
    initial: files.filter(file => initialIDs.includes(file.record.id)).length,
    actions: cycles.filter(row => !["review", "stop", "abandon"].includes(row.choice.action)).length,
    researched: (report.research || []).length,
    selected: report.selected_image_ids || report.choice?.selected_image_ids || []};
}

function focusCycleBranch(id) {
  openRunSection("development");
  const card = [...byId("development-branches").children].find(node => node.dataset.branchId === id);
  if (card) { card.scrollIntoView({behavior: "smooth", block: "center"}); card.focus({preventScroll: true}); }
}

function cycleBranchLinks(holder, ids, view) {
  const links = cycleElement("div", "branch-links");
  for (const id of ids || []) {
    const button = cycleElement("button", "branch-link", view.labels.get(id) || id);
    button.addEventListener("click", () => focusCycleBranch(id)); links.append(button);
  }
  holder.append(links);
}

function focusCycleResearch(directory) {
  openRunSection("research");
  const session = [...byId("run-research").children].find(node => node.dataset.directory === directory);
  if (session) session.scrollIntoView({behavior: "smooth", block: "start"});
}

function renderCycleWorkspace(result) {
  const view = cycleViewModel(result);
  byId("run-development-section").hidden = !view;
  byId("run-cycle-map").hidden = !view;
  byId("run-cycle-summary").hidden = !view;
  byId("run-stage-list").hidden = Boolean(view);
  if (!view) {
    for (const id of ["run-cycle-map", "run-cycle-summary", "development-status", "development-branches", "development-history"]) byId(id).replaceChildren();
    return false;
  }
  const {workflow, report, files, cycles, observations} = view;
  renderCycleMap(byId("run-cycle-map"), workflow, view);
  const busy = BUSY_RUN_STATES.has(selectedRunRecord?.status);
  const progress = selectedRunRecord?.data?.progress_message || selectedRunRecord?.progress_message;
  let status = busy ? progress || "The inquiry is in progress." : report.reason || "Development has not been reached in this run.";
  if (report.human_review_requested) status = "Paused for human review. The model requested assessment; no human judgment is recorded by this cycle.";
  const summary = byId("run-cycle-summary"); summary.replaceChildren();
  const fixture = workflow.fixture || report.origin === "fixture_actor";
  const fixtureLabel = "Fixture example · scripted responses and substitute images. No artistic inference or human assessment.";
  if (fixture) summary.append(cycleElement("p", "fixture-notice", fixtureLabel));
  const notice = cycleElement("div", report.human_review_requested ? "cycle-notice review-request" : "cycle-notice");
  notice.append(cycleElement("strong", "", report.human_review_requested ? "Human review requested" : busy ? "Current activity" : "Saved outcome"), cycleElement("p", "", status));
  const metrics = cycleElement("div", "cycle-metrics");
  for (const label of [`${view.actions} / ${workflow.policy.max_steps} development actions recorded`,
    `${view.researched} / ${workflow.policy.research_sessions} reopened research sessions`,
    `${observations.size} saved image inspections`, `${files.length} retained branches`]) metrics.append(cycleElement("span", "", label));
  const open = cycleElement("button", "", "Explore branches & decisions →");
  open.addEventListener("click", () => openRunSection("development"));
  summary.append(notice, metrics, open);
  const state = byId("development-status"); state.replaceChildren();
  if (fixture) state.append(cycleElement("p", "fixture-notice", fixtureLabel));
  state.append(cycleElement("p", "cycle-notice", status));
  byId("development-step-count").textContent = cycles.length || "";
  byId("branch-count").textContent = `${files.length} saved · ${observations.size} inspected`;
  renderCycleBranches(view);
  renderCycleHistory(view);
  return true;
}

function renderCycleBranches(view) {
  const holder = byId("development-branches");
  const existing = new Map([...holder.children].map(node => [node.dataset.branchId, node]));
  const wanted = new Set(view.files.map(file => file.record.id));
  for (const node of [...holder.children]) if (!wanted.has(node.dataset.branchId)) node.remove();
  for (const file of view.files) {
    const record = file.record, id = record.id;
    const card = existing.get(id) || cycleElement("article", "branch-card");
    card.dataset.branchId = id; card.tabIndex = -1;
    card.className = "branch-card" + (view.selected.includes(id) ? " model-selected" : "");
    const signature = JSON.stringify([file, view.selected, view.observations.get(id)]);
    if (card.dataset.signature !== signature) {
      const priorImage = card.querySelector("a"), inspectionOpen = card.querySelector("details")?.open;
      card.replaceChildren();
      const image = priorImage || cycleElement("a", "branch-image");
      image.href = scopedURL(file.url); image.target = "_blank"; image.rel = "noopener";
      if (!priorImage) { const picture = cycleElement("img", ""); picture.src = scopedURL(file.url); picture.alt = file.label || view.labels.get(id); picture.loading = "lazy"; image.append(picture); }
      const body = cycleElement("div", "branch-body");
      body.append(cycleElement("h4", "", view.labels.get(id)), cycleElement("p", "branch-title", file.label || id));
      if (view.selected.includes(id)) body.append(cycleElement("span", "selection-label", "Model selection"));
      const parent = record.parent_image_id;
      body.append(cycleElement("p", "branch-meta", parent ? "Revises" : view.workflow.initial_image_ids.includes(id) ? "Initial study image" : "Fresh reformulation"));
      if (parent) cycleBranchLinks(body, [parent], view);
      const choice = view.cycles.find(row => row.render?.images?.some(im => im.id === id))?.choice;
      if (choice?.selected_image_ids?.length > 1) {
        body.append(cycleElement("p", "branch-meta", "Also draws on")); cycleBranchLinks(body, choice.selected_image_ids.slice(1), view);
      }
      const observation = view.observations.get(id);
      body.append(cycleElement("span", "inspection-label", observation ? "Image inspected · model observation" : "No saved inspection yet"));
      if (observation) {
        body.append(cycleElement("p", "branch-observation", observation.possible_next_question || observation.uncertainty));
        const detail = cycleElement("details", "cycle-evidence"); detail.open = Boolean(inspectionOpen);
        detail.append(cycleElement("summary", "", "Full image observation"));
        for (const [key, label] of [["visible_relationships", "Visible relationships"], ["unexpected_effects", "Unexpected effects"], ["explanation_gaps", "Gaps in the explanation"], ["uncertainty", "Uncertainty"]]) {
          researchText(detail, label, Array.isArray(observation[key]) ? observation[key].join("\n") || "None reported" : observation[key]);
        }
        body.append(detail);
      }
      card.append(image, body); card.dataset.signature = signature;
    }
    if (!existing.has(id)) holder.append(card);
  }
  if (!view.files.length) holder.append(cycleElement("p", "empty", "Saved studies and revisions will appear here with their parent links. No branch is discarded when another is developed."));
}

function renderCycleHistory(view) {
  const holder = byId("development-history");
  const expanded = new Map([...holder.children].map(node => [node.dataset.step, node.querySelector("details")?.open]));
  holder.replaceChildren();
  if (!view.cycles.length) {
    holder.append(cycleElement("p", "empty", "The first development choice follows at least three realized studies and inspection of every planned initial image. Initial research is available in Research."));
    return;
  }
  const names = {research: "Reopen research", continue: "Revise a saved branch", combine: "Combine branches", reformulate: "Reformulate", review: "Pause for human review", stop: "Stop", abandon: "Abandon this direction"};
  for (const row of [...view.cycles].reverse()) {
    const choice = row.choice;
    const card = cycleElement("article", "cycle-step"); card.dataset.step = String(row.step);
    const heading = cycleElement("div", "cycle-step-heading");
    heading.append(cycleElement("span", "cycle-index", String(row.step + 1).padStart(2, "0")), cycleElement("h4", "", names[choice.action] || choice.action), cycleElement("span", "cycle-attribution", view.report.origin === "fixture_actor" ? "Fixture actor" : "Qwen decision"));
    card.append(heading, cycleElement("p", "cycle-change", choice.what_changed));
    if (choice.new_question) card.append(cycleElement("blockquote", "cycle-question", choice.new_question));
    cycleBranchLinks(card, choice.selected_image_ids, view);
    if (row.research) {
      card.append(cycleElement("p", "branch-meta", `Research returned: ${text(row.research.status).replaceAll("_", " ")} · ${row.research.reasoning_calls} calls`));
      const open = cycleElement("button", "", "Inspect sources & findings →");
      open.addEventListener("click", () => focusCycleResearch(`${view.workflow.development_directory}/research/cycle-${String(row.step).padStart(2, "0")}`)); card.append(open);
    }
    if (row.render?.images?.length) {
      const outcomes = cycleElement("div", "cycle-outcomes");
      for (const image of row.render.images) {
        const file = view.files.find(f => f.record.id === image.id);
        if (file) {
          const button = cycleElement("button", "cycle-output");
          const thumb = cycleElement("img", ""); thumb.src = scopedURL(file.url); thumb.alt = view.labels.get(image.id);
          button.append(thumb, cycleElement("span", "", `${view.labels.get(image.id)} · ${row.observation ? "inspected" : "inspection not saved"}`));
          button.addEventListener("click", () => focusCycleBranch(image.id)); outcomes.append(button);
        }
      }
      card.append(outcomes);
    }
    const detail = cycleElement("details", "cycle-evidence");
    detail.open = expanded.has(String(row.step)) ? Boolean(expanded.get(String(row.step))) : row === view.cycles.at(-1);
    detail.append(cycleElement("summary", "", "Reason, commitments & image observations"));
    researchText(detail, "Reason for this action", choice.reason);
    researchText(detail, "Commitments", choice.commitments);
    if (row.observation) {
      researchText(detail, "Visible relationships", row.observation.visible_relationships.join("\n"));
      researchText(detail, "Unexpected effects", row.observation.unexpected_effects.join("\n") || "None reported");
      researchText(detail, "Gaps in the explanation", row.observation.explanation_gaps.join("\n") || "None reported");
      researchText(detail, "Uncertainty", row.observation.uncertainty);
      researchText(detail, "Next question", row.observation.possible_next_question);
    }
    if (choice.action === "review") researchText(detail, "Human review", "Requested only. This decision does not record a human assessment.");
    if (row.render?.status === "partial") researchText(detail, "Rendering", "Incomplete or failed. Saved images and failures remain available.");
    card.append(detail); holder.append(card);
  }
}
