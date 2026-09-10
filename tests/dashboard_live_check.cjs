/* Exercise the actual polling code with a minimal DOM and controlled HTTP replies.
   No browser, models, user artifacts or development submissions are involved. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') {
    this.tag = tag; this.children = []; this.dataset = {}; this.attributes = {};
    this.value = ''; this.hidden = false; this.disabled = false; this.checked = false;
    this.scrollTop = 0; this.classList = { toggle() {} }; this.listeners = {};
  }
  append(...nodes) { for (const node of nodes) { if (typeof node === 'object') node.parent = this; this.children.push(node); } }
  insertBefore(node, reference) {
    if (node.parent) node.remove();
    node.parent = this;
    const index = this.children.indexOf(reference);
    this.children.splice(index < 0 ? this.children.length : index, 0, node);
  }
  replaceChildren(...nodes) { this.children = []; this.textContent = ''; this.append(...nodes); }
  remove() { this.parent.children = this.parent.children.filter((child) => child !== this); }
  addEventListener(name, action) { this.listeners[name] = action; }
  scrollIntoView() { this.scrolled = true; }
  focus() { this.focused = true; }
  setAttribute(key, value) { this.attributes[key] = value; }
  setCustomValidity(value) { this.validationMessage = value; }
  querySelector(tag) {
    for (const child of this.children) {
      if (typeof child !== 'object') continue;
      if (child.tag === tag) return child;
      const nested = child.querySelector(tag); if (nested) return nested;
    }
    return null;
  }
  querySelectorAll() { return []; }
  closest() { return nodes.get('table-scroll'); }
  reset() { throw new Error('Progress polling must never reset a form'); }
}
const nodes = new Map();
const get = (id) => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
get('table-scroll').scrollTop = 41;
const window = { addEventListener() {}, scrollX: 0, scrollY: 823, scrollTo(x, y) { this.scrollX = x; this.scrollY = y; } };
let reply;
let requestLog = [];
let timerId = 0;
const timers = new Map();
const context = vm.createContext({
  URL, location: { origin: "http://localhost", pathname: "/", hash: "#runs" }, history: { replaceState() {} },
  document: { querySelectorAll() { return []; }, getElementById: get, createElement: (tag) => new Element(tag) }, window,
  localStorage: { getItem() { return null; } }, console,
  setTimeout(callback, delay) { const id = ++timerId; timers.set(id, { callback, delay }); return id; },
  clearTimeout(id) { timers.delete(id); },
  fetch: async (url, options) => {
    assert.ok(!options?.method || options.method === 'GET', 'Polling must be read-only');
    requestLog.push(url);
    return { ok: true, json: async () => reply(url) };
  },
});
let source = fs.readFileSync('src/kusama/dashboard_static/dashboard.js', 'utf8');
vm.runInContext(fs.readFileSync('src/kusama/dashboard_static/dashboard_cycle.js', 'utf8'), context);
vm.runInContext(fs.readFileSync('src/kusama/dashboard_static/dashboard_feed.js', 'utf8'), context);
// Explicit UI startup is separate from testing the polling transition itself.
source = source.replace(/^initialize\(\);$/m, '');
vm.runInContext(source, context);
window.scrollY = 823;
const run = (imageCount, status = 'rendering') => ({
  id: 'run', name: 'run', kind: 'reference_creative_experiment', title: 'Study', brief: 'Brief', status,
  model_calls: imageCount + 12, max_calls: 48, image_count: imageCount, planned_images: 24,
  runtime_seconds: imageCount * 165, runtime_scope: 'recorded_operations',
  artifacts: Array.from({ length: imageCount }, (_, i) => ({ name: `d01-r${i + 1}.png`, type: 'image', preview: true, url: `/image/${i}`, label: `Image ${i}` })),
  data: { images: Array.from({ length: imageCount }, (_, i) => ({ path: `d01-r${i + 1}.png`, size: [512, 640] })) },
});
const setReply = (detail) => { reply = (url) => url === '/api/runs' ? { runs: [detail] } : url === '/api/runs/run/inspect' ? {images: detail.artifacts, files: detail.artifacts} : url === '/api/continuations' ? { continuations: [] } : url === '/api/sources' ? { sources: [] } : url === '/api/inspirations' ? { inspirations: [] } : url === '/api/research' ? { research: [] } : detail; };

(async () => {
  for (const [id, value] of Object.entries({'new-run-steps': 24, 'new-run-directions': 3,
      'new-run-images': 1, 'new-run-development-steps': 3, 'new-run-development-research': 8,
      'new-run-minutes': 120})) get(id).value = String(value);
  reply = () => ({ profiles: [], default_profile: 'fixture', total_token_limit: 16384, reserved_output_tokens: 1800 });
  await vm.runInContext('loadStartOptions()', context);
  assert.equal(get('start-run').disabled, false);
  assert.match(get('run-authorization').textContent, /688,128 Qwen tokens/);
  assert.match(get('run-authorization').textContent, /16,384 tokens: up to 14,584 input and 1,800 output/);
  assert.match(get('run-authorization').textContent, /up to 6 images/);
  assert.match(get('run-authorization').textContent, /64 requests and 16 acquisitions/);
  assert.equal(get('new-run-development-steps').validationMessage, '');
  get('new-run-directions').value = '6'; get('new-run-images').value = '2';
  vm.runInContext('updateStartSummary()', context);
  assert.match(get('new-run-development-steps').validationMessage, /at most 12/);
  get('new-run-directions').value = '3'; get('new-run-images').value = '1';
  context.initial = run(0);
  vm.runInContext("lastRuns = [initial]; selectedId = 'run';", context);
  get('operator').value = 'Operator';
  get('prepare-json').value = 'Unsaved continuation';
  get('decision-json').value = 'Unsaved decision';
  const drafts = ['operator', 'prepare-json', 'decision-json'].map((id) => [id, get(id), get(id).value]);

  setReply(run(1));
  await vm.runInContext('pollRunProgress()', context);
  assert.equal(get('images').children.length, 1);
  const firstImage = get('images').children[0];
  assert.equal(timers.size, 1);
  assert.equal([...timers.values()][0].delay, 15000);

  setReply(run(2));
  await vm.runInContext('pollRunProgress()', context);
  assert.equal(get('images').children.length, 2);
  assert.equal(get('images').children[0], firstImage, 'Existing images must retain their DOM identity');
  assert.equal(window.scrollY, 823);
  assert.equal(get('table-scroll').scrollTop, 41);
  assert.equal(vm.runInContext('selectedId', context), 'run');
  for (const [id, node, value] of drafts) { assert.equal(get(id), node); assert.equal(get(id).value, value); }

  const partialObservationRun = run(24, 'needs_human_review');
  partialObservationRun.planned_observations = 12;
  partialObservationRun.observer_status = { status: 'partial', attempted_pairs: 12,
    valid_pairs: 8, rejected_pairs: 4, rejected_response_count: 5 };
  setReply(partialObservationRun);
  await vm.runInContext('pollRunProgress()', context);
  assert.equal(get('images').children.length, 24);
  assert.ok(get('run-facts').children.some((node) => node.textContent === '8 / 12 valid responses · 4 pairs rejected · partial'),
    'A completed render must visibly distinguish valid observations from unresolved rejected pairs');
  assert.equal(timers.size, 0, 'Stop polling when the run settles');
  assert.equal(get('prepare-json').value, 'Unsaved continuation');

  context.initial = run(24, 'observing');
  vm.runInContext('lastRuns = [initial]', context);
  reply = () => { throw new Error('Temporary network interruption'); };
  await vm.runInContext('pollRunProgress()', context);
  assert.equal([...timers.values()][0].delay, 60000, 'Back off after a read failure');
  assert.equal(get('prepare-json').value, 'Unsaved continuation');

  let resolveDetail;
  reply = (url) => url === '/api/runs' ? { runs: [run(24)] } : new Promise((resolve) => { resolveDetail = resolve; });
  const pending = vm.runInContext('pollRunProgress()', context);
  while (!resolveDetail) await new Promise(setImmediate);
  vm.runInContext("selectedId = 'another-run'; ++detailRequest", context);
  get('run-name').textContent = 'Newly selected run';
  resolveDetail(run(24));
  await pending;
  assert.equal(get('run-name').textContent, 'Newly selected run', 'Late progress responses must not overwrite user selection');

  // The loop is a read-only projection of this run's frozen policy and evidence.
  const treeText = node => typeof node === 'string' ? node : [node.textContent || '', ...node.children.map(treeText)].join(' ');
  const image = id => ({id, path: `/fixture/${id}.png`, sha256: id});
  const observation = {visible_relationships: ['Two forms share an interval'], unexpected_effects: ['The gap dominates'],
    explanation_gaps: ['Agency is uncertain'], uncertainty: 'Interpretation is unresolved', possible_next_question: 'Can the interval become the subject?'};
  const choice = (action, ids) => ({action, selected_image_ids: ids, commitments: 'Keep the tension unresolved',
    what_changed: 'The gap became more consequential', reason: 'Investigate what the image revealed', new_question: action === 'review' ? '' : 'Can the gap become the subject?'});
  const branch = {...image('cycle-01-image-000'), parent_image_id: 'image-001'};
  const initialImages = [0, 1, 2].map(i => image(`image-00${i}`));
  const report = {status: 'complete', origin: 'fixture_actor', observations: initialImages.map(im => ({image_id: im.id, ...observation})),
    selected_image_ids: [branch.id], human_review_requested: true, research: [{step: 0}],
    cycles: [{step: 0, choice: choice('research', ['image-001']), research: {status: 'hypotheses_ready', reasoning_calls: 7}},
      {step: 1, choice: choice('combine', ['image-001', 'image-002']), render: {status: 'complete', images: [branch]}, observation},
      {step: 2, choice: choice('review', [branch.id])}]};
  const workflow = {policy: {max_steps: 3, research_sessions: 1}, initial_image_count: 3,
    initial_image_ids: initialImages.map(i => i.id), development_directory: 'render/study_development', fixture: true, development: report};
  context.cycleResult = {development_cycle: workflow, images: [branch, ...initialImages].map(im => ({name: im.id + '.png', label: im.id,
    url: '/image/' + im.id, record: im}))};
  vm.runInContext("selectedRunRecord = {kind: 'artistic_studio_run', status: 'completed'}; renderCycleWorkspace(cycleResult)", context);
  assert.equal(get('run-stage-list').hidden, true, 'A fixed stage strip must not represent a cycle run');
  assert.equal(get('run-development-section').hidden, false);
  assert.match(treeText(get('run-cycle-map')), /Findings return to the next choice/);
  assert.match(treeText(get('run-cycle-summary')), /Paused for human review/);
  assert.match(treeText(get('run-cycle-summary')), /2 \/ 3 development actions recorded/);
  assert.match(treeText(get('run-cycle-summary')), /no human judgment/);
  assert.match(treeText(get('run-cycle-summary')), /Fixture example/);
  assert.equal(get('development-branches').children[0].dataset.branchId, 'image-000', 'Initial studies precede descendants even if files arrive out of order');
  const branchCard = get('development-branches').children.find(c => c.dataset.branchId === branch.id);
  assert.match(branchCard.className, /model-selected/);
  assert.match(treeText(branchCard), /Revises\s+Initial 2/);
  assert.match(treeText(branchCard), /Also draws on\s+Initial 3/);
  assert.equal(get('development-history').children.length, 3);
  const earlier = get('development-history').children[0].querySelector('details'); earlier.open = true;
  const inspection = branchCard.querySelector('details'); inspection.open = true;
  assert.match(treeText(inspection), /Agency is uncertain/);
  const picture = branchCard.querySelector('a');
  report.selected_image_ids = ['image-001'];
  vm.runInContext('renderCycleWorkspace(cycleResult)', context);
  assert.equal(get('development-history').children[0].querySelector('details').open, true, 'Polling preserves expanded evidence');
  assert.equal(branchCard.querySelector('a'), picture, 'Unchanged branch images retain their DOM identity');
  assert.equal(branchCard.querySelector('details').open, true, 'An updated selection preserves open image observations');
  branchCard.querySelector('button').listeners.click();
  assert.equal(get('development-branches').children[1].focused, true, 'Parent links revisit the exact branch');
  const initialSection = new Element(); initialSection.dataset.directory = 'research';
  const reopenedSection = new Element(); reopenedSection.dataset.directory = 'render/study_development/research/cycle-00';
  get('run-research').replaceChildren(initialSection, reopenedSection);
  const researchButton = get('development-history').children[2].children.find(c => c.tag === 'button');
  researchButton.listeners.click();
  assert.equal(reopenedSection.scrolled, true, 'Research links open the exact nested session');

  context.cycleResult.development_cycle.development = {status: 'partial', observations: [], cycles: []};
  context.cycleResult.images = [];
  vm.runInContext("selectedRunRecord = {status: 'researching', data: {progress_message: 'Exploring the question.'}}; renderCycleWorkspace(cycleResult)", context);
  assert.match(treeText(get('development-status')), /Exploring the question/);
  assert.match(treeText(get('development-history')), /first development choice follows at least three/);
  assert.doesNotMatch(treeText(get('run-cycle-summary')), /Human review requested/);
  vm.runInContext("selectedRunRecord = {status: 'failed'}; renderCycleWorkspace(cycleResult)", context);
  assert.match(treeText(get('development-status')), /has not been reached/);
  vm.runInContext('renderCycleWorkspace({})', context);
  assert.equal(get('run-development-section').hidden, true, 'Historical runs must not acquire a cycle they never executed');
  assert.equal(get('run-stage-list').hidden, false);
  // Mixed evidence uses saved timestamps across categories, including older runs
  // with no timestamp. A research step is not duplicated as an execution event.
  const savedFile = (name, at, type = 'json') => ({name, modified_at: at, url: '/files/' + name, type, bytes: 100});
  const stepFile = savedFile('research/step-000.json', '2026-09-09T14:00:00Z');
  const newImage = savedFile('render/new.png', '2026-09-09T14:03:00Z', 'image');
  const oldImage = savedFile('render/old.png', '2026-09-09T10:01:00-04:00', 'image');
  const logFile = savedFile('worker.log', '2026-09-09T14:04:00Z', 'log');
  const receipt = savedFile('receipt.json', '2026-09-09T14:04:01Z');
  const missingTime = savedFile('undated.txt', null, 'text');
  context.feedResult = {files: [oldImage, stepFile, receipt, logFile, newImage, missingTime], images: [oldImage, newImage],
    research: [{directory: 'research', fixture: true, steps: [{index: 0, action: {action: 'search', query: '<script>untrusted source</script>'}, status: 'completed'}]}],
    activity: [{id: stepFile.name, at: stepFile.modified_at, label: 'search', file: stepFile},
      {id: 'event-2', at: '2026-09-09T14:02:00Z', label: 'Rendering started', origin: 'studio coordinator'}]};
  vm.runInContext('resetRunFeed(); renderRunFeed(feedResult)', context);
  const feedIDs = () => get('run-updates').children.map(n => n.dataset.entryId);
  assert.deepEqual(feedIDs(), ['file:receipt.json', 'image:render/new.png', 'activity:studio coordinator:event-2',
    'image:render/old.png', 'research:research/step-000.json', 'file:undated.txt']);
  assert.match(treeText(get('run-updates').children[0]), /2 saved files updated/);
  assert.match(treeText(get('run-updates').children[4]), /Fixture actor/);
  assert.match(treeText(get('run-updates').children[5]), /Time not recorded/);
  const retained = get('run-updates').children[1];
  retained.querySelector('details').open = true;
  context.feedResult.activity.push({id: 'event-3', at: '2026-09-09T14:05:00Z', label: 'Completed'});
  vm.runInContext('renderRunFeed(feedResult)', context);
  assert.equal(get('run-updates').children[2], retained, 'New updates prepend without replacing an existing image');
  assert.equal(retained.querySelector('details').open, true);
  get('run-update-type').value = 'image'; vm.runInContext('renderFeedEntries()', context);
  assert.deepEqual(feedIDs(), ['image:render/new.png', 'image:render/old.png']);
  get('run-update-type').value = ''; get('run-update-search').value = 'untrusted source';
  vm.runInContext('renderFeedEntries()', context);
  assert.deepEqual(feedIDs(), ['research:research/step-000.json']);
  assert.match(treeText(get('run-updates')), /<script>untrusted source<\/script>/, 'Saved source text is rendered literally');
  get('run-update-search').value = 'no matching evidence'; vm.runInContext('renderFeedEntries()', context);
  assert.match(treeText(get('run-updates')), /No updates match/);
  vm.runInContext('resetRunFeed(); renderRunFeed({})', context);
  assert.match(treeText(get('run-updates')), /No updates saved yet/);
  assert.equal(get('run-updates-more').hidden, true);

  // A log preview is loaded only when opened, and reload remains read-only.
  const fileHolder = new Element(); context.fileHolder = fileHolder; context.logFile = logFile;
  let fileReads = 0;
  context.fetch = async (url, options) => {
    assert.equal(url, logFile.url); assert.ok(!options.method || options.method === 'GET'); fileReads++;
    return {ok: true, text: async () => 'latest saved log'};
  };
  vm.runInContext('feedFile(fileHolder, logFile)', context);
  assert.equal(fileReads, 0);
  const fileDetails = fileHolder.querySelector('details'); fileDetails.open = true; fileDetails.listeners.toggle();
  await new Promise(setImmediate);
  assert.equal(fileReads, 1); assert.match(treeText(fileHolder), /latest saved log/);
  fileHolder.querySelector('button').listeners.click(); await new Promise(setImmediate);
  assert.equal(fileReads, 2);
  // Stop controls use the existing endpoint without navigation, duplicate posts,
  // or applying a late response to a different selected run.
  const allNodes = () => {
    const found = new Set();
    const visit = node => { if (!node || typeof node !== 'object' || found.has(node)) return; found.add(node); for (const child of node.children || []) visit(child); };
    for (const node of nodes.values()) visit(node);
    return [...found];
  };
  context.document.querySelectorAll = selector => selector === '[data-stop-run]' ? allNodes().filter(node => node.dataset?.stopRun) : [];
  context.activeStopRun = {...run(1), kind: 'artistic_studio_run', title: 'Running inquiry', data: {run_id: 'studio-fixture'}};
  context.finishedStopRun = {...context.activeStopRun, id: 'finished', status: 'completed'};
  context.legacyStopRun = {...run(1), id: 'legacy'};
  get('run-search').value = ''; get('run-status-filter').value = '';
  vm.runInContext('lastRuns = [activeStopRun, finishedStopRun, legacyStopRun]; selectedId = activeStopRun.id; renderRuns(lastRuns); renderDetail(activeStopRun)', context);
  const buttons = () => allNodes().filter(node => node.dataset?.stopRun === 'run' && !node.hidden);
  assert.equal(buttons().length, 3, 'Recent inquiry, run list and selected workspace each expose Stop run');
  assert.ok(buttons().every(button => button.textContent === 'Stop run' && !button.disabled));
  assert.equal(allNodes().filter(node => ['finished', 'legacy'].includes(node.dataset?.stopRun)).length, 0);
  const posts = []; let resolveStop;
  context.fetch = async (url, options) => {
    posts.push({url, options});
    return new Promise(resolve => { resolveStop = resolve; });
  };
  let stoppedPropagation = false;
  const listStop = get('runs').children[0].children.at(-1).children[0];
  const stopping = listStop.listeners.click({stopPropagation() { stoppedPropagation = true; }});
  assert.equal(stoppedPropagation, true);
  assert.equal(posts.length, 1);
  assert.equal(posts[0].url, '/api/runs/run/stop');
  assert.equal(posts[0].options.method, 'POST');
  assert.equal(posts[0].options.headers['X-Kusama-Review'], '1');
  assert.equal(posts[0].options.body, '{}');
  assert.ok(buttons().every(button => button.disabled && button.textContent === 'Requesting stop…'));
  await get('stop-run').listeners.click();
  assert.equal(posts.length, 1, 'Another control cannot submit the same pending request');
  vm.runInContext('renderRuns(lastRuns)', context);
  assert.ok(buttons().every(button => button.disabled), 'Polling cannot reenable an in-flight request');
  vm.runInContext('renderDetail(finishedStopRun)', context);
  const otherProgress = get('run-progress').textContent;
  resolveStop({ok: true, json: async () => ({status: 'rendering', stop_requested: true})});
  await stopping;
  assert.equal(vm.runInContext('selectedRunRecord.id', context), 'finished');
  assert.equal(get('run-progress').textContent, otherProgress);
  assert.ok(buttons().every(button => button.disabled && button.textContent === 'Stopping…'));
  assert.match(get('stop-status').textContent, /Running inquiry.*after the current action.*kept/);
  vm.runInContext('renderRuns(lastRuns); renderDetail(activeStopRun)', context);
  assert.ok(buttons().every(button => button.disabled && button.textContent === 'Stopping…'));
  vm.runInContext("activeStopRun = {...activeStopRun, status: 'stopped'}; lastRuns = [activeStopRun]; renderRuns(lastRuns); renderDetail(activeStopRun)", context);
  assert.equal(buttons().length, 0, 'Settled runs lose the stop controls');

  context.retryStopRun = {...context.activeStopRun, id: 'retry', status: 'researching', title: 'Retry inquiry'};
  vm.runInContext('lastRuns = [retryStopRun]; renderRuns(lastRuns); renderDetail(retryStopRun)', context);
  context.fetch = async () => ({ok: false, json: async () => ({detail: 'Temporary connection failure'})});
  await get('stop-run').listeners.click();
  assert.equal(get('stop-run').disabled, false);
  assert.match(get('stop-status').textContent, /Temporary connection failure.*try again/);
  context.fetch = async () => ({ok: true, json: async () => ({status: 'researching', stop_requested: true})});
  await get('stop-run').listeners.click();
  assert.equal(get('stop-run').textContent, 'Stopping…');
  assert.equal(get('stop-run').disabled, true);
  console.log('Live polling and stop controls: saved state, single submission, pending display, failure recovery and selection races passed.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
