/* Exercise the actual polling code with a minimal DOM and controlled HTTP replies.
   No browser, models, user artifacts or human-review submissions are involved. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') {
    this.tag = tag; this.children = []; this.dataset = {}; this.attributes = {};
    this.value = ''; this.hidden = false; this.disabled = false; this.checked = false;
    this.scrollTop = 0; this.classList = { toggle() {} };
  }
  append(...nodes) { for (const node of nodes) { node.parent = this; this.children.push(node); } }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  remove() { this.parent.children = this.parent.children.filter((child) => child !== this); }
  addEventListener() {}
  setAttribute(key, value) { this.attributes[key] = value; }
  querySelector(tag) {
    for (const child of this.children) {
      if (child.tag === tag) return child;
      const nested = child.querySelector(tag); if (nested) return nested;
    }
    return null;
  }
  closest() { return nodes.get('table-scroll'); }
  reset() { throw new Error('Progress polling must never reset a form'); }
}
const nodes = new Map();
const get = (id) => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
get('table-scroll').scrollTop = 41;
const window = { scrollX: 0, scrollY: 823, scrollTo(x, y) { this.scrollX = x; this.scrollY = y; } };
let reply;
let requestLog = [];
let timerId = 0;
const timers = new Map();
const context = vm.createContext({
  document: { getElementById: get, createElement: (tag) => new Element(tag) }, window,
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
// Explicit UI startup is separate from testing the polling transition itself.
source = source.replace(/^refresh\(\);$/m, '').replace(/^refreshReviews\(\)\.catch\(.*$/m, '');
vm.runInContext(source, context);
const run = (imageCount, status = 'rendering') => ({
  id: 'run', name: 'run', kind: 'reference_creative_experiment', title: 'Study', brief: 'Brief', status,
  model_calls: imageCount + 12, max_calls: 48, image_count: imageCount, planned_images: 24,
  runtime_seconds: imageCount * 165, runtime_scope: 'recorded_operations',
  artifacts: Array.from({ length: imageCount }, (_, i) => ({ name: `d01-r${i + 1}.png`, type: 'image', preview: true, url: `/image/${i}`, label: `Image ${i}` })),
  data: { images: Array.from({ length: imageCount }, (_, i) => ({ path: `d01-r${i + 1}.png`, size: [512, 640] })) },
});
const setReply = (detail) => { reply = (url) => url === '/api/runs' ? { runs: [detail] } : url.startsWith('/api/reviews/collections?') ? { collections: [] } : detail; };

(async () => {
  context.initial = run(0);
  vm.runInContext("lastRuns = [initial]; selectedId = 'run'; activeComparison = { assignment_id: 'keep-assignment' };", context);
  get('reviewer').value = 'Unsaved reviewer';
  get('review-notes').value = 'Unsaved artistic reasoning';
  get('review-pair').value = 'keep-pair';
  get('finished-quality-radio').checked = true;
  get('potential-radio').checked = true;
  get('advance-candidate').value = 'keep-candidate';
  get('advance-reason').value = 'Unsaved reason';
  const drafts = ['reviewer', 'review-notes', 'review-pair', 'advance-candidate', 'advance-reason'].map((id) => [id, get(id), get(id).value]);

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
  assert.equal(vm.runInContext('activeComparison.assignment_id', context), 'keep-assignment');
  for (const [id, node, value] of drafts) { assert.equal(get(id), node); assert.equal(get(id).value, value); }
  assert.equal(get('finished-quality-radio').checked, true);
  assert.equal(get('potential-radio').checked, true);
  assert.ok(!requestLog.some((url) => url.includes('/pairs/')), 'Polling must not reopen an active comparison');

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
  assert.ok(requestLog.some((url) => url.startsWith('/api/reviews/collections?')), 'Discover new review collections on completion');
  assert.equal(get('review-notes').value, 'Unsaved artistic reasoning');

  context.initial = run(24, 'observing');
  vm.runInContext('lastRuns = [initial]', context);
  reply = () => { throw new Error('Temporary network interruption'); };
  await vm.runInContext('pollRunProgress()', context);
  assert.equal([...timers.values()][0].delay, 60000, 'Back off after a read failure');
  assert.equal(get('review-notes').value, 'Unsaved artistic reasoning');

  let resolveDetail;
  reply = (url) => url === '/api/runs' ? { runs: [run(24)] } : new Promise((resolve) => { resolveDetail = resolve; });
  const pending = vm.runInContext('pollRunProgress()', context);
  while (!resolveDetail) await new Promise(setImmediate);
  vm.runInContext("selectedId = 'another-run'; ++detailRequest", context);
  get('run-name').textContent = 'Newly selected run';
  resolveDetail(run(24));
  await pending;
  assert.equal(get('run-name').textContent, 'Newly selected run', 'Late progress responses must not overwrite user selection');
  console.log('Live polling: progressive images, draft preservation, scroll, completion, backoff and selection races passed.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
