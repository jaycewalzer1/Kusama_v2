/* Actual frontend behavior against fixture metadata only. No user artifacts,
   inference, reviewer impersonation or human-review submissions are involved. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') {
    this.tag = tag; this.children = []; this.listeners = {}; this.attributes = {};
    this.dataset = {}; this.hidden = false; this.disabled = false; this.textContent = '';
    this._value = ''; this.classList = { toggle() {} };
  }
  get value() { return this._value; }
  set value(value) {
    this._value = this.tag === 'select' && !this.children.some((n) => n.value === value) ? '' : value;
  }
  get options() { return this.children; }
  get selectedIndex() { return this.children.findIndex((n) => n.value === this.value); }
  set selectedIndex(index) { this.value = this.children[index]?.value || ''; }
  append(...nodes) {
    for (const node of nodes) { node.parent = this; this.children.push(node); }
    if (this.tag === 'select' && this.children.length === nodes.length) this.value = this.children[0]?.value || '';
  }
  replaceChildren(...nodes) { this.children = []; this._value = ''; this.append(...nodes); }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  async dispatch(name) { for (const fn of this.listeners[name] || []) await fn({ target: this, currentTarget: this }); }
  focus() { this.focused = true; }
  reset() { this.resetCount = (this.resetCount || 0) + 1; }
  cloneNode() { const result = new Element(this.tag); result.href = this.href; return result; }
  querySelector(tag) {
    for (const child of this.children) {
      if (child.tag === tag) return child;
      const nested = child.querySelector?.(tag); if (nested) return nested;
    }
    return null;
  }
}
const selects = new Set(['review-collection', 'reviewer-slot', 'review-pair', 'advance-candidate']);
const nodes = new Map();
const get = (id) => {
  if (!nodes.has(id)) nodes.set(id, new Element(selects.has(id) ? 'select' : 'div'));
  return nodes.get(id);
};
const collection = (id, assigned = true) => ({
  id, title: `Fixture ${id}`, brief: 'Fixture brief',
  candidates: [{ id: 'private-parent', label: 'Private parent' }, { id: 'private-edit', label: 'Private method' }],
  pairs: [{ id: 'private-method-pair', candidate_ids: ['private-parent', 'private-edit'] }],
  ...(assigned ? { frozen_assignments: ['reviewer-3', 'reviewer-1', 'reviewer-2', 'reviewer-1'].map((reviewer_slot) => ({
    reviewer_slot, assignment_id: 'private-assignment', method: 'private-method',
    candidate_order: { A: 'private-edit', B: 'private-parent' },
  })) } : {}),
});
const collections = { assigned: collection('assigned'), ordinary: collection('ordinary', false), slow: collection('slow') };
const requestLog = [];
const storageWrites = [];
let delayed = null;
const context = vm.createContext({
  document: { getElementById: get, createElement: (tag) => new Element(tag) },
  localStorage: { getItem() { return 'Remembered ordinary name'; }, setItem(...args) { storageWrites.push(args); } },
  console, setTimeout() {}, clearTimeout() {},
  fetch: async (url, options) => {
    assert.ok(!options?.method || options.method === 'GET', 'Identity guidance must never submit a review');
    requestLog.push(url);
    let value;
    if (url.startsWith('/api/reviews/collections?')) {
      value = { collections: Object.values(collections).map((c) => ({
        id: c.id, title: c.title, candidate_count: 2, comparison_count: 1, reviewed_pair_ids: [],
      })) };
    } else {
      const match = url.match(/^\/api\/reviews\/collections\/([^/?]+)(.*)$/);
      assert.ok(match, `Unexpected fixture request ${url}`);
      const [, id, suffix] = match;
      if (!suffix) {
        value = id === 'slow' ? await new Promise((resolve) => { delayed = resolve; }) : collections[id];
      } else if (suffix === '/history') {
        value = { votes: [] };
      } else if (suffix.startsWith('/pairs/')) {
        const reviewer = decodeURIComponent(url.split('?reviewer=')[1]);
        assert.ok(id === 'ordinary' || ['reviewer-1', 'reviewer-2', 'reviewer-3'].includes(reviewer));
        value = { collection_id: id, pair_id: 'private-method-pair', assignment_id: 'fixture-assignment',
          reviewer, candidates: { A: collections[id].candidates[1], B: collections[id].candidates[0] },
          kind: 'before_after', presentation: { condition: 'statement_free' }, reviews: [] };
      } else throw new Error(`Unexpected fixture suffix ${suffix}`);
    }
    return { ok: true, json: async () => value };
  },
});
let source = fs.readFileSync('src/kusama/dashboard_static/dashboard.js', 'utf8');
source = source.replace(/^refresh\(\);$/m, '').replace(/^refreshReviews\(\)\.catch\(.*$/m, '');
vm.runInContext(source, context);
const run = (code) => vm.runInContext(code, context);
const pairRequests = () => requestLog.filter((url) => url.includes('/pairs/'));

(async () => {
  // Study labels use only a stable public suffix, with no brief, method,
  // protocol prefix or incomplete-output hash in the selector.
  const studyLabels = [];
  for (const [partition, count] of [['d', 4], ['h', 8]]) {
    for (let number = 1; number <= count; number++) {
      const study = `${partition}${String(number).padStart(2, '0')}`;
      context.labelFixture = { id: `private-protocol_e0-${study}`, title: 'E0 parent/edit comparison',
        brief: 'Private fixture text must not appear', candidate_count: 7 };
      const label = run('reviewCollectionLabel(labelFixture)');
      assert.equal(label, `E0 parent/edit comparison · study ${study} · 7 images`);
      assert.ok(!label.includes('private') && !label.includes('Private'));
      studyLabels.push(label);
    }
  }
  assert.equal(new Set(studyLabels).size, 12);
  context.labelFixture = { id: 'private-protocol_e0-d01_incomplete_a1b2c3d4', title: 'E0 parent/edit comparison', candidate_count: 4 };
  assert.equal(run('reviewCollectionLabel(labelFixture)'), 'E0 parent/edit comparison · study d01 · 4 images');
  context.labelFixture = { id: 'ordinary_e0-d01', title: 'An ordinary study', candidate_count: 3 };
  assert.equal(run('reviewCollectionLabel(labelFixture)'), 'An ordinary study · 3 images');
  context.labelFixture = { id: 'private-protocol_e0-d01_private-method', title: 'E0 parent/edit comparison', candidate_count: 7 };
  assert.equal(run('reviewCollectionLabel(labelFixture)'), 'E0 parent/edit comparison · 7 images');

  // Discovery loads metadata before a name is supplied; a remembered ordinary
  // name never selects the first allocated identity or dispatches a comparison.
  await run('refreshReviews()');
  assert.ok(requestLog.includes('/api/reviews/collections/assigned'));
  assert.equal(get('reviewer-name-field').hidden, true);
  assert.equal(get('reviewer-slot-field').hidden, false);
  assert.deepEqual(get('reviewer-slot').children.map((n) => [n.value, n.textContent]), [
    ['', 'Choose your allocated slot'], ['reviewer-1', 'reviewer-1'],
    ['reviewer-2', 'reviewer-2'], ['reviewer-3', 'reviewer-3'],
  ]);
  assert.equal(run('reviewerName()'), '');
  assert.equal(pairRequests().length, 0);
  assert.match(get('reviewer-guidance').textContent, /only the slot assigned to you/);
  assert.doesNotMatch(get('review-progress').textContent, /Remembered/);
  assert.ok(!JSON.stringify(get('reviewer-slot').children.map((n) => n.textContent)).includes('private'));

  await run('openReviewCollection()');
  assert.match(get('review-status').textContent, /ask the review coordinator/);
  assert.equal(get('reviewer-slot').focused, true);
  assert.equal(pairRequests().length, 0);
  // An incorrect entry is unselected by a real select element, then validated
  // locally; it never reaches the cryptic server assignment error.
  get('reviewer-slot').value = 'An arbitrary name';
  await get('reviewer-slot').dispatch('change');
  await run('openReviewCollection()');
  assert.equal(pairRequests().length, 0);

  get('reviewer-slot').value = 'reviewer-2';
  await get('reviewer-slot').dispatch('change');
  assert.equal(pairRequests().length, 0, 'Choosing an identity does not open a comparison');
  await run('openReviewCollection()');
  assert.equal(pairRequests().length, 1);
  assert.ok(pairRequests()[0].endsWith('?reviewer=reviewer-2'));
  assert.equal(get('review-workspace').hidden, false);
  assert.deepEqual(get('review-pair').children.map((n) => n.textContent), ['1 of 1']);
  assert.equal(storageWrites.length, 0, 'Allocated identities must not be persisted as remembered names');
  const formResets = get('vote-form').resetCount;
  await run('refreshReviews()');
  assert.equal(get('reviewer-slot').value, 'reviewer-2', 'Refresh preserves an explicitly chosen identity');
  assert.equal(get('vote-form').resetCount, formResets);

  // Ordinary collections retain free-form names and require an explicit open.
  get('review-collection').value = 'ordinary';
  await get('review-collection').dispatch('change');
  assert.equal(pairRequests().length, 1);
  assert.equal(get('reviewer-name-field').hidden, false);
  assert.equal(get('reviewer-slot-field').hidden, true);
  get('reviewer').value = '';
  await run('openReviewCollection()');
  assert.match(get('review-status').textContent, /Enter your own reviewer name/);
  assert.equal(pairRequests().length, 1);
  get('reviewer').value = 'Fixture reviewer';
  await get('reviewer').dispatch('input');
  await run('openReviewCollection()');
  assert.equal(pairRequests().length, 2);
  assert.ok(pairRequests()[1].endsWith('?reviewer=Fixture%20reviewer'));

  // A delayed allocation response cannot replace controls after a user returns
  // to an already loaded ordinary collection.
  get('review-collection').value = 'slow';
  const pending = get('review-collection').dispatch('change');
  while (!delayed) await new Promise(setImmediate);
  get('review-collection').value = 'ordinary';
  await get('review-collection').dispatch('change');
  assert.equal(get('start-review').disabled, false);
  assert.match(get('reviewer-guidance').textContent, /no allocated reviewer slots/);
  delayed(collections.slow);
  await pending;
  assert.equal(get('reviewer-name-field').hidden, false);
  assert.equal(run('reviewerName()'), 'Fixture reviewer');

  // Switching back to an assigned collection asks for allocation again.
  get('review-collection').value = 'assigned';
  await get('review-collection').dispatch('change');
  assert.equal(get('reviewer-slot').value, '');
  assert.equal(pairRequests().length, 2);
  console.log('Reviewer allocation: twelve neutral study labels, early metadata, unique neutral slots, explicit identity, ordinary names and stale-response checks passed.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
