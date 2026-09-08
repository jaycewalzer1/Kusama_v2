"""Offline matched workflow contracts; fixtures are not artistic evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from kusama import comparison_modes as modes
from kusama.artifacts import sha256, write_json
from kusama.ledger import SpecMismatch
from kusama.resources import OvernightConfig, SessionBudget, load_config
from kusama.reviews import ReviewStore
from test_e0_runner import FixtureIndex, FixtureDirector


class Backend:
    calls, loads, failure = [], 0, None
    def __init__(self, config): self.config = config
    def prepare(self):
        type(self).loads += 1
        return {'model_id': 'fixture-renderer', 'fixture_only': True}
    def prompt_token_count(self, prompt): return 10
    def close(self): pass
    def generate(self, prompt, seed, parent=None):
        type(self).calls.append({'prompt': prompt, 'seed': seed, 'parent': sha256(parent) if parent else None})
        if self.failure == len(self.calls): raise RuntimeError('Fixture failed rendering')
        if self.failure == 'interrupt': raise KeyboardInterrupt('Fixture uncertain call')
        return Image.new('RGB', (self.config.width, self.config.height), (len(self.calls)*10, seed, 2))


class Encoder:
    calls = 0
    def __init__(self, config, *, device): assert device == 'cpu'
    def images(self, paths):
        type(self).calls += 1
        return np.eye(512, dtype=np.float32)[[self.calls % 512]]
    def close(self): pass


class ArchiveFixture:
    def __init__(self): self.rows = {}
    def add(self, candidate, vector, *, quality, quality_origin, protected):
        assert quality is None and quality_origin == 'unavailable'
        if candidate['candidate_id'] in self.rows: assert self.rows[candidate['candidate_id']] == candidate
        self.rows[candidate['candidate_id']] = candidate
    def active_pool(self):
        return {'different': [], 'exploratory_fallback': list(self.rows.values()), 'protected': [], 'favored': [],
                'quality_available': False, 'underlying_candidate_count': len(self.rows)}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path/'overnight'; root.mkdir()
    monkeypatch.setattr('kusama.resources.shutil.disk_usage', lambda p: SimpleNamespace(total=10**12, used=10**11, free=9*10**11))
    config = OvernightConfig.model_validate({**load_config().model_dump(), 'device': 'cpu',
        'render_profiles': {'fixture': {'width': 64, 'height': 64, 'steps': 1, 'guidance': 1., 'precision': 'fp32', 'offload': 'none', 'vae_tiling': False}},
        'default_render_profile': 'fixture'})
    with SessionBudget(root, config): pass
    index = root/'index'; index.mkdir()
    write_json(index/'manifest.json', {'fixture_only': True})
    for i in range(2): Image.new('RGB', (64, 64), ('red', 'blue')[i]).save(index/f'source{i}.png')
    def index_factory(path):
        index = FixtureIndex(path)
        index.manifest['encoder']['revision'] = config.clip_revision
        return index
    monkeypatch.setattr(modes, 'RetrievalIndex', index_factory)
    calls, archives = [], {}
    def multimodal(director, system, request, images, validate, **kwargs):
        assert [row[0] for row in images[-2:]] == ['source0', 'source1']
        calls.append({'request': request, 'image_count': len(images), 'render_count': len(Backend.calls)})
        return {'fixture_only': True, 'metadata': {'input_tokens': 20, 'generated_tokens': 30},
                'output': validate({'instruction': f'Fixture prompt {len(calls)}', 'intended_effect': 'Unverified fixture prediction', 'uncertainty': 'Unknown'})}
    monkeypatch.setattr(modes, '_multimodal_json', multimodal)
    def archive(index, output, **kwargs):
        if str(output) not in archives: archives[str(output)] = ArchiveFixture()
        return archives[str(output)]
    brief = {'brief_id': 'fixture-brief', 'brief': 'An unresolved visual relationship.', 'partition': 'development', 'authorship': 'fixture_only'}
    Backend.calls, Backend.loads, Backend.failure, FixtureDirector.loads, Encoder.calls = [], 0, None, 0, 0
    return root, index, brief, calls, archive, archives


def execute(setup, mode=None):
    root, index, brief, _, archive, _ = setup
    return modes.run('fixture_compare', brief, index, root=root, mode=mode, backend_factory=Backend,
                     director_factory=FixtureDirector, encoder_factory=Encoder, archive_factory=archive)


def test_plan_no_execution_equal_attempt_and_selection_ceilings(setup):
    root, index, brief, *_ = setup
    result = modes.plan('fixture_compare', brief, index, root=root)
    spec = json.loads(Path(result['plan']).read_text())
    assert result['renderer_attempt_ceiling'] == 16
    assert spec['per_mode']['renderer_attempt_ceiling'] == 4 and spec['per_mode']['pool_candidate_count'] == 4
    assert spec['per_mode']['pool_selection_opportunities_per_reviewer'] == 1
    assert Backend.calls == [] and FixtureDirector.loads == 0


def test_all_four_modes_count_sixteen_images_and_equal_real_review_pools(setup):
    result = execute(setup)
    assert len(result['modes']) == 4 and all(row['saved_candidates'] == 4 for row in result['modes'])
    assert all(row['generation_status'] == 'generation_complete' for row in result['modes'])
    assert result['evaluation_complete'] is False and len(Backend.calls) == 16
    store = ReviewStore(setup[0])
    assert len(store.collections()) == 4
    for collection in store.collections():
        pool = store.pool_assignment(collection['id'], 'fixture-reviewer')
        assert pool['complete'] and len(pool['candidates']) == 4 and pool['human_label'] is None
        assert store.pool_history(collection['id']) == []
    with SessionBudget(setup[0]) as budget:
        assert budget.summary()['renderer_attempts'] == 16
    assert len(setup[3]) == 13  # One direct prompt, four proposals/criticisms in each other mode.
    assert Encoder.calls == 4


def test_direct_samples_one_frozen_prompt_and_grounded_freezes_four_before_outputs(setup):
    execute(setup, 'direct_best_of_n')
    assert len(setup[3]) == 1 and len({row['prompt'] for row in Backend.calls}) == 1
    assert all(row['parent'] is None for row in Backend.calls)
    assert [row['seed'] for row in Backend.calls] == list(modes.SEEDS)
    execute(setup, 'grounded_no_iteration')
    assert all(row['render_count'] == 4 for row in setup[3][1:])
    assert len({row['prompt'] for row in Backend.calls[4:]}) == 4
    assert all(row['parent'] is None for row in Backend.calls[4:])


def test_generic_critique_uses_previous_saved_parent_and_search_retains_branches(setup):
    execute(setup, 'ordinary_critique_edit')
    folder = setup[0]/'comparisons'/'fixture_compare'/'ordinary_critique_edit'
    state = json.loads((folder/'state.json').read_text())
    assert [row['parent'] for row in Backend.calls[1:]] == [row['sha256'] for row in state['candidates'][:3]]
    execute(setup, 'kusama_search')
    search = json.loads((setup[0]/'comparisons'/'fixture_compare'/'kusama_search'/'state.json').read_text())
    assert Backend.calls[5]['parent'] == Backend.calls[6]['parent'] == search['candidates'][0]['sha256']
    archive = next(iter(setup[5].values()))
    assert len(archive.rows) == 4 and all(row['quality'] is None for row in archive.rows.values())
    selection = json.loads((setup[0]/'comparisons'/'fixture_compare'/'kusama_search'/'reasoning'/'parent-selection-03.json').read_text())
    assert selection['quality_origin'] == 'unavailable' and 'Seeded exploration' in selection['selection_basis']


def test_complete_resume_dispatches_no_model_or_renderer_calls(setup):
    execute(setup)
    first = len(Backend.calls), Backend.loads, len(setup[3]), FixtureDirector.loads, Encoder.calls
    execute(setup)
    assert (len(Backend.calls), Backend.loads, len(setup[3]), FixtureDirector.loads, Encoder.calls) == first


def test_failure_consumes_one_of_four_without_retries_and_pool_excluded(setup):
    Backend.failure = 2
    result = execute(setup, 'direct_best_of_n')
    assert len(Backend.calls) == 4 and result['modes'][0]['saved_candidates'] == 3
    assert result['modes'][0]['generation_status'] == 'incomplete_outputs'
    pool = ReviewStore(setup[0]).pool_assignment(result['modes'][0]['review_collection_id'], 'fixture-reviewer')
    assert pool['eligibility'] == 'incomplete_pool_excluded'
    execute(setup, 'direct_best_of_n')
    assert len(Backend.calls) == 4


def test_uncertain_execution_is_not_retried(setup):
    Backend.failure = 'interrupt'
    assert execute(setup, 'direct_best_of_n')['modes'][0]['generation_status'] == 'uncertain_execution'
    Backend.failure = None
    assert execute(setup, 'direct_best_of_n')['modes'][0]['generation_status'] == 'uncertain_execution'
    assert len(Backend.calls) == 1


def test_changed_raw_proposal_blocks_resume_and_sealed_brief_rejected(setup):
    execute(setup, 'ordinary_critique_edit')
    path = setup[0]/'comparisons'/'fixture_compare'/'ordinary_critique_edit'/'reasoning'/'proposal-01.json'
    row = json.loads(path.read_text()); row['output']['instruction'] = 'Tampered'; write_json(path, row)
    with pytest.raises(SpecMismatch): execute(setup, 'ordinary_critique_edit')
    assert len(Backend.calls) == 4
    with pytest.raises(ValueError, match='development briefs only'):
        modes.plan('sealed_rejected', {**setup[2], 'partition': 'sealed_test'}, setup[1], root=setup[0])


def test_interrupted_reasoning_stays_uncertain_without_another_call(setup, monkeypatch):
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(True)
        raise KeyboardInterrupt('Fixture interrupted Qwen')
    monkeypatch.setattr(modes, '_multimodal_json', interrupted)
    assert execute(setup, 'direct_best_of_n')['modes'][0]['generation_status'] == 'uncertain_execution'
    assert execute(setup, 'direct_best_of_n')['modes'][0]['generation_status'] == 'uncertain_execution'
    assert calls == [True] and Backend.calls == []


def test_sealed_index_manifest_blocks_all_modes_before_source_ranking(setup, monkeypatch):
    factory = modes.RetrievalIndex
    def contaminated(path):
        index = factory(path)
        index.manifest['source_pool'] = {'split': 'heldout'}
        index.query = lambda *a, **k: pytest.fail('Sealed manifest must stop before querying sources')
        return index
    monkeypatch.setattr(modes, 'RetrievalIndex', contaminated)
    with pytest.raises(ValueError, match='Sealed source pools'):
        execute(setup)
    assert Backend.calls == [] and Backend.loads == 0 and setup[3] == []
