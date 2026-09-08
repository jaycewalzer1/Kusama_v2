"""Offline protocol fixtures; these are not E0 generation or human evaluation."""
import json
from pathlib import Path

import pytest
from PIL import Image

from kusama.evaluation import (CONDITIONS, episode_brief, file_hash, freeze_episode,
                               prepare_protocol, protocol_status, record_attempt)


def prepared(tmp_path):
    source = tmp_path / 'fixture_holdouts.json'
    config = json.loads(Path('configs/e0.yaml').read_text())
    source.write_text(json.dumps({'schema_version': 1, 'protocol_id': config['protocol_id'],
        'briefs': [{'brief_id': f'e0-h{i+1:02}', 'brief': f'Clearly synthetic test fixture holdout {i}',
                    'partition': 'sealed_test', 'authorship': 'assistant_provisional'} for i in range(8)]}))
    prepare_protocol(tmp_path, Path('configs/e0.yaml'), holdout_file=source)
    return tmp_path


def frozen(tmp_path):
    prepared(tmp_path)
    image = tmp_path / 'parent.png'
    Image.new('RGB', (4, 4)).save(image)
    return freeze_episode(tmp_path, 'e0-d01', parent={'path': str(image), 'sha256': file_hash(image)},
                          action_requests={name: {'action': name} for name in CONDITIONS},
                          render_profile={'renderer': 'fixture', 'size': [4, 4]},
                          reference_snapshot_sha256='reference-snapshot', memory_snapshot_sha256='memory-snapshot')


def test_protocol_exact_counts_and_sealed_exposure(tmp_path):
    root = prepared(tmp_path)
    manifest = json.loads((root / 'manifest.json').read_text())
    assignments = [json.loads(line) for line in (root / 'review_assignments.jsonl').read_text().splitlines()]
    assert manifest['planned']['images'] == 84
    assert len([x for x in assignments if x['kind'] == 'parent_edit']) == 216
    assert len([x for x in assignments if x['kind'] == 'six_candidate_or_none']) == 36
    assert len({x['assignment_id'] for x in assignments}) == 252
    assert all('brief' not in x for x in manifest['briefs'] if x['partition'] == 'sealed_test')
    with pytest.raises(ValueError, match='only available'):
        episode_brief(root, 'e0-h01')
    with pytest.raises(ValueError, match='only available'):
        episode_brief(root, 'e0-h01', purpose='heldout_execution', episode_id='e0-h02')
    assert episode_brief(root, 'e0-h01', purpose='heldout_execution', episode_id='e0-h01')['brief']
    assert protocol_status(root)['missing_images'] == 84
    assert protocol_status(root)['genuine_judgments_in_protocol_store'] == 0
    prepare_protocol(root, Path('configs/e0.yaml'))  # Idempotent and no votes.


def test_frozen_parent_actions_seed_pair_and_no_quality_retry(tmp_path):
    freeze = frozen(tmp_path)
    assert freeze['edit_seeds'] == [23, 41]
    with pytest.raises(ValueError, match='three'):
        freeze_episode(tmp_path, 'e0-d01', parent=freeze['parent'], action_requests={'exploratory': {'a': 1}},
                       render_profile={'x': 1}, reference_snapshot_sha256='r', memory_snapshot_sha256='m')
    common = {'root': tmp_path, 'brief_id': 'e0-d01', 'condition': CONDITIONS[0], 'seed': 23,
              'request_sha256': freeze['request_sha256']}
    record_attempt(**common, attempt=1, status='succeeded', output_path=freeze['parent']['path'], execution_id='real-fixture-execution-1')
    with pytest.raises(ValueError, match='reconciled failure'):
        record_attempt(**common, attempt=2, status='succeeded', output_path=freeze['parent']['path'], execution_id='new', reconciled=True)
    assert protocol_status(tmp_path)['verified_edits'] == 1


def test_uncertain_retry_requires_reconciliation_and_counts(tmp_path):
    freeze = frozen(tmp_path)
    common = {'root': tmp_path, 'brief_id': 'e0-d01', 'condition': CONDITIONS[0], 'seed': 23,
              'request_sha256': freeze['request_sha256']}
    record_attempt(**common, attempt=1, status='uncertain', execution_id='first')
    with pytest.raises(ValueError, match='reconciled failure'):
        record_attempt(**common, attempt=2, status='technical_failure', execution_id='second')
    record_attempt(**common, attempt=2, status='technical_failure', execution_id='second', reconciled=True)
    with pytest.raises(ValueError, match='Invalid attempt'):
        record_attempt(**common, attempt=3, status='technical_failure', execution_id='third', reconciled=True)
    assert protocol_status(tmp_path)['recorded_edit_attempts'] == 2


def test_changed_parent_or_holdout_config_rejected(tmp_path):
    freeze = frozen(tmp_path)
    with pytest.raises(ValueError, match='configuration freeze'):
        freeze_episode(tmp_path, 'e0-h01', parent=freeze['parent'], action_requests=freeze['action_requests'],
                       render_profile=freeze['render_profile'], reference_snapshot_sha256='r', memory_snapshot_sha256='m')
    (tmp_path / 'sealed' / 'briefs.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        protocol_status(tmp_path)


def test_prepare_has_no_default_source_embedded_holdouts(tmp_path):
    with pytest.raises(ValueError, match='external holdout file'):
        prepare_protocol(tmp_path, Path('configs/e0.yaml'))
