"""Unregistered recovery artifacts must not become new executable instructions."""
import json

import pytest

from kusama.creative import run
from kusama.ledger import Ledger
from test_creative import setup, FakeDirector, FakeFlux


@pytest.mark.parametrize('mutation', ['path_escape', 'different_prompt', 'missing_direction'])
def test_crash_before_direction_registration_revalidates_before_model_loading(setup, monkeypatch, mutation):
    source, config = setup
    original = Ledger.register_artifact

    def fail_after_write(self, job_id, path, *args, **kwargs):
        if str(path) == 'directions.json':
            raise OSError('Crash after directions write, before artifact registration')
        return original(self, job_id, path, *args, **kwargs)

    with monkeypatch.context() as injection:
        injection.setattr(Ledger, 'register_artifact', fail_after_write)
        with pytest.raises(OSError, match='before artifact registration'):
            run(source, config, directions=3, stage='direct', job_id='integrity', director_factory=FakeDirector)
    directory = config.output_dir / 'integrity'
    aggregate = directory / 'directions.json'
    data = json.loads(aggregate.read_text())
    if mutation == 'path_escape':
        data['directions'][0]['id'] = '../escaped'
    elif mutation == 'different_prompt':
        data['directions'][0]['flux_prompt'] = 'This instruction was never produced by the saved Qwen call.'
    else:
        data['directions'].pop()
    aggregate.write_text(json.dumps(data))

    def forbidden_backend(config):
        pytest.fail('Corrupt/unvalidated saved directions must be rejected before loading a model')

    with pytest.raises(ValueError):
        run(resume_dir=directory, stage='render', backend_factory=forbidden_backend,
            director_factory=forbidden_backend)
    assert FakeFlux.calls == 0
    assert not (directory.parent / 'escaped-r1.png').exists()
    assert not list(directory.glob('d*-r*.png'))
