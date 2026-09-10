"""Budgeted renderer sessions with durable invocation recovery."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import uuid
from PIL import Image
from .artifacts import save_image, sha256
from .costs import CostJournal, tracked_local_call
from .evidence import immutable_json
from .experiment import Measurement
from .inspiration import require_charged_invocation
from .ledger import BudgetExceeded, SpecMismatch
from .resources import now


def _profile_unchanged(receipt):
    if receipt and receipt.get('file') and sha256(Path(receipt['file'])) != receipt['file_sha256']:
        raise SpecMismatch('Frozen render profile file changed; no load or dispatch authorized')


def _preload_seconds(budget, kind, supplied):
    """Use the reservation policy plus observed load overhead before loading."""
    seconds = budget.estimated_seconds(kind, supplied)
    load_seconds = 0
    if (budget.root/'costs.sqlite3').exists():
        with CostJournal(budget.root, read_only=True) as costs:
            for row in costs.summary()['records']:
                start, finish = row['start'], row['finish']
                model = start.get('model', '').casefold()
                relevant = 'flux' in model if kind == 'renderer' else 'qwen' in model
                if relevant and start.get('scope') == 'model_load' and finish:
                    load_seconds = max(load_seconds, 1.25*(finish.get('runtime_seconds') or 0))
    return seconds+load_seconds


class UncertainExecution(RuntimeError):
    pass


class RendererSession:
    def __init__(self, budget, ledger, job_id, directory, config, backend_factory, *, render_profile_receipt=None):
        self.budget, self.ledger, self.job_id, self.directory = budget, ledger, job_id, Path(directory)
        self.config, self.backend = config, backend_factory(config)
        self.prepared = False
        self.metadata = {}
        self.session_id = uuid.uuid4().hex[:12]
        self.render_profile_receipt = render_profile_receipt

    def close(self):
        self.backend.close()

    def attempt(self, index, request, *, estimated_seconds=180):
        _profile_unchanged(self.render_profile_receipt)
        ledger, budget = self.ledger, self.budget
        path = self.directory / f'image-{index:02}.png'
        record_path = self.directory / f'attempt-{index:02}.json'
        key = f'{self.job_id}:render:{index}'
        attempt = ledger.get_attempt(self.job_id, index)
        if attempt is not None:
            require_charged_invocation(ledger, self.job_id, index, budget, key, 'renderer')
            if attempt.get('request') != request:
                raise SpecMismatch('Saved charged render request changed; existing pixels cannot be relabeled')
        if attempt is not None and record_path.exists():
            ledger.verify_artifacts(self.job_id)
            record = json.loads(record_path.read_text())
            if record['request'] != request:
                raise SpecMismatch('Frozen render request changed')
            if record['status'] == 'completed':
                output = record['output']
                if sha256(Path(output['path'])) != output['sha256']:
                    raise SpecMismatch('Recorded output bytes changed')
                ledger.register_artifact(self.job_id, path.name, attempt_index=index, kind='image')
                ledger.register_artifact(self.job_id, record_path.name, attempt_index=index, kind='image_record')
                ledger.save_attempt(self.job_id, index, status='completed', payload={'record': record_path.name})
                budget.settle(key, 'completed')
                return record
            if record['status'] == 'technical_failure':
                if path.exists():
                    raise UncertainExecution('Unexpected saved output after technical failure must be reconciled before retry')
                ledger.register_artifact(self.job_id, record_path.name, attempt_index=index, kind='technical_failure')
                ledger.save_attempt(self.job_id, index, status='failed', payload={'record': record_path.name})
                budget.settle(key, 'failed')
                return record
        if attempt is not None and path.exists():
            # Byte-preserving recovery after an interrupted write; disclose the recovery.
            with Image.open(path) as image:
                if image.mode != 'RGB' or image.size != (self.config.width, self.config.height):
                    raise UncertainExecution('Interrupted output is invalid; reconcile without blind retry')
            recovered = {'status': 'completed', 'request': request, 'output': {'path': str(path), 'sha256': sha256(path)},
                         'measurements': {}, 'backend': {}, 'recovery': 'Verified saved output after existing charged invocation; original timing may be unavailable.'}
            immutable_json(record_path, recovered)
            ledger.register_artifact(self.job_id, path.name, attempt_index=index, kind='image')
            ledger.register_artifact(self.job_id, record_path.name, attempt_index=index, kind='image_record')
            ledger.save_attempt(self.job_id, index, status='completed', payload={'record': record_path.name, 'recovery': True})
            budget.settle(key, 'completed')
            return recovered
        if attempt is not None:
            ledger.save_attempt(self.job_id, index, status='uncertain')
            budget.settle(key, 'uncertain')
            raise UncertainExecution('Uncertain charged image execution requires reconciliation; no retry dispatched')
        if path.exists() or record_path.exists():
            raise SpecMismatch('Uncharged output cannot be admitted as execution')
        parent = request.get('parent')
        if parent and sha256(Path(parent['path'])) != parent['sha256']:
            raise SpecMismatch('Saved parent bytes changed')
        if not self.prepared:
            if budget.summary()['renderer_remaining'] < 1:
                raise BudgetExceeded('No renderer capacity remains; model load not dispatched')
            deadline = datetime.fromisoformat(budget.deadline)-timedelta(minutes=budget.config.handoff_reserve_minutes)
            if now()+timedelta(seconds=_preload_seconds(budget, 'renderer', estimated_seconds)) >= deadline:
                raise BudgetExceeded('Execution window exhausted before model load; handoff reserve retained')
            with tracked_local_call(budget.root, self.job_id, 'flux-load-'+self.session_id, 'black-forest-labs/FLUX.2-klein-base-4B', scope='model_load') as cost:
                with Measurement(self.config.device) as measurement:
                    self.metadata = self.backend.prepare()
                cost['measurements'] = measurement.result
            self.prepared = True
        if hasattr(self.backend, 'prompt_token_count') and self.backend.prompt_token_count(request['instruction']) > self.config.max_sequence_length:
            raise ValueError('Frozen instruction exceeds FLUX token capacity; no inference or retry authorized')
        if not budget.reserve(key, 'renderer', estimated_seconds=estimated_seconds):
            raise UncertainExecution('Global image reservation already exists without local evidence')
        ledger.open_attempt(self.job_id, index, payload={'request': request})
        if not ledger.reserve_call(self.job_id, index):
            raise UncertainExecution('Local charged invocation cannot be repeated')
        try:
            with tracked_local_call(budget.root, self.job_id, f'image-{index}', 'black-forest-labs/FLUX.2-klein-base-4B') as cost:
                with Measurement(self.config.device) as measurement:
                    output = self.backend.generate(request['instruction'], request['seed'], Path(parent['path']) if parent else None)
                    if output.mode != 'RGB' or output.size != (self.config.width, self.config.height):
                        raise ValueError('Renderer returned invalid image shape/mode')
                    save_image(output, path)
                cost['measurements'] = measurement.result
            record = {'status': 'completed', 'request': request, 'output': {'path': str(path), 'sha256': sha256(path)},
                      'measurements': measurement.result, 'backend': self.metadata}
            immutable_json(record_path, record)
            ledger.register_artifact(self.job_id, path.name, attempt_index=index, kind='image')
            ledger.register_artifact(self.job_id, record_path.name, attempt_index=index, kind='image_record')
            ledger.save_attempt(self.job_id, index, status='completed', payload={'record': record_path.name})
            budget.settle(key, 'completed')
            return record
        except BaseException as exc:
            if not isinstance(exc, Exception) or path.exists():
                ledger.save_attempt(self.job_id, index, status='uncertain', payload={'error': f'{type(exc).__name__}: {exc}'})
                budget.settle(key, 'uncertain')
                raise UncertainExecution('Interrupted/uncertain charged image; retained evidence, no retry') from exc
            record = {'status': 'technical_failure', 'request': request, 'error': f'{type(exc).__name__}: {exc}',
                      'output': None, 'backend': self.metadata, 'human_label': None, 'measurements': measurement.result,
                      'retry_policy': 'One identical request retry permitted; both attempts counted.'}
            immutable_json(record_path, record)
            ledger.register_artifact(self.job_id, record_path.name, attempt_index=index, kind='technical_failure')
            ledger.save_attempt(self.job_id, index, status='failed', payload={'record': record_path.name})
            budget.settle(key, 'failed')
            return record


