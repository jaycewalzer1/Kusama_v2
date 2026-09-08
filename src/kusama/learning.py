"""Human-grounded selector and guarded multimodal SFT; no fabricated checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np

from .datasets import load_snapshot
from .evaluation import digest, file_hash, immutable_json

LABELS = ('A', 'B', 'tie', 'unsure')
REVERSE = {'A': 'B', 'B': 'A', 'tie': 'tie', 'unsure': 'unsure'}
QLORA_PROFILE = {'quantization': 'nf4', 'compute_dtype': 'bfloat16', 'rank': 32, 'alpha': 64,
                 'dropout': .05, 'learning_rate': 1e-4, 'microbatch': 1, 'gradient_accumulation': 32,
                 'gradient_clip': 1.0, 'gradient_checkpointing': True, 'max_epochs': 2,
                 'language_projections': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}


def pair_features(e_a, e_b):
    a, b = np.asarray(e_a, dtype=np.float64), np.asarray(e_b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('CLIP embeddings require matching finite vectors')
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        raise ValueError('CLIP embeddings must be nonzero')
    delta = a / np.linalg.norm(a) - b / np.linalg.norm(b)
    return np.concatenate((delta, np.abs(delta)))


def reversed_presentation(row):
    row = deepcopy(row)
    row['presented_images'] = [dict(image, position=position) for position, image in zip(('A', 'B'), reversed(row['presented_images']))]
    for event in row.get('model_input', []):
        if isinstance(event, dict) and 'candidate_order' in event:
            event['candidate_order'] = deepcopy(row['presented_images'])
    row['label'] = REVERSE[row['label']]
    row['augmentation_of'] = row['event_id']
    # Keep event and group identity: reversal is not an independent judgment.
    return row


def softmax(values):
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def normalized_label_probabilities(sequence_log_probabilities, *, temperature=1.0):
    """Normalize COMPLETE label+stop sequence scores; labels need not be one token."""
    values = np.asarray([sequence_log_probabilities[label] for label in LABELS], dtype=float)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    if not np.isfinite(values).all():
        raise ValueError('Every allowed complete label sequence requires a finite score')
    return dict(zip(LABELS, softmax(values / temperature).tolist()))


def classification_metrics(probabilities, labels):
    probabilities = np.asarray(probabilities)
    if not len(labels):
        return {'count': 0, 'log_loss': None, 'brier': None, 'accuracy': None}
    targets = np.asarray([LABELS.index(label) for label in labels])
    truth = np.eye(len(LABELS))[targets]
    return {'count': len(labels), 'log_loss': float(-np.log(np.maximum(probabilities[np.arange(len(labels)), targets], 1e-12)).mean()),
            'brier': float(np.square(probabilities-truth).sum(axis=1).mean()),
            'accuracy': float((probabilities.argmax(axis=1) == targets).mean())}



def calibrate_development(log_probabilities, rows):
    """One temperature selected using development groups only; never test labels."""
    if not rows or any(row.get('split') != 'development' for row in rows):
        raise ValueError('Calibration requires explicitly development-only rows')
    values = np.asarray(log_probabilities, dtype=float)
    if values.shape != (len(rows), len(LABELS)) or not np.isfinite(values).all():
        raise ValueError('Calibration score shape or values invalid')
    candidates = []
    for temperature in (.25, .5, 1.0, 2.0, 4.0):
        probabilities = softmax(values / temperature)
        losses = []
        for group in sorted({row['split_group'] for row in rows}):
            indices = [i for i, row in enumerate(rows) if row['split_group'] == group]
            losses.append(classification_metrics(probabilities[indices], [rows[i]['label'] for i in indices])['log_loss'])
        candidates.append((float(np.mean(losses)), temperature))
    loss, temperature = min(candidates)
    return {'temperature': temperature, 'grouped_development_log_loss': loss,
            'independent_development_groups': len({row['split_group'] for row in rows}),
            'metrics': classification_metrics(softmax(values / temperature), [row['label'] for row in rows])}


def _fit_logistic(x, labels, regularization, steps=1500):
    """Full-batch convex multinomial objective; intercept is not L2 penalized."""
    y = np.eye(len(LABELS))[[LABELS.index(label) for label in labels]]
    weights = np.zeros((x.shape[1], len(LABELS)))
    intercept = np.zeros(len(LABELS))
    # Lipschitz-safe step for softmax cross entropy and the L2 penalty.
    lipschitz = .5 * np.linalg.norm(np.column_stack([x, np.ones(len(x))]), ord=2) ** 2 / len(x) + regularization
    lr = 1 / max(lipschitz, 1e-6)
    for _ in range(steps):
        error = softmax(x @ weights + intercept) - y
        grad = x.T @ error / len(x) + regularization * weights
        bias_grad = error.mean(axis=0)
        weights -= lr * grad
        intercept -= lr * bias_grad
        if max(float(np.abs(grad).max()), float(np.abs(bias_grad).max())) < 1e-7:
            break
    return weights, intercept


def _counts(splits):
    return {split: {'rows': len(rows), 'independent_groups': len({r['split_group'] for r in rows}),
                    'human_review_events': len({event for r in rows for event in r['source_event_ids']})}
            for split, rows in splits.items()}


def _embedding_reader(cache):
    value = json.loads(Path(cache).read_text())
    if not all(value.get(key) for key in ('model_id', 'revision', 'preprocessing', 'embeddings')) or value.get('representation') != 'CLIP':
        raise ValueError('Embedding cache requires CLIP identity, pinned revision, preprocessing and image-hash keys')
    def features(row):
        return pair_features(*[value['embeddings'][image['sha256']] for image in row['presented_images']])
    return value, features


def train_selector(snapshot, output, *, embedding_cache=None, execute=False):
    if (Path(output) / 'result.json').exists():
        raise ValueError('Training output already has an immutable result; choose a new output directory')
    manifest, splits = load_snapshot(snapshot)
    if manifest['kind'] not in ('selector', 'judge_sft'):
        raise ValueError('Selector requires situated pairwise human labels')
    result = {'component': 'selector', 'snapshot_id': manifest['snapshot_id'], 'counts': _counts(splits),
              'status': 'insufficient_data', 'checkpoint': None, 'personalizations': {},
              'claims': 'No learned artistic advantage or test evaluation is implied.'}
    reviewers = sorted({(r['reviewer'], r['target'].get('dimension', 'unspecified')) for rows in splits.values() for r in rows})
    eligible = []
    for reviewer, dimension in reviewers:
        selected = {split: [r for r in rows if r['reviewer'] == reviewer and r['target'].get('dimension', 'unspecified') == dimension] for split, rows in splits.items()}
        train, dev = selected['train'], selected['development']
        classes = Counter(row['label'] for row in train)
        key = f'{reviewer}:{dimension}'
        if len({r['split_group'] for r in train}) < 3 or len({r['split_group'] for r in dev}) < 1 or len(classes) < 2:
            result['personalizations'][key] = {'status': 'unsupported_personalization', 'counts': _counts(selected),
                                               'constant_baseline': max(classes, key=classes.get) if classes else None,
                                               'missing_requirements': ['at least three train groups, one development group, and two original training classes']}
        else:
            eligible.append((key, train, dev))
    if not eligible:
        result['missing_requirements'] = ['genuine reviewer labels in at least five independent groups',
                                          'per reviewer/dimension: three train groups, one development group, two original label classes']
        immutable_json(Path(output) / 'result.json', result)
        return result
    if embedding_cache is None:
        result.update(status='capability_unavailable', missing_requirements=['pinned CLIP embedding cache keyed by reviewed image SHA-256'])
        immutable_json(Path(output) / 'result.json', result)
        return result
    descriptor, features = _embedding_reader(embedding_cache)
    if not execute:
        result.update(status='ready_not_executed', missing_requirements=['explicit --execute'], eligible_personalizations=len(eligible))
        immutable_json(Path(output) / 'result.json', result)
        return result
    models = {}
    for key, train, dev in eligible:
        augmented = train + [reversed_presentation(row) for row in train]
        x = np.asarray([features(row) for row in augmented])
        labels = [row['label'] for row in augmented]
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-8] = 1
        x = (x-mean)/scale
        dev_x = (np.asarray([features(row) for row in dev])-mean)/scale
        scored = []
        for l2 in (.001, .01, .1, 1.0, 10.0):
            weights, intercept = _fit_logistic(x, labels, l2)
            probabilities = softmax(dev_x @ weights + intercept)
            # Each brief/group has equal development weight despite row counts.
            losses = []
            for group in sorted({row['split_group'] for row in dev}):
                indexes = [i for i, row in enumerate(dev) if row['split_group'] == group]
                losses.append(classification_metrics(probabilities[indexes], [dev[i]['label'] for i in indexes])['log_loss'])
            scored.append((float(np.mean(losses)), l2, weights, intercept))
        loss, l2, weights, intercept = min(scored, key=lambda x: (x[0], x[1]))
        probabilities = softmax(dev_x @ weights + intercept)
        models[key] = {'mean': mean.tolist(), 'scale': scale.tolist(), 'weights': weights.tolist(), 'intercept': intercept.tolist(),
                       'regularization': l2, 'labels': LABELS}
        reverse_x = (np.asarray([features(reversed_presentation(row)) for row in dev])-mean)/scale
        reverse_p = softmax(reverse_x @ weights + intercept)[:, [1, 0, 2, 3]]
        favorites = [i for i, row in enumerate(dev) if row['label'] in ('A', 'B')]
        result['personalizations'][key] = {'status': 'trained', 'original_training_rows': len(train), 'augmented_rows': len(augmented),
                                          'original_review_events': len({eid for row in train for eid in row['source_event_ids']}),
                                          'independent_training_groups': len({row['split_group'] for row in train}),
                                          'grouped_development_log_loss': loss, 'regularization': l2,
                                          'development_metrics': classification_metrics(probabilities, [r['label'] for r in dev]),
                                          'order_reversal_mean_probability_error': float(np.abs(probabilities-reverse_p).mean()),
                                          'human_favorite_retention': float(np.mean([LABELS[probabilities[i].argmax()] == dev[i]['label'] for i in favorites])) if favorites else None,
                                          'retention_scope': 'pairwise development choices; not a six-candidate pool metric'}
    checkpoint = {'schema_version': 1, 'component': 'selector', 'snapshot_id': manifest['snapshot_id'],
                  'clip_descriptor': {k: descriptor[k] for k in ('model_id', 'revision', 'preprocessing', 'representation')},
                  'embedding_cache_sha256': file_hash(embedding_cache), 'models': models}
    path = immutable_json(Path(output) / 'selector.json', checkpoint)
    result.update(status='trained', checkpoint=str(path.resolve()), test_evaluated=False)
    immutable_json(Path(output) / 'result.json', result)
    return result


def validate_target_modules(module_names):
    """Exact language-module allowlist; never match vision or merger projections."""
    targets = []
    for name in module_names:
        parts = name.split('.')
        if parts[-1] not in QLORA_PROFILE['language_projections']:
            continue
        if any(term in name.lower() for term in ('visual', 'vision', 'merger', 'embed', 'lm_head')):
            continue
        if 'language_model' in parts and 'layers' in parts:
            targets.append(name)
    if not targets:
        raise ValueError('No explicitly enumerated language attention/MLP target modules found')
    return targets


def sft_examples(rows, component):
    examples = []
    for row in rows:
        if component == 'judge':
            text = json.dumps({'context': row['model_input'], 'reviewer': row['reviewer'],
                               'instruction': 'Judge the presented A then B images. Reply with exactly A, B, tie, or unsure.'}, sort_keys=True)
            images = row['presented_images']
            completion = row['label']
        else:
            text = json.dumps({'pre_action_state': row['model_input'],
                               'instruction': 'Return the executable artistic decision supported by this prior state.'}, sort_keys=True)
            # Explicit pre-action image list is part of target metadata for file
            # transport only; every hash must also occur in decision-time input.
            images = row['target'].get('input_images', [])
            input_text = json.dumps(row['model_input'], sort_keys=True)
            if any(image['sha256'] not in input_text for image in images):
                raise ValueError('Director input image was not available in decision-time evidence')
            completion = json.dumps(row['target']['action'], sort_keys=True)
        examples.append({'text': text, 'images': images, 'completion': completion, 'source_event_id': row['event_id']})
    return examples


class ResponseOnlyCollator:
    """Fail closed if the actual processor cannot preserve prefix or image order."""
    def __init__(self, processor, max_input_tokens=8192):
        self.processor = processor
        self.max_input_tokens = max_input_tokens
        self.audits = []

    def __call__(self, examples):
        if len(examples) != 1:
            raise ValueError('Validated multimodal profile supports microbatch one only')
        from PIL import Image
        row = examples[0]
        images = []
        for spec in row['images']:
            if file_hash(spec['path']) != spec['sha256']:
                raise ValueError('Training image hash changed')
            with Image.open(spec['path']) as image:
                images.append(image.convert('RGB').copy())
        content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': row['text']}]
        prefix = self.processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        eos = self.processor.tokenizer.eos_token
        if not eos:
            raise ValueError('An explicit shared completion stopping token is required')
        kwargs = {'images': images} if images else {}
        before = self.processor(text=[prefix], return_tensors='pt', **kwargs)
        batch = self.processor(text=[prefix + row['completion'] + eos], return_tensors='pt', **kwargs)
        prefix_length = before['input_ids'].shape[1]
        ids = batch['input_ids']
        if ids.shape[1] > self.max_input_tokens or ids.shape[1] <= prefix_length:
            raise ValueError('Actual processed length exceeds admission limit or completion is empty')
        if not ids[0, :prefix_length].equal(before['input_ids'][0]):
            raise ValueError('Processor changed prefix tokens across completion boundary')
        if images and not any(key.startswith('pixel_values') for key in batch):
            raise ValueError('Processor did not include actual visual tensors')
        labels = ids.clone()
        labels[:, :prefix_length] = -100
        labels[batch['attention_mask'] == 0] = -100
        batch['labels'] = labels
        self.audits.append({'source_event_id': row['source_event_id'], 'image_hash_order': [x['sha256'] for x in row['images']],
                            'input_tokens': ids.shape[1], 'prompt_tokens': prefix_length,
                            'completion_tokens': int((labels != -100).sum()),
                            'visual_tensor_shapes': {k: list(v.shape) for k, v in batch.items() if k.startswith('pixel_values')}})
        return batch


def score_allowed_labels(model, processor, example, *, max_input_tokens=8192):
    """Autoregressive sum over every label token and the same EOS; no one-token shortcut."""
    import torch
    collator = ResponseOnlyCollator(processor, max_input_tokens)
    scores = {}
    model.eval()
    with torch.no_grad():
        for label in LABELS:
            batch = collator([{**example, 'completion': label}])
            batch = {k: v.to(model.device) for k, v in batch.items()}
            labels = batch.pop('labels')
            logits = model(**batch).logits[:, :-1].float()
            targets = labels[:, 1:]
            active = targets != -100
            selected = logits.log_softmax(-1).gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            scores[label] = float(selected[active].sum())
    return normalized_label_probabilities(scores)


def train_sft(snapshot, output, *, component, model_path=None, execute=False, smoke=False, max_input_tokens=8192):
    if (Path(output) / 'result.json').exists():
        raise ValueError('Training output already has an immutable result; choose a new output directory')
    if component not in ('judge', 'director'):
        raise ValueError('Unknown SFT component')
    manifest, splits = load_snapshot(snapshot)
    if manifest['kind'] != f'{component}_sft':
        raise ValueError('SFT snapshot kind does not match component')
    counts = _counts(splits)
    missing_data = []
    if counts['train']['rows'] < 20:
        missing_data.append(f"train rows: {counts['train']['rows']}; require at least 20 genuine eligible rows")
    if counts['train']['independent_groups'] < 3:
        missing_data.append(f"train groups: {counts['train']['independent_groups']}; require at least 3")
    if counts['development']['independent_groups'] < 1:
        missing_data.append(f"development groups: {counts['development']['independent_groups']}; require at least 1")
    dependencies = {module: importlib.util.find_spec(module) is not None for module in ('torch', 'transformers', 'peft', 'trl', 'datasets', 'bitsandbytes')}
    result = {'component': component, 'snapshot_id': manifest['snapshot_id'], 'counts': counts, 'checkpoint': None,
              'profile': QLORA_PROFILE, 'dependencies': dependencies, 'mechanics_verified': False,
              'training_result_available': False, 'improvement_evaluated': False}
    if missing_data:
        result.update(status='insufficient_data', missing_requirements=missing_data)
    elif not all(dependencies.values()) or model_path is None or not Path(model_path).is_dir():
        result.update(status='capability_unavailable', missing_requirements=[f'missing dependency: {x}' for x, valid in dependencies.items() if not valid] + ([] if model_path and Path(model_path).is_dir() else ['local pinned model directory']))
    elif not execute:
        result.update(status='ready_not_executed', missing_requirements=['explicit --execute; disposable integration smoke precedes all full training'])
    else:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            result.update(status='capability_unavailable', missing_requirements=['CUDA with BF16 and verified bitsandbytes NF4 support for the optional QLoRA profile'])
        else:
            result = _execute_sft(splits, result, output, component, Path(model_path), smoke, max_input_tokens)
    immutable_json(Path(output) / 'result.json', result)
    return result


def _execute_sft(splits, result, output, component, model_path, smoke_only, max_input_tokens):
    """Optional real executor. This path must pass its own disposable update gate."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
    def load_base():
        base = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                                         quantization_config=quantization, dtype=torch.bfloat16, device_map={'': torch.cuda.current_device()})
        return prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    base = load_base()
    targets = validate_target_modules([name for name, _ in base.named_modules()])
    config = LoraConfig(r=32, lora_alpha=64, lora_dropout=.05, target_modules=targets, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(base, config)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any('lora_' not in name or any(x in name for x in ('visual', 'vision', 'merger', 'embed', 'lm_head')) for name, _ in trainable):
        raise ValueError('Unexpected trainable parameter; frozen-parameter gate failed')
    def frozen_hashes(current):
        import hashlib
        return {name: hashlib.sha256(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                for name, parameter in current.named_parameters() if not parameter.requires_grad}
    before_frozen = frozen_hashes(model)
    examples = sft_examples(splits['train'], component)
    development = sft_examples(splits['development'], component)
    collator = ResponseOnlyCollator(processor, max_input_tokens)
    # Admit every real train/development batch without truncation before an update.
    for example in examples + development:
        collator([example])
    batch = {k: v.to(model.device) for k, v in collator([examples[0]]).items()}
    model.train()
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-4)
    initial_adapters = {name: p.detach().clone() for name, p in trainable}
    loss = model(**batch).loss
    if not torch.isfinite(loss):
        raise ValueError('Disposable integration loss is not finite')
    loss.backward()
    gradient_names = [name for name, p in trainable if p.grad is not None and torch.isfinite(p.grad).all() and bool(p.grad.abs().sum() > 0)]
    if not gradient_names:
        raise ValueError('No finite nonzero intended adapter gradients')
    torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
    optimizer.step()
    if not any(not p.detach().equal(initial_adapters[name]) for name, p in trainable):
        raise ValueError('Disposable adapter update changed no trainable weights')
    if frozen_hashes(model) != before_frozen:
        raise ValueError('Frozen base parameter changed during integration update')
    integration_path = output / 'integration_only_adapter'
    model.save_pretrained(integration_path)
    model.eval()
    with torch.no_grad():
        reference_logits = model(**batch).logits.detach().cpu()
    # Reload onto the same unchanged base; no extra quantized model load required.
    reloaded_base = model.unload()
    reloaded = PeftModel.from_pretrained(reloaded_base, integration_path, is_trainable=False)
    reloaded.eval()
    with torch.no_grad():
        reload_logits = reloaded(**batch).logits.detach().cpu()
    if not torch.allclose(reference_logits, reload_logits, rtol=.002, atol=.002):
        raise ValueError('Adapter save/reload logits mismatch')
    integration = {'status': 'passed', 'evidence_kind': 'real_data_disposable_training_mechanics', 'loss': float(loss.detach()),
                   'gradient_names': gradient_names, 'frozen_parameters_unchanged': True, 'save_reload_verified': True,
                   'target_modules': targets, 'batch_audits': collator.audits, 'research_checkpoint': False}
    immutable_json(output / 'integration_report.json', integration)
    result.update(mechanics_verified=True, integration_adapter=str(integration_path.resolve()))
    if smoke_only:
        return result | {'status': 'integration_only', 'training_result_available': False}
    # Discard the disposable update: full training begins with fresh adapter weights.
    fresh_base = reloaded.unload()
    model = get_peft_model(fresh_base, config)
    args = SFTConfig(output_dir=str(output / 'training'), per_device_train_batch_size=1,
                     per_device_eval_batch_size=1, gradient_accumulation_steps=32, num_train_epochs=2,
                     learning_rate=1e-4, max_grad_norm=1.0, gradient_checkpointing=True, bf16=True,
                     max_length=None, packing=False, completion_only_loss=True,
                     dataset_kwargs={'skip_prepare_dataset': True}, remove_unused_columns=False,
                     report_to='none', save_strategy='no', eval_strategy='epoch')
    trainer = SFTTrainer(model=model, args=args, processing_class=processor, data_collator=collator,
                         train_dataset=Dataset.from_list(examples), eval_dataset=Dataset.from_list(development))
    trainer.train()
    checkpoint = output / f'{component}_adapter'
    trainer.save_model(str(checkpoint))
    processor.save_pretrained(checkpoint)
    if component == 'judge':
        probabilities = [score_allowed_labels(model, processor, example, max_input_tokens=max_input_tokens) for example in development]
        logs = np.log(np.maximum([[row[label] for label in LABELS] for row in probabilities], 1e-30))
        calibration = calibrate_development(logs, splits['development'])
        immutable_json(checkpoint / 'development_calibration.json', calibration)
        result['development_calibration'] = calibration
    return result | {'status': 'trained', 'checkpoint': str(checkpoint.resolve()), 'training_result_available': True,
                     'test_evaluated': False, 'release_status': 'not_released_pending_human_heldout_comparison'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('component', choices=('selector', 'judge', 'director'))
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--embedding-cache', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-input-tokens', type=int, default=8192)
    args = parser.parse_args(argv)
    if args.component == 'selector':
        result = train_selector(args.snapshot, args.output, embedding_cache=args.embedding_cache, execute=args.execute)
    else:
        result = train_sft(args.snapshot, args.output, component=args.component, model_path=args.model_path,
                           execute=args.execute, smoke=args.smoke, max_input_tokens=args.max_input_tokens)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] in ('trained', 'integration_only', 'ready_not_executed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
