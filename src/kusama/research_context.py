"""Explicit, token-counted research prompt projections before inference admission."""
from copy import deepcopy
from pathlib import Path
import re
from types import SimpleNamespace

from .director import MODEL_ID, _prepare_multimodal_inputs
from .evidence import digest


class ContextCapacityPause(ValueError):
    """No inference occurred; retain the session when essential inputs cannot fit."""
    def __init__(self, admission):
        self.admission = admission
        super().__init__(f"Research context saved at capacity: {admission['input_tokens']} input tokens; "
                         f"limit {admission['input_token_limit']} with {admission['reserved_output_tokens']} "
                         f"output tokens reserved from {admission['combined_token_limit']}. "
                         'No model call was reserved or dispatched; all research evidence is retained.')


def candidate_page(sources, *, step, offset=0, limit=8, existing=None):
    """Stable provider-interleaved pages; paging is access policy, never a relevance vote."""
    if existing is None:
        groups = {}
        for source in sources:
            groups.setdefault(source.get('provider'), []).append(source)
        ordered = [group[i] for i in range(max(map(len, groups.values()), default=0))
                   for group in groups.values() if i < len(group)]
        total, start = len(ordered), offset
        if offset > total:
            raise ValueError('Candidate page offset exceeds the saved result count')
        selected = ordered[offset:offset + limit]
    else:
        total, start = existing['total'], existing['offset']
        selected = sources[:limit]
    following = start + len(selected) if start + len(selected) < total else None
    return deepcopy(selected), {'step': step, 'offset': start, 'returned': len(selected),
        'total': total, 'next_offset': following,
        'order': 'provider round-robin, then original encounter order; not relevance ranking',
        'recall': {'action': 'recall', 'recall_steps': [step], 'offset': following, 'limit': limit}
                  if following is not None and step is not None else None}


def _bounded_tool(tool, page_size, *, step=None):
    if not tool:
        return tool
    result = {}
    for key, value in tool.items():
        if key == 'sources':
            sources, page = candidate_page(value, step=tool.get('result_step', step), limit=page_size,
                                            existing=tool.get('sources_page'))
            cards = []
            for source in sources:
                card = {k: source[k] for k in ('candidate_id', 'source_id', 'provider', 'acquired', 'relevance_score',
                                               'exposure', 'exposure_origin') if k in source}
                for field, cap in [('title', 120), ('maker', 80), ('medium', 100), ('date', 48), ('culture', 64), ('description', 200)]:
                    if source.get(field) is not None:
                        card[field] = str(source[field])[:cap]
                cards.append(card)
            result.update(sources=cards, sources_page=page)
        elif key == 'sources_page':
            continue  # Rebuilt from the actual visible page, including narrowed recalled pages.
        elif key == 'recalled':
            result[key] = [{**{k: row[k] for k in ('index', 'tool_error') if k in row},
                            'tool_result': _bounded_tool(row.get('tool_result'), page_size, step=row['index'])}
                           for row in value]
        elif key in ('search_snapshots', 'snapshot_id', 'snapshot_path', 'snapshot_sha256',
                     'source_snapshot_id', 'source_identity'):
            continue  # Full transport provenance remains in the saved tool result.
        elif isinstance(value, str):
            result[key] = value[:1000 if key == 'text' else 320]
        elif key == 'links':
            result[key] = [{k: item[k] for k in ('id', 'link_id', 'title', 'kind') if k in item} for item in value[:2]]
            for link in result[key]:
                if isinstance(link.get('title'), str):
                    link['title'] = link['title'][:160]
            result['link_count'] = tool.get('link_count', len(value))
        elif key == 'scores':
            result[key] = [{k: item[k] for k in ('candidate_id', 'score') if k in item} for item in value[:page_size]]
            result['scores_total'] = len(value)
        elif key in ('provider_errors', 'exclusions'):
            result[key] = [{'excerpt': str(item)[:200]} for item in value[:2]]
            result[key + '_total'] = len(value)
        else:
            result[key] = deepcopy(value)
    if isinstance(tool.get('text'), str) and len(result.get('text', '')) < len(tool['text']):
        result['next_offset'] = tool.get('offset', 0) + len(result['text'])
        result['text_excerpt'] = True
    return result


def _rich_tool(tool, page_size, *, step=None):
    """Page source cards without shortening their descriptive evidence."""
    if not tool:
        return tool
    result = deepcopy(tool)
    for field in ('search_snapshots', 'snapshot_id', 'snapshot_path', 'snapshot_sha256',
                  'source_snapshot_id', 'source_identity'):
        result.pop(field, None)
    if 'sources' in tool:
        result['sources'], result['sources_page'] = candidate_page(tool['sources'],
            step=tool.get('result_step', step), limit=page_size, existing=tool.get('sources_page'))
        for card in result['sources']:
            for field in ('catalog_import', 'metadata_sha256'):
                card.pop(field, None)
    if 'recalled' in tool:
        result['recalled'] = [{**row, 'tool_result': _rich_tool(row.get('tool_result'), page_size,
                                                            step=row['index'])}
                             for row in tool['recalled']]
    return result


def _schema_without_annotations(value):
    if isinstance(value, list):
        return [_schema_without_annotations(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {key: ({name: _schema_without_annotations(schema) for name, schema in item.items()}
                  if key in ('properties', '$defs', 'patternProperties') else _schema_without_annotations(item))
            for key, item in value.items() if key not in ('title', 'description', 'default')}


def project_context(context, stage):
    """Keep problem, schema constraints, images, IDs and pending draft intact."""
    result = deepcopy(context)
    if stage == 0:
        return result
    if stage in (11, 12, 13, 14, 15):
        payload = result['payload']
        size = {11: 8, 12: 6, 13: 4, 14: 2, 15: 1}[stage]
        recent = payload.get('recent_actions', [])
        payload['last_tool_result'] = _rich_tool(payload.get('last_tool_result'), size,
            step=recent[-1].get('step') if recent else None)
        visible = set()
        def collect(tool):
            if not tool:
                return
            visible.update(c.get('candidate_id') for c in tool.get('sources', []))
            for row in tool.get('recalled', []):
                collect(row.get('tool_result'))
        collect(payload.get('last_tool_result'))
        # Existing memory ordering is retained; this projection adds no ranking.
        remaining = size
        for index, card in enumerate(payload.get('candidate_memory', [])):
            identifier = card.get('candidate_id')
            if identifier in visible or remaining <= 0:
                payload['candidate_memory'][index] = {k: v for k, v in card.items()
                    if k in ('candidate_id', 'source_id', 'provider', 'acquired',
                             'relevance_score', 'first_seen_step', 'title')}
                payload['candidate_memory'][index]['details_in'] = (
                    'last_tool_result.sources (same candidate_id)' if identifier in visible else
                    'Saved candidate metadata: open this candidate_id as source_id; no acquisition required.')
            else:
                remaining -= 1
        payload['recent_actions'] = recent[-2:]
        payload['output_schema'] = _schema_without_annotations(payload['output_schema'])
        payload['prompt_projection'] = {'stage': stage, 'candidate_page_size': size,
            'schema_display_titles_removed': True,
            'omitted_or_shortened_fields': ['last_tool_result.pages_and_transport_provenance',
                'candidate_memory.repeated_or_deferred_details', 'recent_actions.older_entries',
                'output_schema.annotations'],
            'policy': 'Smaller pages preserve complete visible descriptions. Recall the indicated step/offset '
                      'for other results, or open a saved candidate_id for its metadata. Full context retained.'}
        return result
    if stage in (7, 8, 9, 10):
        # Earlier projections preserve their historical meaning for replay. New
        # stages bound the fan-out itself and disclose how to retrieve every page.
        result = project_context(context, 6)
        payload = result['payload']
        page_size = {7: 8, 8: 4, 9: 2, 10: 1}[stage]
        recent = context['payload'].get('recent_actions', [])
        payload['last_tool_result'] = _bounded_tool(context['payload'].get('last_tool_result'), page_size,
                                                   step=recent[-1].get('step') if recent else None)
        if stage >= 8:
            payload['output_schema'] = _schema_without_annotations(payload['output_schema'])
        payload['prompt_projection'].update(stage=stage, candidate_page_size=page_size,
            schema_display_titles_removed=stage >= 8,
            policy='Bounded candidate pages and tool excerpts. Full results remain in saved steps; '
                   'recall the indicated step/offset for more candidates or read a source for full text. '
                   'Original problem, question, operator context, pending draft, images and schema constraints preserved.')
        payload['prompt_projection']['omitted_or_shortened_fields'] = sorted(set(
            payload['prompt_projection']['omitted_or_shortened_fields'] +
            ['last_tool_result.candidate_pages_and_tool_excerpts'] +
            (['output_schema.annotations'] if stage >= 8 else [])))
        return result
    if stage not in (1, 2, 3, 4, 5, 6):
        raise ValueError('Unknown research prompt projection stage')
    payload = result['payload']
    omitted = []
    # Cut low relevance metadata and old actions before touching source evidence.
    candidates = payload.get('candidate_memory', [])
    if stage >= 1 and candidates:
        count = max(0, 12 - stage * 2)
        ranked = sorted(candidates, key=lambda c: -(c.get('relevance_score') if c.get('relevance_score') is not None else -1))
        if len(ranked) > count:
            payload['candidate_memory'] = ranked[:count]
            omitted.append('candidate_memory.low_relevance_entries')
    if len(payload.get('recent_actions', [])) > 2:
        payload['recent_actions'] = payload['recent_actions'][-2:]
        omitted.append('recent_actions.older_entries')
    def tool_excerpt(tool, path):
        if not tool:
            return
        for field in ('source_identity', 'snapshot_id', 'snapshot_path', 'snapshot_sha256'):
            if field in tool:
                del tool[field]
                omitted.append(path + '.' + field)
        for index, source in enumerate(tool.get('sources', [])):
            prefix = f'{path}.sources[{index}]'
            for field in ('catalog_import', 'metadata_sha256'):
                if field in source:
                    del source[field]
                    omitted.append(prefix + '.' + field)
            if stage >= 2:
                cap = {2: 240, 3: 120, 4: 0, 5: 0, 6: 0}[stage]
                for field in ('description', 'tags'):
                    original = source.get(field)
                    if isinstance(original, (str, list)):
                        excerpt = original[:cap] if isinstance(original, str) else original[:(4 if stage < 4 else 0)]
                        if excerpt != original:
                            source[field] = excerpt
                            omitted.append(prefix + '.' + field)
        for index, row in enumerate(tool.get('recalled', [])):
            tool_excerpt(row.get('tool_result'), f'{path}.recalled[{index}].tool_result')
    tool_excerpt(payload.get('last_tool_result'), 'last_tool_result')
    if stage >= 2:
        for index, source in enumerate(payload.get('candidate_memory', [])):
            cap = 120 if stage == 2 else 0
            if len(source.get('description', '')) > cap:
                source['description'] = source['description'][:cap]
                omitted.append(f'candidate_memory[{index}].description')
    if stage >= 3:
        for index, source in enumerate(payload.get('source_context', [])):
            for field in ('text', 'claims'):
                value = source.get(field)
                if value:
                    source[field] = value[:240] if isinstance(value, str) else value[:1]
                    if source[field] != value:
                        omitted.append(f'source_context[{index}].{field}')
    if stage >= 4:
        for field, count in (('recent_actions', 2), ('notebook', 1)):
            if len(payload.get(field, [])) > count:
                payload[field] = payload[field][-count:]
                omitted.append(field + '.older_entries')
        for entry in payload.get('notebook', []):
            for field in ('note', 'reason'):
                if len(entry.get(field) or '') > 320:
                    entry[field] = entry[field][:320]
                    omitted.append('notebook[*].' + field)
    if stage >= 5:
        # Preserve every operational ID and descriptive source card. Omit
        # transport provenance and reference repeated candidate metadata instead
        # of sending a second identical copy; complete records remain saved.
        hits = (payload.get('last_tool_result') or {}).get('sources', [])
        by_id = {hit.get('candidate_id'): hit for hit in hits if hit.get('candidate_id')}
        for index, hit in enumerate(hits):
            for field in ('url', 'origin'):
                if field in hit:
                    del hit[field]
                    omitted.append(f'last_tool_result.sources[{index}].{field}')
            for field in ('description', 'tags', 'acquired_source_id'):
                if field in hit and hit[field] in ('', None, []):
                    del hit[field]
                    omitted.append(f'last_tool_result.sources[{index}].{field}')
            if 'prompt_excerpt' in hit:
                del hit['prompt_excerpt']
                omitted.append(f'last_tool_result.sources[{index}].prompt_excerpt')
        for index, candidate in enumerate(payload.get('candidate_memory', [])):
            hit = by_id.get(candidate.get('candidate_id'))
            if hit is None:
                continue
            shared = [key for key in ('title', 'maker', 'description')
                      if key in candidate and candidate[key] == hit.get(key)]
            if shared:
                for key in shared:
                    del candidate[key]
                    omitted.append(f'candidate_memory[{index}].{key}')
                candidate['details_in'] = 'last_tool_result.sources (same candidate_id)'
    if stage >= 6:
        for path, entries in [('candidate_memory', payload.get('candidate_memory', [])),
                              ('last_tool_result.sources', (payload.get('last_tool_result') or {}).get('sources', []))]:
            for index, entry in enumerate(entries):
                for field, cap in [('title', 96), ('maker', 80), ('medium', 80), ('culture', 64), ('date', 48)]:
                    value = entry.get(field)
                    if isinstance(value, str) and len(value) > cap:
                        entry[field] = value[:cap]
                        omitted.append(f'{path}[{index}].{field}')
    payload['prompt_projection'] = {
        'stage': stage, 'schema_display_titles_removed': False,
        'omitted_or_shortened_fields': sorted({re.sub(r'\[\d+\]', '[*]', path) for path in omitted}),
        'field_path_notation': '[*] groups affected entries; full source context remains saved',
        'policy': 'Token-budget projection; full context is preserved in the decision source file. '
                  'Schema titles/constraints, images, problem, current question, analysis and pending draft preserved. '
                  'Recall earlier steps or read acquired sources for more evidence.'}
    return result


def fit_context(context, count_tokens, *, output_tokens=1800, combined_limit=8192):
    limit = combined_limit - output_tokens
    attempts = []
    stages = ([0, 11, 12, 13, 14, 15, *range(1, 11)]
              if context['payload'].get('context_projection_policy') == 'rich_candidate_pages_v1' else range(11))
    for stage in stages:
        projected = project_context(context, stage)
        count = count_tokens(projected)
        attempts.append({'stage': stage, 'input_tokens': count})
        if count <= limit:
            return projected, {'full_context_sha256': digest(context), 'stage': stage,
                'input_tokens': count, 'input_token_limit': limit,
                'reserved_output_tokens': output_tokens, 'combined_token_limit': combined_limit,
                'json_serialization': 'compact', 'attempts': attempts}
    raise ContextCapacityPause({'full_context_sha256': digest(context), 'stage': stage,
        'input_tokens': count, 'input_token_limit': limit, 'reserved_output_tokens': output_tokens,
        'combined_token_limit': combined_limit, 'json_serialization': 'compact', 'attempts': attempts})


def admit_context(context, config, *, base=None):
    # This loads only the pinned local processor, never model weights or network.
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=config.model_revision,
        cache_dir=config.cache_dir, local_files_only=True, trust_remote_code=False, use_fast=False)
    holder = SimpleNamespace(config=config, processor=processor, execution_settings={'status': 'processor_only'})
    def count(projected):
        from .image_prompt import image_label
        entries = [(image_label(image),
                    ((Path(base) / image['path']) if base else Path(image['path']))) for image in projected['images']]
        prepared, _ = _prepare_multimodal_inputs(holder, projected['system'], projected['payload'], entries,
            allow_text_only=not entries, compact_json=True)
        return prepared['metadata']['input_tokens']
    return fit_context(context, count, output_tokens=config.max_new_tokens, combined_limit=config.max_input_tokens)
