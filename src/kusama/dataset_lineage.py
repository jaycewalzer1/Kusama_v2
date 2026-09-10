"""Conservative, read-only original-brief resolution for live review exports.

Names, directory prefixes and similar wording never establish independence.
Only registered byte identities, recorded parent links, and explicit curator
declarations can resolve a collection. This module loads no model or outcome.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

from .evidence import digest, file_hash

ADAPTER_VERSION = 'review-original-brief-v2'


class LineageProblem(ValueError):
    pass


class ReviewLineageResolver:
    def __init__(self, artifact_root, *, declarations=None):
        self.root = Path(artifact_root).resolve()
        self.declarations = declarations
        if declarations is not None:
            if (declarations.get('schema_version') != 1
                    or not all(isinstance(declarations.get(k), str) and declarations[k].strip() for k in ('author', 'reason'))
                    or not isinstance(declarations.get('collections'), dict)):
                raise ValueError('Curator lineage declarations require schema_version, author, reason and collections')
        path = self.root / 'jobs.sqlite3'
        self.db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) if path.is_file() else None
        if self.db:
            self.db.row_factory = sqlite3.Row
            self.db.create_function('lineage_sha256', 1, lambda value: hashlib.sha256(value.encode()).hexdigest())
            self.db.create_function('lineage_json_digest', 1, lambda value: digest(json.loads(value)) if value else None)
            self.db.execute('BEGIN')

    def close(self):
        if self.db:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _path(self, value, base=None):
        path = Path(value)
        path = (path if path.is_absolute() else (base or self.root) / path).resolve()
        if not path.is_relative_to(self.root):
            raise LineageProblem('lineage_path_outside_artifact_root')
        return path

    def _file(self, path, expected):
        if not path.is_file() or file_hash(path) != expected:
            raise LineageProblem('lineage_image_or_input_hash_mismatch')

    def _job(self, job_id, proofs):
        # Extract only ancestry metadata. In particular, do not read prompts,
        # action requests, observations, or sealed brief text into the resolver.
        row = self.db.execute('''SELECT job_id,directory,parent_id,spec_fingerprint,
            lineage_sha256(spec_json) AS actual_spec_sha256,
            json_extract(spec_json,'$.kind') AS kind,
            json_extract(spec_json,'$.inspiration_job') AS inspiration_job,
            json_extract(spec_json,'$.source_input_sha256') AS source_input_sha256,
            json_extract(spec_json,'$.project_id') AS project_id,
            json_extract(spec_json,'$.plan.parent') AS parent_path,
            json_extract(spec_json,'$.plan.parent_sha256') AS parent_sha256,
            json_extract(spec_json,'$.sealed') AS sealed,
            json_extract(spec_json,'$.partition') AS partition,
            json_extract(spec_json,'$.split') AS split,
            json_extract(spec_json,'$.runtime.partition') AS runtime_partition,
            json_extract(spec_json,'$.runtime.sealed') AS runtime_sealed,
            json_extract(spec_json,'$.runtime.protocol_manifest_sha256') AS protocol_manifest_sha256,
            lineage_json_digest(json_extract(spec_json,'$.runtime')) AS runtime_digest,
            json_extract(spec_json,'$.frozen_request_sha256') AS frozen_request_sha256,
            json_type(spec_json,'$.request') AS request_type,
            json_type(spec_json,'$.request.parent') AS request_parent_type,
            json_extract(spec_json,'$.plan_sha256') AS plan_sha256,
            json_extract(spec_json,'$.mode') AS mode,
            json_extract(spec_json,'$.stage') AS stage
            FROM jobs WHERE job_id=?''', (job_id,)).fetchone()
        if row is None:
            raise LineageProblem('missing_original_job')
        job = dict(row)
        if job['actual_spec_sha256'] != job['spec_fingerprint']:
            raise LineageProblem('lineage_job_spec_hash_mismatch')
        proofs.append({'type': 'registered_job', **job})
        if (job['sealed'] or job['runtime_sealed']
                or any(job[key] in ('sealed', 'sealed_test', 'test') for key in ('partition', 'split', 'runtime_partition'))):
            raise LineageProblem('sealed_original_lineage')
        return job

    def _studio(self, project_id, proofs):
        row = self.db.execute("SELECT json_extract(data,'$.artifacts'),json_extract(data,'$.creation_spec.parent_sha256') FROM studio_projects WHERE id=?", (project_id,)).fetchone()
        if row is None:
            raise LineageProblem('missing_studio_project')
        artifacts = json.loads(row[0])
        root = artifacts.get('root')
        if not root or root.get('origin') != 'imported' or root.get('sha256') != row[1]:
            raise LineageProblem('invalid_studio_import_identity')
        # These stable records exclude mutable observations, decisions and UI.
        proofs.append({'type': 'studio_artifact_records', 'project_id': project_id,
                       'records_sha256': digest(artifacts), 'artifacts': artifacts,
                       'creation_parent_sha256': row[1]})
        return artifacts

    def _studio_origin(self, project_id, parent_sha256, proofs, trail):
        artifacts = self._studio(project_id, proofs)
        matching = [a for a in artifacts.values() if a['sha256'] == parent_sha256]
        if not matching:
            raise LineageProblem('edit_parent_absent_from_studio_project')
        for artifact in matching:
            visited = set()
            while artifact.get('parent_artifact_id') is not None:
                if artifact['id'] in visited:
                    raise LineageProblem('cyclic_studio_parent_lineage')
                visited.add(artifact['id'])
                artifact = artifacts.get(artifact['parent_artifact_id'])
                if artifact is None:
                    raise LineageProblem('missing_studio_parent_artifact')
            if artifact['id'] != 'root':
                raise LineageProblem('disconnected_studio_parent_lineage')
        root = artifacts['root']
        if not root.get('source_path'):
            raise LineageProblem('missing_studio_import_source')
        origin = self._origin(self._path(root['source_path']), root['sha256'], proofs, trail)
        self._file(self._path(root['path']), root['sha256'])
        return origin

    def _bound_ancestor_file(self, directory, filename, expected):
        matches = []
        for ancestor in (directory, *directory.parents):
            if not ancestor.is_relative_to(self.root):
                break
            path = self._path(ancestor / filename)
            if path.is_file() and file_hash(path) == expected:
                matches.append(path)
        if len(matches) != 1:
            raise LineageProblem('missing_or_ambiguous_frozen_authority_hash')
        return matches[0]

    def _protocol_origin(self, job, proofs, trail):
        directory = self._path(job['directory'])
        manifest_path = self._bound_ancestor_file(directory, 'manifest.json', job['protocol_manifest_sha256'])
        manifest = json.loads(manifest_path.read_text())
        relative = directory.relative_to(manifest_path.parent).parts
        if len(relative) != 3 or relative[0] != 'episodes' or relative[2] not in ('parent', 'edits'):
            raise LineageProblem('unbound_protocol_episode_directory')
        bid, stage = relative[1:]
        entries = [x for x in manifest.get('briefs', []) if x.get('brief_id') == bid]
        if manifest.get('schema_version') != 1 or not manifest.get('protocol_id') or len(entries) != 1:
            raise LineageProblem('unbound_public_protocol_brief')
        entry = entries[0]
        proofs.append({'type': 'public_protocol_partition', 'path': str(manifest_path),
                       'sha256': job['protocol_manifest_sha256'], 'protocol_id': manifest['protocol_id'],
                       'brief_id': bid, 'partition': entry.get('partition'),
                       'registered_episode_directory': str(directory.parent)})
        # This happens before any per-episode file, private brief or image read.
        if entry.get('partition') in ('sealed', 'sealed_test', 'test'):
            raise LineageProblem('sealed_original_lineage')
        if entry.get('partition') != 'development' or not isinstance(entry.get('brief'), str) or not entry['brief'].strip():
            raise LineageProblem('unsupported_public_protocol_partition')
        runtime_path = directory.parent / 'runtime_identity.json'
        if not runtime_path.is_file() or digest(json.loads(runtime_path.read_text())) != job['runtime_digest']:
            raise LineageProblem('protocol_runtime_identity_mismatch')
        proofs.append({'type': 'frozen_protocol_runtime', 'path': str(runtime_path),
                       'sha256': file_hash(runtime_path), 'runtime_digest': job['runtime_digest']})
        group = 'public-protocol:' + job['protocol_manifest_sha256'] + ':' + bid
        if stage == 'parent':
            if job['request_type'] != 'object' or job['request_parent_type'] != 'null':
                raise LineageProblem('invalid_protocol_original_parent_request')
        else:
            frozen_path = directory.parent / 'frozen.json'
            if not job['frozen_request_sha256'] or not frozen_path.is_file():
                raise LineageProblem('missing_protocol_frozen_request')
            frozen = json.loads(frozen_path.read_text())
            wire = {k: v for k, v in frozen.items() if k != 'request_sha256'}
            if (frozen.get('request_sha256') != job['frozen_request_sha256'] or digest(wire) != job['frozen_request_sha256']
                    or frozen.get('protocol_id') != manifest['protocol_id'] or frozen.get('brief_id') != bid
                    or frozen.get('partition') != 'development'):
                raise LineageProblem('protocol_frozen_request_identity_mismatch')
            parent = frozen['parent']
            origin = self._origin(self._path(parent['path']), parent['sha256'], proofs, trail)
            if origin.get('group_id') != group:
                raise LineageProblem('protocol_parent_project_identity_mismatch')
            proofs.append({'type': 'frozen_protocol_request', 'path': str(frozen_path), 'sha256': file_hash(frozen_path),
                           'request_sha256': job['frozen_request_sha256'], 'parent': parent})
        brief_hash = digest(entry['brief'])
        proofs.append({'type': 'original_brief_digest', 'job_id': job['job_id'], 'brief_sha256': brief_hash,
                       'public_brief_id': bid, 'origin_group': group})
        return {'job_id': job['job_id'], 'group_id': group, 'brief_sha256': brief_hash}

    def _comparison_origin(self, job, proofs):
        directory = self._path(job['directory'])
        path = self._bound_ancestor_file(directory, 'plan.json', job['plan_sha256'])
        plan = json.loads(path.read_text())
        brief = plan.get('brief', {})
        proofs.append({'type': 'frozen_comparison_plan', 'path': str(path), 'sha256': job['plan_sha256'],
                       'comparison_id': plan.get('comparison_id'), 'partition': plan.get('partition'),
                       'brief_partition': brief.get('partition'), 'brief_id': brief.get('brief_id')})
        if plan.get('partition') in ('sealed', 'sealed_test', 'test') or brief.get('partition') in ('sealed', 'sealed_test', 'test'):
            raise LineageProblem('sealed_original_lineage')
        if (plan.get('schema_version') != 1 or plan.get('partition') != 'development' or brief.get('partition') != 'development'
                or not plan.get('comparison_id') or not brief.get('brief_id') or not isinstance(brief.get('brief'), str)
                or not brief['brief'].strip() or job['stage'] != 'rendering' or job['mode'] not in plan.get('modes', [])
                or directory.relative_to(path.parent).parts != (job['mode'], 'rendering')):
            raise LineageProblem('comparison_plan_identity_mismatch')
        group = 'comparison-plan:' + job['plan_sha256']
        brief_hash = digest(brief['brief'])
        proofs.append({'type': 'original_brief_digest', 'job_id': job['job_id'], 'brief_sha256': brief_hash,
                       'public_brief_id': brief['brief_id'], 'origin_group': group})
        return {'job_id': job['job_id'], 'group_id': group, 'brief_sha256': brief_hash}

    def _job_origin(self, job_id, proofs, trail, expected_studio=None):
        marker = ('job', job_id)
        if marker in trail:
            raise LineageProblem('cyclic_job_lineage')
        trail = trail | {marker}
        job = self._job(job_id, proofs)
        directory = self._path(job['directory'])
        if job['protocol_manifest_sha256']:
            return self._protocol_origin(job, proofs, trail)
        if job['plan_sha256']:
            return self._comparison_origin(job, proofs)
        if job['kind'] == 'connection_generation':
            if not job['inspiration_job'] or not job['source_input_sha256']:
                raise LineageProblem('missing_generation_source_identity')
            source = self._job(job['inspiration_job'], proofs)
            source_input = self._path(source['directory']) / 'input.json'
            self._file(source_input, job['source_input_sha256'])
            proofs.append({'type': 'frozen_source_input', 'path': str(source_input), 'sha256': job['source_input_sha256']})
            return self._job_origin(source['job_id'], proofs, trail)
        if job['kind'] == 'controlled_edit':
            if not job['parent_id'] or not job['parent_sha256'] or not job['parent_path']:
                raise LineageProblem('missing_edit_parent_identity')
            parent_job = self._job(job['parent_id'], proofs)
            if parent_job['kind'] != 'studio_episode' or not parent_job['project_id']:
                raise LineageProblem('unsupported_edit_parent_lineage')
            if expected_studio is not None and parent_job['project_id'] != expected_studio:
                raise LineageProblem('review_studio_job_identity_mismatch')
            origin = self._studio_origin(parent_job['project_id'], job['parent_sha256'], proofs, trail)
            self._file(self._path(job['parent_path'], directory), job['parent_sha256'])
            return origin
        if job['kind'] == 'connection_inspiration':
            brief_json = self.db.execute("SELECT json_quote(json_extract(spec_json,'$.brief')) FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]
            brief = json.loads(brief_json)
        elif job['kind'] == 'reference_creative_experiment':
            packet = self.db.execute("SELECT path,sha256 FROM artifacts WHERE job_id=? AND path='packet.json'", (job_id,)).fetchone()
            if packet is None:
                raise LineageProblem('missing_frozen_creative_packet')
            path = self._path(packet['path'], directory)
            self._file(path, packet['sha256'])
            proofs.append({'type': 'frozen_creative_packet', 'path': str(path), 'sha256': packet['sha256']})
            brief = json.loads(path.read_text()).get('brief')
        else:
            raise LineageProblem('unsupported_original_brief_lineage')
        if not brief:
            raise LineageProblem('missing_frozen_original_brief')
        brief_sha256 = digest(brief)
        proofs.append({'type': 'original_brief_digest', 'job_id': job_id, 'brief_sha256': brief_sha256,
                       'policy': 'Exact frozen brief content groups repeated wording conservatively; differing wording does not establish semantic independence.'})
        return {'job_id': job_id, 'brief_sha256': brief_sha256}

    def _origin(self, path, expected, proofs, trail, expected_studio=None):
        marker = ('file', str(path), expected)
        if marker in trail:
            raise LineageProblem('cyclic_import_lineage')
        if self.db is None:
            self._file(path, expected)
            raise LineageProblem('missing_job_lineage_database')
        owners = []
        for row in self.db.execute('''SELECT a.artifact_id,a.job_id,a.path,a.sha256,j.directory
                FROM artifacts a JOIN jobs j ON a.job_id=j.job_id WHERE a.sha256=?''', (expected,)):
            if self._path(row['path'], self._path(row['directory'])) == path:
                owners.append(dict(row))
        if not owners:
            self._file(path, expected)
            raise LineageProblem('unregistered_artifact')
        if len(owners) != 1:
            raise LineageProblem('ambiguous_registered_artifact_owner')
        proofs.append({'type': 'registered_artifact', **owners[0]})
        origin = self._job_origin(owners[0]['job_id'], proofs, trail | {marker}, expected_studio)
        self._file(path, expected)
        return origin

    def resolve(self, collection, *, collection_sha256=None):
        proofs, origins, issues = [], [], []
        cid = collection['id']
        collection_sha256 = collection_sha256 or digest(collection)
        tokens = {f'review-collection:{cid}'}
        for candidate in collection['candidates']:
            tokens.add('review-candidate:' + digest([cid, candidate['id']]))
            tokens.add('candidate-sha256:' + candidate['sha256'])
            try:
                path = self._path(candidate['path'])
                studio = candidate.get('studio')
                if studio:
                    if self.db is None:
                        raise LineageProblem('missing_job_lineage_database')
                    artifact = self._studio(studio['project_id'], proofs).get(studio['artifact_id'])
                    if not artifact or artifact['sha256'] != candidate['sha256'] or self._path(artifact['path']) != path:
                        raise LineageProblem('review_studio_artifact_identity_mismatch')
                    if studio['artifact_id'] == 'root':
                        origin = self._studio_origin(studio['project_id'], candidate['sha256'], proofs, set())
                    else:
                        origin = self._origin(path, candidate['sha256'], proofs, set(), studio['project_id'])
                else:
                    origin = self._origin(path, candidate['sha256'], proofs, set())
                origins.append(origin)
            except (LineageProblem, sqlite3.Error, KeyError, TypeError, json.JSONDecodeError) as error:
                issues.append({'candidate_id': candidate['id'], 'reason': str(error)})
        jobs = sorted({x['job_id'] for x in origins})
        groups = sorted({x.get('group_id', x['job_id']) for x in origins})
        briefs = sorted({x['brief_sha256'] for x in origins})
        tokens.update('origin-job:' + x for x in jobs)
        tokens.update('origin-group:' + x for x in groups)
        tokens.update('original-brief:' + x for x in briefs)
        # Ancestor bytes also connect imported copies across projects, even if
        # none of those ancestors appears in the reviewed pair itself.
        for proof in proofs:
            if proof['type'] == 'registered_artifact':
                tokens.add('candidate-sha256:' + proof['sha256'])
            elif proof['type'] == 'studio_artifact_records':
                tokens.update('candidate-sha256:' + a['sha256'] for a in proof['artifacts'].values())
            elif proof['type'] == 'registered_job':
                if proof['parent_sha256']:
                    tokens.add('candidate-sha256:' + proof['parent_sha256'])
                if proof['kind'] in ('connection_inspiration', 'reference_creative_experiment'):
                    tokens.add('origin-job:' + proof['job_id'])
        declaration = self.declarations and self.declarations['collections'].get(cid)
        if declaration is not None:
            if (declaration.get('collection_sha256') != collection_sha256
                    or type(declaration.get('sealed')) is not bool
                    or not all(isinstance(declaration.get(k), str) and declaration[k].strip()
                               for k in ('project_id', 'brief_id', 'trajectory_id'))):
                raise ValueError('Curator lineage declaration must bind exact collection bytes and explicit canonical IDs')
            proofs.append({'type': 'explicit_curator_declaration', 'manifest_sha256': digest(self.declarations),
                           'author': self.declarations['author'], 'reason': self.declarations['reason'], **declaration})
        # A declaration can supply missing ancestry or group differently worded
        # briefs, but never excuse contradictory ownership, cycles or bad bytes.
        missing = {'missing_job_lineage_database', 'unregistered_artifact'}
        declared_missing = declaration is not None and all(x['reason'] in missing for x in issues)
        status = 'resolved' if (not issues and len(briefs) == 1) or declared_missing else 'unresolved'
        if len(briefs) > 1 or any('ambiguous' in x['reason'] or 'mismatch' in x['reason'] or 'cyclic' in x['reason'] for x in issues):
            status = 'ambiguous'
        sealed = any(x['reason'] == 'sealed_original_lineage' for x in issues) or bool(declaration and declaration['sealed'])
        if sealed:
            status = 'unresolved'
        fallback = 'quarantined:' + collection_sha256
        canonical = {key: fallback for key in ('project_id', 'brief_id', 'trajectory_id')}
        if status == 'resolved':
            canonical = ({key: 'curated:' + declaration[key] for key in canonical} if declaration else
                         {'project_id': 'origin-group:' + groups[0], 'brief_id': 'original-brief:' + briefs[0],
                          'trajectory_id': 'origin-group:' + groups[0]})
        return {'adapter': ADAPTER_VERSION, 'status': status, **canonical, 'sealed': sealed,
                'collection_id': cid, 'collection_sha256': collection_sha256, 'lineage_ids': sorted(tokens),
                'original_jobs': jobs, 'original_brief_sha256': briefs, 'proofs': proofs, 'issues': issues,
                'original_groups': groups,
                'declaration_supplied': declaration is not None,
                'boundary': 'Exact frozen wording and recorded ancestry support grouping, not semantic independence. Differently worded versions of one brief require explicit curator grouping. Collection-local candidate IDs never establish cross-brief ancestry.'}
