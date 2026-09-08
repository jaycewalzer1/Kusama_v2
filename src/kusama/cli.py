"""One entry point for the source-connected frozen studio and evidence tools."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys
import zipfile

from .artifacts import sha256, write_json
from .resources import SessionBudget, load_config


def doctor(root=Path('artifacts/overnight'), config_path=Path('configs/overnight.yaml')):
    import psutil
    import torch
    from .config import MODEL_ID as FLUX_MODEL, MODEL_REVISION as FLUX_REVISION
    from .director import MODEL_ID as QWEN_MODEL, MODEL_REVISION as QWEN_REVISION
    from .retrieval import CLIP_MODEL
    root=Path(root).resolve()
    with SessionBudget(root,config_path) as budget:
        config=budget.config
        disk=shutil.disk_usage(root)
        models=[]
        for identifier,revision in ((FLUX_MODEL,FLUX_REVISION),(QWEN_MODEL,QWEN_REVISION),(CLIP_MODEL,config.clip_revision)):
            snapshot=Path(config.cache_dir).resolve()/('models--'+identifier.replace('/','--'))/'snapshots'/revision
            files=[p for p in snapshot.rglob('*') if p.is_file()] if snapshot.exists() else []
            models.append({'model_id':identifier,'revision':revision,'cached_files':len(files),
                           'cached_bytes':sum(p.stat().st_size for p in files),'snapshot':str(snapshot),
                           'weights_present':any(p.suffix in ('.safetensors','.bin') for p in files),
                           'training_enabled':False})
        report={'schema_version':1,'checked_at':datetime.now(timezone.utc).isoformat(),
            'runtime':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
                       'host_memory_bytes':psutil.virtual_memory().total,'available_memory_bytes':psutil.virtual_memory().available,
                       'mps_available':torch.backends.mps.is_available(),'cuda_available':torch.cuda.is_available(),
                       'assigned_device':config.device,'active_gpu_workload_limit':1},
            'disk':{'total_bytes':disk.total,'free_bytes':disk.free,'required_free_floor':max(config.minimum_free_bytes,int(config.minimum_free_fraction*disk.total))},
            'models':models,'budget':budget.summary(),'config':config.model_dump(),
            'packages':{name:importlib.metadata.version(name) for name in ('torch','transformers','diffusers','pydantic','fastapi','numpy','pillow')},
            'training_optional_dependencies':{},
            'capabilities':{'reference_ingestion':True,'whole_detail_clip_retrieval':True,'connection_proposals':True,
                            'generation_and_masked_continuation':True,'blinded_review_and_none_pool':True,
                            'human_preference_learning_evidence':False,'weight_updates_performed':False},
            'accounting_boundary':'Instrumented local Qwen/FLUX calls are counted. Coding-agent service token usage and human time are not available to this ledger.'}
        for name in ('peft','trl','bitsandbytes'):
            try:report['training_optional_dependencies'][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:report['training_optional_dependencies'][name]=None
        try:report['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        except subprocess.CalledProcessError:report['git_commit']=None
        write_json(root/'doctor.json',report)
    return report


def export_evidence(root, output):
    """Portable ZIP with consistent SQLite backups and exact artifact byte hashes.

Never alters frozen manifests to disguise original absolute paths. path_map
allows a consumer to resolve originals after relocation; execution on a new
host still requires explicit path reconciliation and cached model checkpoints.
"""
    root,output=Path(root).resolve(),Path(output).resolve()
    if not root.is_dir() or output.exists():
        raise ValueError('Export requires an existing artifact root and a fresh output directory')
    files=[p for p in sorted(root.rglob('*')) if p.is_file() and not p.is_relative_to(output)
           and not any(part in ('exports','__pycache__') for part in p.relative_to(root).parts)
           and not p.name.endswith(('-wal','-shm','.tmp','.lock')) and p.suffix!='.log']
    output.mkdir(parents=True)
    database_dir=output/'database_snapshots';database_dir.mkdir()
    records=[]
    archive=output/'kusama_evidence.zip'
    try:
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as zipped:
            for path in files:
                if path.is_symlink():
                    raise ValueError('Evidence export does not follow symlinks')
                relative=str(path.relative_to(root));source=path;kind='exact_bytes'
                if path.suffix in ('.sqlite3','.sqlite'):
                    source=database_dir/relative;source.parent.mkdir(parents=True,exist_ok=True)
                    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as original,sqlite3.connect(source) as copy:
                        original.backup(copy)
                    kind='consistent_sqlite_backup'
                before=sha256(source)
                zipped.write(source,'artifacts/'+relative)
                if sha256(source)!=before:raise ValueError('Artifact changed during export: '+relative)
                records.append({'path':relative,'archive_path':'artifacts/'+relative,'sha256':before,
                                'bytes':source.stat().st_size,'original_path':str(path),'copy_kind':kind})
            manifest={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'original_root':str(root),
                      'files':records,'path_map':{r['original_path']:r['archive_path'] for r in records},
                      'models_included':False,'source_code_included':False,
                      'resume_boundary':'Restore original root for exact local resume, or reconcile original absolute ledger/config paths explicitly on another host. Model caches and code are separate.',
                      'unchanged_original_manifests':True}
            zipped.writestr('export_manifest.json',json.dumps(manifest,indent=2))
        write_json(output/'export_manifest.json',manifest)
        write_json(output/'archive_hash.json',{'path':archive.name,'sha256':sha256(archive),'files':len(records)})
    except BaseException:
        write_json(output/'export_failure.json',{'status':'incomplete','archive':str(archive)})
        raise
    return {'archive':str(archive),'manifest':str(output/'export_manifest.json'),'files':len(records),'sha256':sha256(archive)}


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('--help','-h'):
        print('Kusama frozen source-connected studio\n\nCommands: doctor, ingest, index, inspire, run, resume, review, dataset, train, evaluate, export\nUse COMMAND --help for exact arguments. No command fabricates human feedback or training evidence.')
        return 0
    command,args=argv[0],argv[1:]
    if command=='doctor':
        p=argparse.ArgumentParser(prog='kusama doctor');p.add_argument('--root',type=Path,default=Path('artifacts/overnight'))
        p.add_argument('--config',type=Path,default=Path('configs/overnight.yaml'));a=p.parse_args(args)
        result=doctor(a.root,a.config)
        print(json.dumps({'report':str(a.root/'doctor.json'),'device':result['runtime']['assigned_device'],
                          'models':result['models'],'budget':result['budget']},indent=2));return 0
    if command=='ingest':
        from .corpus import main as call
        return call(args)
    if command=='index':
        from .retrieval import main as call
        return call(args)
    if command=='inspire':
        from .inspiration import main as call
        return call(args)
    if command in ('run','resume'):
        from .episodes import main as call
        return call(args)
    if command=='review':
        from .dashboard import main as call
        return call(args)
    if command=='dataset':
        from .datasets import main as call
        return call(args)
    if command=='train':
        from .learning import main as call
        return call(args)
    if command=='evaluate':
        from .evaluation import main as call
        return call(args)
    if command=='export':
        p=argparse.ArgumentParser(prog='kusama export');p.add_argument('--root',type=Path,default=Path('artifacts/overnight'))
        p.add_argument('--output',type=Path,required=True);a=p.parse_args(args)
        print(json.dumps(export_evidence(a.root,a.output),indent=2));return 0
    raise SystemExit('Unknown command: '+command)


if __name__=='__main__':raise SystemExit(main())
