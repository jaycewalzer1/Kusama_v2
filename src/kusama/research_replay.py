"""No-dispatch counterfactual validation of the preserved 16-step research trace.

Pointwise checks retain each step's historical incoming evidence even if a prior
step would now fail. Causal replay separately stops at the new terminal boundary.
Neither invents replacement model actions, rationales, observations or judgments.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import json
from pydantic import ValidationError
from . import research as r, research_rules as rules


class NoDispatch:
    def __getattr__(self, name):
        raise AssertionError('Replay must not dispatch sources, images or inference: ' + name)


def replay_trace(trace):
    request = r.ResearchRequest(id='trace-replay',problem=trace['problem'],authorship='recorded operator',
        project_id='replay',brief_id='replay',resource_root=Path('/not-used'),max_steps=24,fixture=True).model_dump(mode='json')
    spec={'request':request,'progress_policy':dict(r.PROGRESS_POLICY),'schema':r.ResearchAction.model_json_schema(),
          'source_config':{'providers':['met','artic','cleveland']},'snapshots':{}}
    history, causal, results = [], [], []
    terminal = None
    for saved in trace['steps']:
        state=r._state(spec,history)
        original=deepcopy(saved['action'])
        problems=[]
        try:
            envelope=r.ResearchActionEnvelope.model_validate(original)
            # This is the actual new executor, with a backend that cannot dispatch.
            r._execute_action(Path('/not-used'),spec,state,history,envelope,NoDispatch())
        except (ValueError, TypeError, KeyError) as exc:
            problems.append(str(exc))
        value=deepcopy(original)
        hyp=value.get('hypothesis')
        if hyp:
            hyp=rules.canonical_hypothesis(hyp,state['aliases'])
        # Independent shared gates expose masked failures; e.g. all 16 archived
        # actions omitted reason, which otherwise hides the later grounding bug.
        try: r.ResearchAction.model_validate(value)
        except ValidationError as exc:
            problems += [e['msg'] for e in exc.errors(include_input=False,include_url=False)]
        if hyp:
            try: rules.validate_visual_contract(hyp)
            except ValueError as exc: problems.append(str(exc))
            if value['action']=='commit':
                sources=[state['opened'][c['source_id']] for c in hyp['sources'] if c['source_id'] in state['opened']]
                try: rules.validate_reinspection(value.get('reinspection',[]),sources,saved.get('seen_images',[]))
                except ValueError as exc: problems.append(str(exc))
        problems=list(dict.fromkeys(problems))
        result={'step':saved['index']+1,'action':value['action'],'status':'rejected' if problems else 'accepted',
                'reasons':problems,'original_step_sha256':saved['original_step_sha256'],
                'original_raw_completion_sha256':rules.digest(saved.get('raw_completion'))}
        results.append(result)
        if terminal is None:
            cstate=r._state(spec,causal)
            try:
                row=r._execute_action(Path('/not-used'),spec,cstate,causal,r.ResearchActionEnvelope.model_validate(original),NoDispatch())
            except (ValueError, TypeError, KeyError) as exc:
                row={'index':len(causal),'action':original,'tool_error':str(exc),'status':'tool_rejected'}
            causal.append(row)
            terminal=r._terminal_status(spec,causal)
        historical=deepcopy(saved)
        historical['opened_sources']=[deepcopy(trace['sources'][sid]) for sid in saved['opened_source_ids']]
        if value['action']=='acquire' and historical['opened_sources']:
            historical['canonical_aliases']={value['candidate_id']:historical['opened_sources'][0]['source_id']}
        if historical.get('original_commit'):
            historical['committed_hypothesis']=deepcopy(hyp)
        history.append(historical)
    expected={3,5,8,9,11,12,13,15,16}
    rejected={s['step'] for s in results if s['status']=='rejected'}
    guards=expected <= rejected and any('relevance gate' in s for s in results[2]['reasons']) and any('grounding' in s or 'entailment' in s for s in results[15]['reasons'])
    return {'result':'PASS' if guards else 'FAIL', 'kind':'software_replay_regression',
        'recorded_run_acceptance':'FAIL', 'run_id':trace['run_id'],'expected_rejections':sorted(expected),
        'steps':results,'causal_replay':{'steps_executed':len(causal),'terminal_status':terminal,'replacement_actions':0},
        'boundary':'PASS means required software rejections occurred. It does not mean the artistic run passed. Pointwise checks retain original incoming evidence; causal replay stops independently. All missing rationales remain missing.',
        'inference_calls':0,'network_requests':0,'render_calls':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=replay_trace(json.loads(args.trace.read_text()))
    raw=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(raw)
    print(raw)
    return 0 if report['result']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
