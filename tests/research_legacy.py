"""Freeze pre-v13 fixture behavior to retain earlier recovery regression coverage.

New plans always enable grounded_search. V13 end-to-end and rule tests live in
 test_research_grounded.py; these historical scripted actors do not author its
new question analysis, citations or reinspection fields.
"""
from kusama import research


def use_legacy_protocol(monkeypatch):
    monkeypatch.setattr(research, 'PROGRESS_POLICY', {k: v for k, v in research.PROGRESS_POLICY.items() if k != 'grounded_search'})
    monkeypatch.setattr(research, 'VERSION', 'independent-research-v12-image-prompts')
    monkeypatch.setattr(research.ResearchAction, 'model_json_schema', classmethod(lambda cls, **kw: _schema_without_reason(cls, **kw)))


def _schema_without_reason(cls, **kwargs):
    from pydantic import BaseModel
    schema = BaseModel.model_json_schema.__func__(cls, **kwargs)
    schema['required'] = [k for k in schema.get('required', []) if k != 'reason']
    return schema
