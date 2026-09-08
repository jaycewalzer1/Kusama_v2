"""Execute front-end polling behavior against controlled DOM/HTTP fixtures."""
from pathlib import Path
from html.parser import HTMLParser
import shutil
import subprocess

import pytest


def test_live_progress_preserves_review_drafts_and_selection():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for the dashboard JavaScript behavior test')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, str(root / 'tests' / 'dashboard_live_check.cjs')],
                            cwd=root, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_allocated_reviewer_identity_is_explicit_and_neutral():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for the dashboard JavaScript behavior test')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, str(root / 'tests' / 'dashboard_review_identity_check.cjs')],
                            cwd=root, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_reviewer_controls_have_explicit_unambiguous_labels():
    class Markup(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = {}
            self.labels = {}
            self.current_label = None

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if 'id' in attrs:
                assert attrs['id'] not in self.ids, 'Control IDs must be unique'
                self.ids[attrs['id']] = (tag, attrs)
            if tag == 'label':
                self.current_label = attrs.get('for')
                if self.current_label:
                    assert self.current_label not in self.labels
                    self.labels[self.current_label] = ''

        def handle_data(self, data):
            if self.current_label:
                self.labels[self.current_label] += data

        def handle_endtag(self, tag):
            if tag == 'label':
                self.current_label = None

    markup = Markup()
    markup.feed((Path(__file__).resolve().parents[1] / 'src/kusama/dashboard_static/index.html').read_text())
    for control, tag, label in [('reviewer', 'input', 'Your reviewer name'),
                                ('reviewer-slot', 'select', 'Your allocated reviewer slot')]:
        assert markup.ids[control][0] == tag
        assert markup.labels[control] == label, 'Labels must not include select option text'
        assert markup.ids[control][1]['aria-describedby'] == 'reviewer-guidance'
    assert 'reviewer-guidance' in markup.ids
    assert 'hidden' in markup.ids['reviewer-slot-field'][1]
