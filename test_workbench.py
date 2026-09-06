import copy
import json

import pytest
from fastapi.testclient import TestClient
import workbench as wb


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(wb, 'MANUAL', tmp_path / 'manual')
    return TestClient(wb.app)


def test_save_roundtrip_conflict_and_history(client):
    ep = 'episode_000001'
    initial = client.get(f'/api/episodes/{ep}/annotation').json()
    obj = copy.deepcopy(initial['annotation'])
    obj['subtasks'][1]['end_frame'] += 1
    obj['subtasks'][2]['start_frame'] += 1
    obj['subtasks'][2]['start_time'] = -100
    result = client.put(f'/api/episodes/{ep}/annotation', json={'annotation': obj, 'revision': None})
    assert result.status_code == 200
    saved = result.json()
    assert saved['annotation']['subtasks'][2]['start_time'] == round(obj['subtasks'][2]['start_frame'] / 15, 6)
    loaded = client.get(f'/api/episodes/{ep}/annotation').json()
    assert loaded['source'] == 'manual'
    assert loaded['annotation'] == saved['annotation']
    assert len(list((wb.MANUAL / 'history' / ep).glob('*.json'))) == 1
    assert client.put(f'/api/episodes/{ep}/annotation', json={'annotation': obj}).status_code == 409
    assert client.get(f'/api/episodes/{ep}/annotation?source=automatic').json()['annotation'] == initial['annotation']


@pytest.mark.parametrize('kind', ['gap', 'overlap', 'float', 'end', 'skill'])
def test_reject_invalid_partition(client, kind):
    ep = 'episode_000000'
    obj = client.get(f'/api/episodes/{ep}/annotation').json()['annotation']
    if kind == 'gap': obj['subtasks'][1]['start_frame'] += 1
    if kind == 'overlap': obj['subtasks'][1]['start_frame'] -= 1
    if kind == 'float': obj['subtasks'][1]['start_frame'] = 12.0
    if kind == 'end': obj['subtasks'][-1]['end_frame'] -= 1
    if kind == 'skill': obj['subtasks'][1]['skill_id'] = 'invalid'
    assert client.put(f'/api/episodes/{ep}/annotation', json={'annotation': obj}).status_code == 422
    assert not wb.MANUAL.exists()


def test_frame_bounds_and_all_views(client):
    for meta in client.get('/api/episodes').json():
        for view in wb.annotate.VIEWS:
            r = client.get(f'/api/episodes/{meta["id"]}/frames/{view}/{meta["frames"]-1}')
            assert r.status_code == 200 and r.headers['content-type'] == 'image/jpeg'
        assert client.get(f'/api/episodes/{meta["id"]}/frames/head_rgb/{meta["frames"]}').status_code == 404
    assert client.get('/api/episodes/invalid/annotation').status_code == 404


def test_job_key_not_returned(client, monkeypatch):
    received = []
    monkeypatch.setattr(wb, 'run_job', lambda *args: received.append(args))
    monkeypatch.setattr(wb, 'jobs', {})
    response = client.post('/api/jobs', json={'episode': 'episode_000001', 'api_key': 'test-secret'})
    assert response.status_code == 200
    assert 'test-secret' not in client.get('/api/jobs/current').text
    assert client.post('/api/jobs', json={'episode': 'episode_000000'}).status_code == 409
