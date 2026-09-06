import json
import pytest
import annotate
import automatic_segment as automatic
from multiview_review import windows


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(annotate,'OUT',tmp_path)
    proposal={'trajectory_id':'episode_test','frame_count':40,'fps':15,'motion_intervals':[[5,20],[25,35]],'arm_used':'left'}
    return proposal,tmp_path


@pytest.mark.parametrize('event,reason',[
    ({'start_frame':18,'end_frame':23},'interaction_outside_motion_intervals'),
    ({'start_frame':3,'end_frame':8},'interaction_outside_motion_intervals'),
    ({'start_frame':18,'end_frame':28},'interaction_outside_motion_intervals'),
    ({'start_frame':6.0,'end_frame':9},'invalid_frame_interval'),
    ({'start_frame':6,'end_frame':9,'additional_actions':[{'skill_id':'press'}]},'unresolved_additional_actions'),
])
def test_conflict_preserves_previous(setup,event,reason):
    proposal,out=setup
    previous=out/'automatic/check/annotations/episode_test.json'
    previous.parent.mkdir(parents=True);previous.write_text('previous result',encoding='utf-8')
    with pytest.raises(ValueError):automatic.export_prediction(proposal,[event],stage='check')
    assert previous.read_text(encoding='utf-8')=='previous result'
    audit=json.loads((out/'automatic/check/export_audit/episode_test.json').read_text(encoding='utf-8'))
    assert audit['status']=='blocked' and audit['conflicts'][0]['reason']==reason


def test_later_interval_not_lost(setup):
    proposal,out=setup
    result=automatic.export_prediction(proposal,[{'start_frame':28,'end_frame':30}],stage='check')
    assert [(s['start_frame'],s['end_frame']) for s in result['subtasks'] if s['skill_id']=='press']==[(28,30)]


def test_overlap_blocks(setup):
    proposal,out=setup
    with pytest.raises(ValueError):automatic.export_prediction(proposal,[{'start_frame':6,'end_frame':10},{'start_frame':10,'end_frame':12}],stage='check')
    assert not (out/'automatic/check/annotations/episode_test.json').exists()


@pytest.mark.parametrize('n',[1,119,120,121,394,596])
def test_full_window_coverage(n):
    ws=windows(n)
    covered=set(f for start,end in ws for f in range(start,end+1))
    assert covered==set(range(n))
    assert all(b[0]<=a[1] for a,b in zip(ws,ws[1:]))
