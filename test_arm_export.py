import json
import pytest
import annotate
from automatic_segment import export_prediction


def candidate(arm):
    return {'trajectory_id':'case','frame_count':30,'fps':15,'motion_intervals':[[5,24]],
        'interaction_proposals':[{'start_frame':12,'end_frame':16,'skill_id':'press','arm_used':arm}]}


def test_right_candidate_is_not_relabelled_left(tmp_path,monkeypatch):
    monkeypatch.setattr(annotate,'OUT',tmp_path)
    result=export_prediction(candidate('right'))
    assert {s['arm_used'] for s in result['subtasks'] if s['skill_id']!='wait'}=={'right'}
    assert {s['arm_used'] for s in result['subtasks'] if s['skill_id']=='wait'}=={'none'}
    assert '左臂' not in json.dumps(result,ensure_ascii=False)
    assert [(s['start_frame'],s['end_frame']) for s in result['subtasks'] if s['skill_id']=='press']==[(12,16)]


def test_conflicting_visual_arm_preserves_previous(tmp_path,monkeypatch):
    monkeypatch.setattr(annotate,'OUT',tmp_path)
    proposal=candidate('right');export_prediction(proposal,stage='check')
    path=tmp_path/'automatic/check/annotations/case.json';previous=path.read_bytes()
    with pytest.raises(ValueError):export_prediction(proposal,[{'start_frame':12,'end_frame':16,'arm_used':'left'}],stage='check')
    assert path.read_bytes()==previous


def test_mixed_arms_not_flattened(tmp_path,monkeypatch):
    monkeypatch.setattr(annotate,'OUT',tmp_path)
    proposal=candidate('right');proposal['segments']=[{'skill_id':'reach','arm_used':'left'}]
    with pytest.raises(ValueError):export_prediction(proposal)


def test_all_wait_needs_no_active_arm(tmp_path,monkeypatch):
    monkeypatch.setattr(annotate,'OUT',tmp_path)
    proposal=candidate('right');proposal['motion_intervals']=[];proposal['interaction_proposals']=[]
    result=export_prediction(proposal)
    assert len(result['subtasks'])==1 and result['subtasks'][0]['arm_used']=='none'
