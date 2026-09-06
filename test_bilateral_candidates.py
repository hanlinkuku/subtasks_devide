import copy
import json
import pytest
import annotate
from automatic_segment import propose


@pytest.mark.parametrize('ep',['episode_000000','episode_000001'])
def test_existing_left_candidates_unchanged(ep):
    frozen=json.loads((annotate.OUT/'automatic/motion_refined'/f'{ep}.json').read_text(encoding='utf-8'))
    assert json.loads(json.dumps(propose(ep,horizontal_withdrawal=True,persist=False)))==frozen


@pytest.mark.parametrize('ep',['episode_000000','episode_000001'])
def test_swapped_telemetry_moves_candidates_to_other_arm(ep,monkeypatch):
    original=annotate.telemetry(ep)
    left=propose(ep,horizontal_withdrawal=True,arm='left',persist=False)
    right=propose(ep,horizontal_withdrawal=True,arm='right',persist=False)
    points,rows,cols=copy.deepcopy(original)
    li=cols.index('observation.state.left_ee_pose');ri=cols.index('observation.state.right_ee_pose')
    for row in rows:row[li],row[ri]=row[ri],row[li]
    monkeypatch.setattr(annotate,'telemetry',lambda _: (points,rows,cols))
    swapped=propose(ep,horizontal_withdrawal=True,arm='right',persist=False)
    assert swapped['motion_intervals']==left['motion_intervals']
    assert [(e['start_frame'],e['end_frame']) for e in swapped['interaction_proposals']]==[(e['start_frame'],e['end_frame']) for e in left['interaction_proposals']]
    assert all(e['arm_used']=='right' for e in swapped['interaction_proposals'])
    assert not right['interaction_proposals']
    assert not propose(ep,arm='left',persist=False)['interaction_proposals']
