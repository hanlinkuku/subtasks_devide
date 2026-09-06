import copy
import pytest
from benchmark import read,BENCH,compare,partition,run,edit_distance


def example():return read(BENCH/'baseline/episode_000001.json')


def test_baseline_self_comparison():
    result=run(BENCH/'baseline')
    assert all(r['changed_frames_from_baseline']==0 for r in result['reports'])
    assert result['promotion_status']=='not_evaluated'


def test_boundary_regression_is_visible():
    base=example();candidate=copy.deepcopy(base)
    candidate['subtasks'][1]['end_frame']+=2
    candidate['subtasks'][2]['start_frame']+=2
    result=compare(base,candidate,base)
    assert result['changed_frames_from_baseline']==2
    assert result['boundaries'][1]['change']=='farther_from_reference'


def test_extra_action_disables_positional_boundary_matching():
    base=example();candidate=copy.deepcopy(base)
    part=candidate['subtasks'][1];extra=copy.deepcopy(part)
    extra['start_frame']=part['end_frame'];part['end_frame']-=1
    extra['skill_id']='press';candidate['subtasks'].insert(2,extra)
    result=compare(base,candidate,base)
    assert result['candidate_sequence_edit_distance']==1
    assert result['boundary_comparison_available'] is False
    assert not result['boundaries']


def test_gap_rejected():
    candidate=example();candidate['subtasks'][1]['start_frame']+=1
    with pytest.raises(ValueError):partition(candidate)


def test_identity_rejected():
    base=example();candidate=copy.deepcopy(base);candidate['trajectory_id']='wrong'
    with pytest.raises(ValueError):compare(base,candidate,base)


def test_sequence_substitution_and_omission():
    assert edit_distance(['reach','press','move'],['reach','move'])==1
    assert edit_distance(['reach','press'],['reach','grasp'])==1
