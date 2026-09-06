import pytest
from fact_review import validate_facts,interpret


def fact(kind,visibility='clear'):
    return {'id':'F1','kind':kind,'visibility':visibility,'change_present':True,'view':'left_wrist_rgb','start_frame':10,'end_frame':11,'description':'observed change'}


@pytest.mark.parametrize('kind',['screen_change','relative_motion','occlusion','local_tip_motion'])
def test_single_cue_not_new_press(kind):
    result=interpret([fact(kind)])
    assert result['status']=='insufficient_evidence'
    assert result['new_action_confirmed'] is False


def test_occluded_deformation_does_not_pass():
    assert interpret([fact('visible_button_deformation','occluded')])['status']=='insufficient_evidence'


def test_combined_cues_are_hypothesis_not_truth():
    result=interpret([fact('screen_change'),dict(fact('local_tip_motion'),id='F2')])
    assert result['status']=='interaction_hypothesis'
    assert result['boundary_replacement_allowed'] is False
    assert result['fact_ids']==['F2','F1']


def test_unseen_frame_rejected():
    with pytest.raises(ValueError):validate_facts({'observations':[fact('screen_change')]},[10,12])


def test_unknown_view_rejected():
    with pytest.raises(ValueError):validate_facts({'observations':[dict(fact('screen_change'),view='invented')]},[10,11])


def test_contact_claim_is_not_direct_fact():
    raw={'observations':[dict(fact('visible_button_deformation'),description='夹爪持续接触按钮并施加压力')]}
    observations=validate_facts(raw,[10,11])
    assert observations[0]['claim_flags']
    assert interpret(observations)['status']=='insufficient_evidence'


def test_no_change_does_not_create_interaction():
    assert interpret([dict(fact('screen_change'),change_present=False),fact('local_tip_motion')])['status']=='insufficient_evidence'


def test_conflicting_change_description_is_flagged():
    observations=validate_facts({'observations':[dict(fact('screen_change'),description='屏幕内容保持不变')]},[10,11])
    assert 'change_claim_conflicts_with_description' in observations[0]['claim_flags']
    assert interpret(observations+[fact('local_tip_motion')])['status']=='insufficient_evidence'


def test_legacy_fact_not_implicitly_positive():
    old=fact('visible_button_deformation');del old['change_present']
    observations=validate_facts({'observations':[old]},[10,11])
    assert interpret(observations)['status']=='insufficient_evidence'
