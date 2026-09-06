from inspect_static_display import usable_reading,summarize


def record(frame,value,readability='clear'):
    return {'frame':frame,'result':{'display_visible':True,'numeric_reading':value,'readability':readability}}


def test_partial_numeric_reading_is_not_used():
    assert usable_reading(record(1,'18.0','partial')['result']) is None


def test_unknown_gap_cannot_make_transition():
    result=summarize([record(1,'18.0'),record(2,'19.0','partial'),record(3,'18.0')])
    assert not result['clear_adjacent_numeric_transitions']
    assert result['excluded_unclear_frames']==[2]


def test_unsampled_gap_cannot_make_precise_transition():
    assert not summarize([record(1,'18.0'),record(3,'19.0')])['clear_adjacent_numeric_transitions']


def test_adjacent_clear_change_is_only_model_evidence():
    result=summarize([record(1,'18.0'),record(2,'18.5')])
    assert result['clear_adjacent_numeric_transitions'][0]['after_frame']==2


def test_missing_decimal_is_format_conflict_not_numeric_change():
    result=summarize([record(43,'18.0'),record(44,'180')])
    assert not result['numeric_change_corroborated_by_model_readings']
    assert result['reading_format_conflicts'][0]['after_frame']==44
