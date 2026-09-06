"""Compare predictions to the selected working reference, not exact contact truth."""
import hashlib
import json
from pathlib import Path

import numpy as np
import annotate


def compare(stage):
    pointer=json.loads((annotate.OUT/'ground_truth'/'current.json').read_text(encoding='utf-8'))
    review=json.loads((annotate.ROOT/pointer['manifest']).read_text(encoding='utf-8'))
    reports=[];all_errors=[];correct=0;total=0
    groups={'into_press':[],'out_of_press':[],'other':[]}
    for entry in review['trajectories']:
        if not entry['ready_for_comparison']:raise ValueError('Reference has unresolved structural issues')
        ep=entry['trajectory_id'];ref_path=annotate.ROOT/entry['file']
        if hashlib.sha256(ref_path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Working reference checksum mismatch')
        pred_path=annotate.OUT/'automatic'/stage/'annotations'/f'{ep}.json'
        ref=json.loads(ref_path.read_text(encoding='utf-8'));pred=json.loads(pred_path.read_text(encoding='utf-8'))
        gt=ref['subtasks'];actual=pred['subtasks'];n=ref['trajectory_end']+1
        aligned=[(s['skill_id'],s['arm_used']) for s in gt]==[(s['skill_id'],s['arm_used']) for s in actual]
        errors=[]
        if aligned:
            for g,p in zip(gt[1:],actual[1:]):
                delta=p['start_frame']-g['start_frame'];all_errors.append(abs(delta))
                previous=gt[g['id']-2]['skill_id']
                group='into_press' if g['skill_id']=='press' else ('out_of_press' if previous=='press' else 'other')
                groups[group].append(abs(delta))
                errors.append({'subtask_id':g['id'],'skill_id':g['skill_id'],
                    'reference_frame':g['start_frame'],'automatic_frame':p['start_frame'],
                    'signed_error_frames':delta,'absolute_error_frames':abs(delta)})
        glabels=[None]*n;plabels=[None]*n
        for seg in gt:
            for frame in range(seg['start_frame'],seg['end_frame']+1):
                if not 0<=frame<n or glabels[frame] is not None:raise ValueError('Invalid reference partition')
                glabels[frame]=(seg['skill_id'],seg['arm_used'])
        if None in glabels:raise ValueError('Reference has gaps')
        for seg in actual:
            for frame in range(seg['start_frame'],seg['end_frame']+1):
                if not 0<=frame<n or plabels[frame] is not None:raise ValueError('Invalid prediction partition')
                plabels[frame]=(seg['skill_id'],seg['arm_used'])
        if None in plabels:raise ValueError('Prediction has gaps')
        hits=sum(a==b for a,b in zip(glabels,plabels));correct+=hits;total+=n
        reports.append({'trajectory_id':ep,'reference_subtasks':len(gt),'automatic_subtasks':len(actual),
            'skill_and_arm_sequence_match':aligned,'boundary_errors':errors,
            'frame_skill_and_arm_accuracy':hits/n,
            'prediction_sha256':hashlib.sha256(pred_path.read_bytes()).hexdigest()})
    result={'stage':stage,'scope':'deviation from the selected working reference; not exact physical-contact error or held-out generalization',
        'reference_manifest':pointer['manifest'],'exact_frame_truth_claimed':False,
        'boundary_tolerance_frames':review.get('boundary_tolerance_frames'),
        'tolerance_counts_are_descriptive_not_acceptance':True,
        'reference_used_during_inference':False,'quantitative_metrics_do_not_cover_action_text_or_object_attributes':True,
        'boundary_count':len(all_errors),'exact_boundaries':sum(x==0 for x in all_errors),
        'boundaries_within_1_frame':sum(x<=1 for x in all_errors),
        'boundaries_within_3_frames':sum(x<=3 for x in all_errors),
        'boundaries_within_5_frames':sum(x<=5 for x in all_errors),
        'deviation_by_transition':{name:{'count':len(values),'mean_absolute_deviation_frames':float(np.mean(values)) if values else None,
            'max_absolute_deviation_frames':max(values,default=None)} for name,values in groups.items()},
        'mean_absolute_boundary_error_frames':float(np.mean(all_errors)) if all_errors else None,
        'max_absolute_boundary_error_frames':max(all_errors,default=None),
        'frame_skill_and_arm_accuracy':correct/total,'trajectories':reports,
        'accepted':False,
        'matches_reference_partition':bool(all_errors) and all(x==0 for x in all_errors) and all(x['skill_and_arm_sequence_match'] for x in reports)}
    annotate.save(annotate.OUT/'automatic'/'comparison'/review['version']/f'{stage}.json',result)
    print(stage,json.dumps({k:v for k,v in result.items() if k!='trajectories'},ensure_ascii=False))
    return result


if __name__=='__main__':
    for stage in ['kinematic','vlm_refined','motion_refined','onset_refined']:compare(stage)
