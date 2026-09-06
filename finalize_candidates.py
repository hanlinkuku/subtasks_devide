"""Export reviewed candidates, with a hard block on unverified exact boundaries.

These historical sample adjudications are not model-only results. Subsequent
user-reviewed reference snapshots live separately in outputs/confirmed.
"""
import json
from pathlib import Path
import numpy as np
import annotate

OUT=annotate.OUT

# start, skill, action, uncertainty interval for this start boundary.
# First/last frames follow the original video; task/quality labels are not used as
# ground-truth atomic boundaries. Intervals record visual/telemetry ambiguity.
REVIEWS={
    'episode_000000': [
        (0,'wait','机械臂保持初始位置',(0,0)),
        (12,'reach','左臂接近温控面板的电源按钮',(12,15)),
        (76,'press','左臂按压温控面板的电源按钮',(76,80)),
        (91,'move','左臂离开电源按钮并移向升温按钮',(89,95)),
        (152,'press','左臂按压升温按钮',(139,154)),
        (176,'move','左臂从温控面板撤回初始位置',(174,179)),
        (284,'wait','机械臂在初始位置等待',(279,287)),
        (331,'reach','左臂再次接近温控面板的电源按钮',(331,336)),
        (438,'press','左臂再次按压温控面板的电源按钮',(436,441)),
        (451,'move','左臂从温控面板撤回初始位置',(448,454)),
        (574,'wait','机械臂在初始位置保持静止',(568,574)),
    ],
    'episode_000001': [
        (0,'wait','机械臂保持初始位置',(0,0)),
        (12,'reach','左臂接近温控面板的电源按钮',(12,15)),
        (118,'press','左臂按压温控面板的电源按钮',(114,125)),
        (153,'move','左臂离开电源按钮并移向降温按钮',(150,155)),
        (206,'press','左臂按压降温按钮',(204,209)),
        (221,'move','左臂从温控面板撤回初始位置',(218,224)),
        (364,'wait','机械臂在初始位置保持静止',(358,364)),
    ],
}


def material_object(active):
    if not active:
        return {'name':'','colors':[],'shapes':[],'materials':[],'sizes':[],'others':[],'description':''}
    return {'name':'温控面板','colors':['白色'],'shapes':['矩形'],'materials':[],
            'sizes':[],'others':['安装在墙面','带显示屏和右侧按钮'],
            'description':'安装在墙面的白色矩形温控面板，显示屏右侧为按钮区域。'}


def signal_audit(ep):
    xyz,rows,cols=annotate.telemetry(ep)
    n=len(rows)
    indices=[row[cols.index('frame_index')] for row in rows]
    timestamps=[row[cols.index('timestamp')] for row in rows]
    assert indices==list(range(n)), 'Source frame indices are not continuous'
    assert np.allclose(timestamps,np.arange(n)/15,atol=5e-6,rtol=0), 'Timestamp mismatch'
    for view in annotate.VIEWS:
        frames=OUT/'frames'/ep/view
        assert sorted(p.name for p in frames.glob('*.jpg'))==[f'{i:06d}.jpg' for i in range(n)]
    conflicts=[]
    excluded=[]
    for path in sorted((OUT/'evidence'/ep).glob('*.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('model')!='qwen3-vl-32b-instruct':
            excluded.append(path.name);continue
        result=data.get('result',{})
        for seg in result.get('segments',result.get('actions',[])):
            s,e=seg.get('start_frame'),seg.get('end_frame')
            if not isinstance(s,int) or not isinstance(e,int) or not 0<=s<=e<n:
                conflicts.append({'file':path.name,'reason':'invalid_frame_interval','segment':seg});continue
            if seg.get('skill_id')=='wait':
                points=xyz[s:e+1]*1000
                excursion=float(np.max(np.linalg.norm(points-points[0],axis=1)))
                if excursion>10:
                    conflicts.append({'file':path.name,'start_frame':s,'end_frame':e,
                        'reason':'wait_conflicts_with_left_end_effector_motion',
                        'excursion_mm':round(excursion,3)})
    return xyz,rows,cols,conflicts,excluded


def export(ep):
    if (OUT/'confirmed'/'human_review.json').exists():
        raise RuntimeError('Historical candidate export is frozen after human review. Run automatic_segment.py --visual for independent predictions.')
    xyz,rows,cols,conflicts,excluded=signal_audit(ep)
    n=len(rows); entries=REVIEWS[ep]; subtasks=[]; unresolved=[]
    for i,(start,skill,action,bounds) in enumerate(entries):
        end=entries[i+1][0]-1 if i+1<len(entries) else n-1
        end_bounds=[v-1 for v in entries[i+1][3]] if i+1<len(entries) else [end,end]
        note='候选标注，尚未通过语义和帧级边界验收。'
        if bounds[0]!=bounds[1]:note+=f'起始边界待核范围为{bounds[0]}—{bounds[1]}帧。'
        if end_bounds[0]!=end_bounds[1]:note+=f'结束边界待核范围为{end_bounds[0]}—{end_bounds[1]}帧。'
        if skill=='press':note+='按钮区域受到夹爪遮挡；画面响应与末端运动支持存在按钮操作，但不能单独证明物理接触的精确帧或连续按压次数。'
        elif skill in ['reach','move']:note+='依据目标相对腕部相机的位移及左末端位置变化判断运动；右臂未见相应操作。'
        else:note+='此处为初始、动作间或末尾停留；微小反馈抖动不拆为独立动作。'
        subtasks.append({'id':i+1,'standard_action_text':action+'。','skill_id':skill,
            'arm_used':'none' if skill=='wait' else 'left','notes':note,
            'start_frame':start,'end_frame':end,'start_time':round(start/15,6),
            'end_time':round(end/15,6),'target_object_attribute':material_object(skill!='wait')})
        if bounds[0]!=bounds[1]:
            unresolved.append({'subtask_id':i+1,'boundary':'start','candidate_frame':start,
                'earliest_frame':bounds[0],'latest_frame':bounds[1],
                'reason':'接触遮挡或运动由微动逐渐过渡，缺少独立参考标注，不能确认单帧边界。'})
    obj={'trajectory_id':ep,'trajectory_start':0,'trajectory_end':n-1,
         'task_name':'用左臂操作温控面板并调节温度','subtasks':subtasks}
    # Structural checks are deliberately separate from unmeasured semantic quality.
    assert list(obj)==['trajectory_id','trajectory_start','trajectory_end','task_name','subtasks']
    cursor=0
    for s in subtasks:
        assert s['start_frame']==cursor and s['end_frame']>=cursor
        cursor=s['end_frame']+1
    assert cursor==n
    episodes=[json.loads(x) for x in (annotate.DATA/'meta'/'episodes.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    original=next(x for x in episodes if x['episode_index']==rows[0][cols.index('episode_index')])
    annotate.save(OUT/'candidates'/f'{ep}.json',obj)
    audit={'trajectory_id':ep,'source_dataset':str(annotate.DATA),'source_episode':original,
        'model':'qwen3-vl-32b-instruct','adjudication':'assistant_visual_and_telemetry_review',
        'model_only_output':False,'sample_specific_adjudication':True,
        'frame_count':n,'fps':15,'frame_interval_convention':'inclusive',
        'time_convention':'timestamp of indexed frame; not exclusive clip-end time',
        'structural_checks':{'all_four_views_complete':True,'source_indices_match':True,
            'timestamps_match':True,'no_gap_or_overlap':True},
        'quality_gate':{'accepted':False,'semantic_precision':None,'semantic_recall':None,
            'boundary_error_frames':None,'reason':'No independent ground truth; physical contact boundaries unresolved'},
        'unresolved_boundaries':unresolved,'rejected_model_proposals':conflicts,
        'excluded_non_32b_evidence':excluded}
    annotate.save(OUT/'quality'/f'{ep}.json',audit)
    print(ep,'candidate_subtasks',len(subtasks),'unresolved_boundaries',len(unresolved),
          'rejected_proposals',len(conflicts),'ACCEPTED=False')
    return audit


if __name__=='__main__':
    reports=[export(ep) for ep in REVIEWS]
    annotate.save(OUT/'quality'/'status.json',{'accepted':False,
        'candidate_trajectories':len(reports),'accepted_trajectories':0,
        'models_used_for_candidates':['qwen3-vl-32b-instruct'],
        'remaining_work':['Resolve physical contact boundaries against independent reference labels or additional contact evidence',
                          'Measure semantic precision/recall and boundary errors; structural coverage is not semantic completeness']})
