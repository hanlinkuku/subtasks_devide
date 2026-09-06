"""Normalize user-supplied labels without resolving ambiguous frame edits."""
import copy
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'outputs'/'ground_truth'/'v1'


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    dataset=next(ROOT.glob('*/Turn_On_And_Off_The_Air_Conditioner_raw_lerobot'))
    fps=json.loads((dataset/'meta'/'info.json').read_text(encoding='utf-8'))['fps']
    lengths={f'episode_{row["episode_index"]:06d}':row['length'] for row in
        (json.loads(line) for line in (dataset/'meta'/'episodes.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())}
    entries=[]
    for path in sorted((BASE/'source').glob('episode_*.json')):
        source=json.loads(path.read_text(encoding='utf-8'));obj=copy.deepcopy(source)
        ep=obj['trajectory_id'];issues=[];changes=[];boundary_uncertainties=[]
        if obj['trajectory_start']!=0 or obj['trajectory_end']!=lengths[ep]-1:
            issues.append({'reason':'trajectory_range_mismatch'})
        prior=None
        for i,seg in enumerate(obj['subtasks']):
            s,e=seg['start_frame'],seg['end_frame']
            if type(s)!=int or type(e)!=int or not 0<=s<=e<lengths[ep]:
                issues.append({'subtask_id':seg['id'],'reason':'invalid_interval'})
                continue
            if seg['id']!=i+1:issues.append({'subtask_id':seg['id'],'reason':'invalid_id_order'})
            expected=prior['end_frame']+1 if prior else 0
            if s!=expected:
                issues.append({'reason':'overlap' if s<expected else 'gap',
                    'previous_subtask_id':prior['id'] if prior else None,'subtask_id':seg['id'],
                    'previous_end':expected-1,'current_start':s,
                    'affected_frames':[min(s,expected),max(s,expected)-1]})
            if prior and 'press' in [prior['skill_id'],seg['skill_id']]:
                boundary_uncertainties.append({'from_subtask_id':prior['id'],'to_subtask_id':seg['id'],
                    'from_skill':prior['skill_id'],'to_skill':seg['skill_id'],
                    'reference_frame':s,'uncertainty_frames':None,
                    'basis':'User reports difficulty locating press transitions to an exact frame; no numeric tolerance specified'})
            for field,frame in [('start_time',s),('end_time',e)]:
                value=round(frame/fps,6)
                if seg[field]!=value:changes.append({'subtask_id':seg['id'],'field':field,'original':seg[field],'normalized':value})
                seg[field]=value
            seg['notes']=seg['notes'].replace('经人工校对，当前观察下子任务划分及起止帧确认正确。',
                '用户手工标注；按压相关边界可能存在帧级偏差。')
            prior=seg
        if not prior or prior['end_frame']!=lengths[ep]-1:issues.append({'reason':'uncovered_tail'})
        out=BASE/f'{ep}.json';save(out,obj)
        entry={'trajectory_id':ep,'file':out.relative_to(ROOT).as_posix(),
            'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'ready_for_comparison':not issues,'structural_issues':issues,
            'time_corrections':changes,'uncertain_boundaries':boundary_uncertainties}
        entries.append(entry)
        print(ep,'time_corrections',len(changes),'issues',json.dumps(issues,ensure_ascii=False))
    save(BASE/'manifest.json',{'version':'v1','imported_at':datetime.now(timezone.utc).isoformat(),
        'source':'user manually annotated attachment and inline message',
        'user_caveat':'在reach到press和press到move的划分难以精确到帧，可能会有偏差',
        'reference_role':'human_reference_with_boundary_uncertainty','exact_frame_truth_claimed':False,
        'frame_authority':'user-supplied frame indices; times recomputed from source FPS',
        'fps':fps,'frame_interval_convention':'inclusive','boundary_tolerance_frames':None,
        'trajectories':entries})


if __name__=='__main__':main()
