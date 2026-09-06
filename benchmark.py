"""Offline comparison against an immutable baseline; never used by inference."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BENCH=ROOT/'benchmarks/v1'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def partition(doc):
    cursor=doc['trajectory_start'];labels=[]
    for item in doc['subtasks']:
        a,b=item['start_frame'],item['end_frame']
        if type(a)!=int or type(b)!=int or a!=cursor or b<a:
            raise ValueError('Invalid partition: gaps, overlaps or invalid frames')
        labels.extend([(item['skill_id'],item['arm_used'])]*(b-a+1));cursor=b+1
    if cursor!=doc['trajectory_end']+1:raise ValueError('Incomplete partition')
    return labels


def sequence(doc):return [(s['skill_id'],s['arm_used']) for s in doc['subtasks']]


def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        next_row=[i]
        for j,y in enumerate(b,1):next_row.append(min(row[j]+1,next_row[-1]+1,row[j-1]+(x!=y)))
        row=next_row
    return row[-1]


def compare(baseline,candidate,reference):
    if any((d['trajectory_id'],d['trajectory_start'],d['trajectory_end']) !=
           (reference['trajectory_id'],reference['trajectory_start'],reference['trajectory_end']) for d in [baseline,candidate]):
        raise ValueError('Trajectory identity or range mismatch')
    bl,cl,rl=map(partition,[baseline,candidate,reference])
    aligned=sequence(candidate)==sequence(reference)
    boundaries=[]
    if aligned and sequence(baseline)==sequence(reference):
        for i,(b,c,r) in enumerate(zip(baseline['subtasks'][1:],candidate['subtasks'][1:],reference['subtasks'][1:]),2):
            bd=abs(b['start_frame']-r['start_frame']);cd=abs(c['start_frame']-r['start_frame'])
            boundaries.append({'next_subtask':i,'reference':r['start_frame'],'baseline':b['start_frame'],
                'candidate':c['start_frame'],'baseline_deviation':bd,'candidate_deviation':cd,
                'change':'closer_to_reference' if cd<bd else 'farther_from_reference' if cd>bd else 'unchanged',
                'exact_truth_confirmed':False})
    return {'trajectory_id':reference['trajectory_id'],'baseline_subtasks':len(baseline['subtasks']),
        'candidate_subtasks':len(candidate['subtasks']),'reference_subtasks':len(reference['subtasks']),
        'baseline_sequence_edit_distance':edit_distance(sequence(baseline),sequence(reference)),
        'candidate_sequence_edit_distance':edit_distance(sequence(candidate),sequence(reference)),
        'baseline_frame_label_agreement':sum(a==b for a,b in zip(bl,rl))/len(rl),
        'candidate_frame_label_agreement':sum(a==b for a,b in zip(cl,rl))/len(rl),
        'boundary_comparison_available':aligned and sequence(baseline)==sequence(reference),
        'boundaries':boundaries,'changed_frames_from_baseline':sum(a!=b for a,b in zip(bl,cl)),
        'changed_annotation_from_baseline':candidate!=baseline}


def run(candidate_dir):
    manifest=read(BENCH/'manifest.json');reports=[]
    for item in manifest['trajectories']:
        for kind in ['baseline','reference']:
            if digest(BENCH/item[kind])!=item[kind+'_sha256']:raise ValueError('Frozen benchmark checksum mismatch')
        candidate_path=candidate_dir/(item['trajectory_id']+'.json')
        result=compare(read(BENCH/item['baseline']),read(candidate_path),read(BENCH/item['reference']))
        result['candidate_sha256']=digest(candidate_path);reports.append(result)
    return {'benchmark_version':manifest['version'],'reference_role':'working_reference_not_exact_frame_truth',
        'inference_performed':False,'reports':reports,'promotion_status':'not_evaluated',
        'reason':'Promotion requires a declared confirmed target case and review of regressions; reference distance alone cannot establish improvement.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--candidate-dir',type=Path,default=ROOT/'outputs/automatic/onset_refined/annotations')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run(args.candidate_dir);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
