"""Replay actual cached responses through the full pipeline in isolated output."""
import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch
import annotate
import benchmark
import run_workbench_pipeline


def run():
    source=annotate.OUT
    dest=source/'experiments/004_scene_prior_e2e/workspace'
    if dest.exists():raise ValueError('Experiment output already exists; inspect it instead of overwriting')
    dest.mkdir(parents=True)
    # Copy immutable visual inputs and actual response caches. No fake responses.
    for name in ['frames','native','automatic/evidence','automatic/onset_evidence']:
        shutil.copytree(source/name,dest/name)
    started=time.perf_counter()
    try:
        annotate.OUT=dest
        with patch('requests.post',side_effect=AssertionError('Offline experiment: model cache miss')):
            for ep in ['episode_000000','episode_000001']:
                run_workbench_pipeline.run(ep,scene_prior=False)
        comparison=benchmark.run(dest/'automatic/onset_refined/annotations')
        text_changes=[]
        for entry in benchmark.read(benchmark.BENCH/'manifest.json')['trajectories']:
            baseline=benchmark.read(benchmark.BENCH/entry['baseline'])
            candidate=benchmark.read(dest/'automatic/onset_refined/annotations'/f"{entry['trajectory_id']}.json")
            if len(baseline['subtasks'])!=len(candidate['subtasks']):continue
            for before,after in zip(baseline['subtasks'],candidate['subtasks']):
                for key in ['standard_action_text','notes','target_object_attribute']:
                    if before[key]!=after[key]:text_changes.append({'trajectory_id':entry['trajectory_id'],'subtask_id':after['id'],'field':key,'before':before[key],'after':after[key]})
        report={'comparison':comparison,'text_changes':text_changes,'actual_network_calls':0,
                'elapsed_pipeline_seconds':round(time.perf_counter()-started,3),'mode':'actual_response_cache_replay',
                'formal_annotation_replaced':False,'scene_prior_removed_only_in_first_visual_stage':True}
        annotate.save(dest.parent/'report.json',report)
        print(json.dumps({'text_changes':len(text_changes),'changed_frames':[r['changed_frames_from_baseline'] for r in comparison['reports']]},ensure_ascii=False))
    finally:annotate.OUT=source


if __name__=='__main__':run()
