"""Paired real onset calls, followed by complete isolated cached replay."""
import json
import shutil
import time
from unittest.mock import patch
import requests
import annotate
import benchmark
import refine_onsets
import run_workbench_pipeline


def run():
    source=annotate.OUT;folder=source/'experiments/005_onset_scene_prior';dest=folder/'workspace'
    if dest.exists():raise ValueError('Output exists; inspect saved run before resuming')
    dest.mkdir(parents=True)
    for name in ['frames','native','telemetry','automatic/evidence','automatic/onset_evidence']:
        shutil.copytree(source/name,dest/name)
    records=[];calls=[];post=requests.post
    def counted_post(*args,**kwargs):
        started=time.perf_counter();response=post(*args,**kwargs)
        calls.append({'status':response.status_code,'elapsed_seconds':round(time.perf_counter()-started,3),
                      'model':kwargs['json']['model'],'parameters':kwargs['json']['parameters']})
        annotate.save(folder/'network_calls.json',calls)
        return response
    try:
        annotate.OUT=dest
        with patch('requests.post',side_effect=counted_post):
            for ep in ['episode_000000','episode_000001']:
                proposal=benchmark.read(source/'automatic/motion_refined'/f'{ep}.json')
                previous=benchmark.read(source/'automatic/onset_refined/decisions'/f'{ep}.json')['details']
                for i,event in enumerate(proposal['interaction_proposals']):
                    preceding=proposal['interaction_proposals'][i-1]['end_frame'] if i else -1
                    results={}
                    order=['control','variant'] if i%2==0 else ['variant','control']
                    for mode in order:
                        results[mode]=refine_onsets.refine_one(ep,event,preceding,proposal['fps'],
                            scene_prior=mode=='control',onset_cache_tag='onset_control_1' if mode=='control' else None)
                    records.append({'trajectory_id':ep,'candidate':event,'historical':previous[i],**results})
                    annotate.save(folder/'paired_results.json',records)
        with patch('requests.post',side_effect=AssertionError('Full replay unexpectedly missed cache')):
            for ep in ['episode_000000','episode_000001']:
                run_workbench_pipeline.run(ep,onset_scene_prior=False)
        comparison=benchmark.run(dest/'automatic/onset_refined/annotations')
        annotate.save(folder/'report.json',{'comparison':comparison,'actual_network_calls':len(calls),
            'complete_replay_network_calls':0,'formal_annotations_replaced':False})
        for ep in ['episode_000000','episode_000001']:
            evidence=folder/'evidence'/ep;evidence.mkdir(parents=True)
            for path in (dest/'automatic/onset_evidence'/ep).glob('*.json'):
                if not (source/'automatic/onset_evidence'/ep/path.name).exists():shutil.copy2(path,evidence/path.name)
        print('COMPLETE',[(r['trajectory_id'],r['changed_frames_from_baseline']) for r in comparison['reports']],flush=True)
    finally:annotate.OUT=source


if __name__=='__main__':run()
