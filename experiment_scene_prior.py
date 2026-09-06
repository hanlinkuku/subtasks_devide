"""Single-variable scene-prior ablation; never replaces formal annotations."""
import json
import time
import annotate
from automatic_segment import refine_event


def run(ep):
    proposal=json.loads((annotate.OUT/'automatic/kinematic'/f'{ep}.json').read_text(encoding='utf-8'))
    baseline=json.loads((annotate.OUT/'automatic/visual'/f'{ep}.json').read_text(encoding='utf-8'))['events']
    rows=[]
    for old,event in zip(baseline,proposal['interaction_proposals'],strict=True):
        started=time.perf_counter()
        new=refine_event(ep,event,proposal['fps'],proposal['frame_count'],scene_prior=False)
        variant_seconds=round(time.perf_counter()-started,3)
        started=time.perf_counter()
        control=refine_event(ep,event,proposal['fps'],proposal['frame_count'],cache_tag='scene_prior_control_1')
        rows.append({'candidate':event,'baseline':old,'without_scene_prior':new,
            'fresh_control':control,'variant_elapsed_seconds':variant_seconds,'control_elapsed_seconds':round(time.perf_counter()-started,3),
            'support_unchanged':old.get('interaction_supported')==new.get('interaction_supported'),
            'boundary_unchanged':[old.get('start_frame'),old.get('end_frame')]==[new.get('start_frame'),new.get('end_frame')]})
        annotate.save(annotate.OUT/'experiments/003_scene_prior'/f'{ep}.json',{'trajectory_id':ep,'results':rows,
            'automatic_annotation_replaced':False,'reference_used_during_inference':False,'complete':len(rows)==len(baseline)})
    return rows


if __name__=='__main__':
    for ep in ['episode_000000','episode_000001']:run(ep)
