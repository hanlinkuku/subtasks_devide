"""Fixed fact extraction then evidence-constrained interpretation; no agent."""
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import annotate
from multiview_review import ask

TYPES={'relative_motion','screen_change','local_tip_motion','visible_button_deformation','visible_object_release','occlusion','no_clear_change','unclear'}
INFERRED_CONTACT_PHRASES=('持续接触','保持接触','施加压力','按压成功','物理接触','保持按压')


def validate_facts(result,frames):
    observations=result.get('observations')
    if not isinstance(observations,list):raise ValueError('Missing observations')
    for i,obs in enumerate(observations):
        if obs.get('view') not in annotate.VIEWS or obs.get('kind') not in TYPES:raise ValueError('Invalid fact type or view')
        a,b=obs.get('start_frame'),obs.get('end_frame')
        if type(a)!=int or type(b)!=int or a not in frames or b not in frames or a>b:raise ValueError('Fact cites unseen frames')
        if obs.get('visibility') not in {'clear','occluded','unclear'} or not isinstance(obs.get('description'),str):raise ValueError('Invalid fact evidence')
        obs['id']=f'F{i+1}'
        obs['claim_flags']=['contact_or_action_inference_in_description'] if any(text in obs['description'] for text in INFERRED_CONTACT_PHRASES) else []
    return observations


def interpret(observations):
    """Evidence sufficiency gate, not a contact classifier or boundary estimator."""
    clear=[o for o in observations if o['visibility']=='clear' and not o.get('claim_flags')]
    deformation=[o['id'] for o in clear if o['kind']=='visible_button_deformation']
    release=[o['id'] for o in clear if o['kind']=='visible_object_release']
    tip=[o['id'] for o in clear if o['kind']=='local_tip_motion']
    response=[o['id'] for o in clear if o['kind']=='screen_change']
    if deformation:
        status='interaction_candidate';reason='有模型报告的可见按钮形变，仍需判断是否独立新动作及时间范围。';refs=deformation
    elif release:
        status='object_release_candidate';reason='有模型报告的可见物体脱离，仍需复核此前是否抓持。';refs=release
    elif tip and response:
        status='interaction_hypothesis';reason='局部尖端运动与界面变化共同提出交互假设，不能证明独立按压次数或精确接触。';refs=tip+response
    else:
        status='insufficient_evidence';reason='相对运动、屏幕变化或遮挡本身不足以确认独立物体交互；不等于动作不存在。';refs=[o['id'] for o in observations]
    return {'status':status,'reason':reason,'fact_ids':refs,'new_action_confirmed':False,'boundary_replacement_allowed':False}


def review(ep,issue,n):
    event=issue['observation']['event']
    start=max(0,event['start_frame']-5);end=min(n-1,event['end_frame']+5)
    frames=np.linspace(start,end,min(18,end-start+1)).round().astype(int).tolist()
    prompt=f'''只做画面事实记录，不做任务标注，不知道任何候选动作，也不要推断行为意图。范围{start}—{end}帧。
逐视角比较提供的图片，描述物体相对相机的位置变化、可见屏幕明暗/内容变化、尖端相对按钮的局部运动、按钮是否能看到形变、是否能直接看到被抓持物体脱离。
禁止使用“按压成功”“施加压力”“保持接触”等推断充当可见事实。按钮被挡住时记录occlusion，不能声称看见按钮形变。夹爪在腕部相机中不动不能说明手臂静止。屏幕数值看不清则不要转写，屏幕变亮不自动代表触发按钮。
不需要填满类别，没有可见证据就不列该事实。仅在实际提供的帧中引用端点；采样间未提供的帧不作判断。
输出JSON：{{"observations":[{{"view":"head_rgb/head_right_rgb/left_wrist_rgb/right_wrist_rgb","start_frame":0,"end_frame":1,"kind":"relative_motion/screen_change/local_tip_motion/visible_button_deformation/visible_object_release/occlusion/no_clear_change/unclear","visibility":"clear/occluded/unclear","description":"具体可见变化，避免动作意图及接触力推断"}}],"limitations":"看不到或无法区分的内容"}}。'''
    raw=ask(ep,'facts_v1',frames,prompt,detail=True)
    facts=validate_facts(raw,frames)
    decision=interpret(facts)
    return {'source_observation':issue['observation'],'previous_review':issue['review'],
            'frames':frames,'sampling':'all_frames' if len(frames)==end-start+1 else 'uniform_max_18',
            'facts':facts,'limitations':raw.get('limitations',''),'interpretation':decision,
            'candidate_label_provided_to_fact_extractor':False}


def run(ep):
    source=annotate.OUT/'automatic/multiview_refined/decisions'/f'{ep}.json'
    audit=json.loads(source.read_text(encoding='utf-8'))
    n=len(annotate.telemetry(ep)[0])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda issue:review(ep,issue,n),audit['dispute_reviews']))
    counts={key:sum(r['interpretation']['status']==key for r in results) for key in ['interaction_candidate','object_release_candidate','interaction_hypothesis','insufficient_evidence']}
    result={'trajectory_id':ep,'method':'label-blind visual facts then deterministic evidence sufficiency gate',
            'results':results,'counts':counts,'new_actions_confirmed':0,'automatic_annotation_replaced':False,
            'reference_used':False,'limitations':'Facts are model observations, not verified ground truth; sufficiency rules are conservative and may miss occluded interactions.'}
    annotate.save(annotate.OUT/'automatic/fact_review'/f'{ep}.json',result)
    print('FACT REVIEW',ep,counts,flush=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('episode');run(parser.parse_args().episode)
