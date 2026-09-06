"""Fixed multiview coverage and local review. No agent or reference labels."""
import base64
import copy
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import annotate
import automatic_segment as automatic

STAGE='multiview_refined'
VERSION='fixed-multiview-v1'


def windows(n, size=120, overlap=30):
    result=[];start=0
    while start<n:
        end=min(n-1,start+size-1);result.append((start,end))
        if end==n-1:break
        start=end-overlap+1
    return result


def ask(ep,kind,frames,prompt,detail=False):
    frames=sorted(set(int(x) for x in frames))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    blobs=[];hashes=[]
    for frame in frames:
        canvas=Image.new('RGB',(1280,772),'#15231d');draw=ImageDraw.Draw(canvas)
        for i,view in enumerate(annotate.VIEWS):
            path=annotate.OUT/'frames'/ep/view/f'{frame:06d}.jpg'
            raw=path.read_bytes();hashes.append(hashlib.sha256(raw).hexdigest())
            x,y=(i%2)*640,(i//2)*386
            draw.text((x+6,y+1),f'FRAME {frame} | {view}',font=font,fill='white')
            with Image.open(io.BytesIO(raw)) as image:canvas.paste(image,(x,y+26))
        buf=io.BytesIO();canvas.save(buf,format='JPEG',quality=92);blobs.append(buf.getvalue())
        if detail:
            # Fixed high-resolution supplement for this dataset, in addition to all
            # four views. This is not agent-selected evidence.
            raw=(annotate.OUT/'native'/ep/f'{frame:06d}.jpg').read_bytes()
            hashes.append(hashlib.sha256(raw).hexdigest())
            with Image.open(io.BytesIO(raw)) as im:
                im=im.copy();ImageDraw.Draw(im).text((6,3),f'FRAME {frame} | left_wrist DETAIL',font=font,fill='red')
                buf=io.BytesIO();im.save(buf,format='JPEG',quality=92);blobs.append(buf.getvalue())
    if detail:prompt='每个采样时刻先提供四视角图，紧接一张同帧左腕原分辨率细节。\n'+prompt
    prompt='每张图片为同一原始帧的四视角：左上head_rgb，右上head_right_rgb，左下left_wrist_rgb，右下right_wrist_rgb。腕部相机随手运动，夹爪在腕部图中固定不代表手臂静止。请同时检查左右臂和物体变化。\n'+prompt
    parameters={'temperature':0.01,'max_tokens':3000}
    signature=hashlib.sha256(json.dumps([VERSION,annotate.MODEL,parameters,prompt,frames,hashes],ensure_ascii=False).encode()).hexdigest()[:20]
    dest=annotate.OUT/'automatic'/STAGE/'evidence'/ep/f'{kind}_{signature}'
    dest.mkdir(parents=True,exist_ok=True)
    cache=dest/'response.json'
    if cache.exists():return json.loads(cache.read_text(encoding='utf-8'))['result']
    content=[{'text':prompt}]
    for i,blob in enumerate(blobs):
        (dest/f'{i:02d}_frame_{frames[i//2 if detail else i]:06d}.jpg').write_bytes(blob)
        content.append({'image':'data:image/jpeg;base64,'+base64.b64encode(blob).decode()})
    annotate.save(dest/'context.json',{'model':annotate.MODEL,'parameters':parameters,'prompt':prompt,'frames':frames,'source_image_sha256':hashes,'version':VERSION,'reference_used':False})
    print('MULTIVIEW',ep,kind,frames[0],frames[-1],flush=True)
    response=requests.post(annotate.BASE+'/services/aigc/multimodal-generation/generation',
        headers={'Authorization':'Bearer '+os.environ['DASHSCOPE_API_KEY']},
        json={'model':annotate.MODEL,'input':{'messages':[{'role':'user','content':content}]},'parameters':parameters},timeout=(30,240))
    response.raise_for_status();data=response.json()
    raw='\n'.join(x.get('text','') for x in data['output']['choices'][0]['message']['content']).strip()
    if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0]
    result=json.loads(raw)
    annotate.save(cache,{'result':result,'raw_text':raw,'usage':data.get('usage')})
    return result


def resolve_dispute(ep,observation,n):
    event=observation['event'];start=max(0,event['start_frame']-5);end=min(n-1,event['end_frame']+5)
    frames=np.linspace(start,end,min(18,end-start+1)).round().astype(int).tolist()
    _,rows,cols=annotate.telemetry(ep)
    motion=[]
    for f in frames:
        entry={'frame':f}
        for arm in ['left','right']:
            idx=cols.index('observation.state.'+arm+'_ee_pose')
            delta=(np.asarray(rows[f][idx][:3])-np.asarray(rows[max(0,f-1)][idx][:3]))*1000
            entry[arm+'_delta_xyz_mm']=delta.round(3).tolist()
        motion.append(entry)
    prompt=f'''一次粗观察提出了可能的额外交互，尚未确认：{json.dumps(event,ensure_ascii=False)}。
现在独立复查{start}—{end}帧。不要把该提议或原回答当作事实。请判断本窗口是否存在可区分的、独立于普通接近/对准/撤回的物体交互。
重点区分：持续移动时面板相对腕部相机变化；屏幕反光/刷新；局部按压推进与保持；松开后重新按压。夹爪遮住按钮不能单独证明接触，屏幕数值没变不能证明按压，屏幕多次刷新不能证明多次按压。
release在这里仅指松开抓持的物体；按按钮后撤归为move，不单独作release。不要推断接触力，不要猜温度或按钮用途。
双臂逐帧位移辅助数据（不是接触传感器）：{json.dumps(motion,separators=(',',':'))}
返回JSON：{{"verdict":"supported/rejected/uncertain","evidence":"直接说明哪些帧与视角支持或反驳该提议","skill_id":"press/move/reach/release/other","arm_used":"left/right/both/unknown"}}。证据不够就uncertain，不能强行二选一。'''
    result=ask(ep,'dispute',frames,prompt,detail=True)
    if result.get('verdict') not in {'supported','rejected','uncertain'} or not isinstance(result.get('evidence'),str):raise ValueError('Invalid dispute verdict')
    return {'observation':observation,'review':result}


def survey(ep,interval):
    start,end=interval
    frames=np.linspace(start,end,min(16,end-start+1)).round().astype(int).tolist()
    prompt=f'''观察范围{start}—{end}帧。独立描述本窗口中可见的动作，不要猜完整任务或预设哪只手操作。
重点发现按钮操作、抓取、推拉等交互，以及右臂是否存在有目的的操作。普通接近和撤回也列出，但不要仅因停在物体前就认定press。连续保持时物体数次响应不等于多次按压。稀疏采样不能确定精确边界。
输出JSON：{{"actions":[{{"skill_id":"reach/move/press/grasp/release/push/pull/rotate/wait","arm_used":"left/right/both/none/unknown","start_frame":0,"end_frame":1,"evidence":"可见证据和视角","uncertain":true}}],"limitations":"遗漏风险或遮挡"}}。边界必须在本窗口内。'''
    result=ask(ep,'survey',frames,prompt)
    for event in result['actions']:
        s,e=event.get('start_frame'),event.get('end_frame')
        if type(s)!=int or type(e)!=int or not start<=s<=e<=end:raise ValueError('Invalid survey interval')
    return {'window':[start,end],'frames':frames,'result':result}


def review_event(ep,event,n):
    start=max(0,event['start_frame']-8);end=min(n-1,event['end_frame']+8)
    # All frames near both boundaries, with coarser context during the middle.
    frames=sorted(set(range(start,min(end,event['start_frame']+5)+1))|set(range(max(start,event['end_frame']-5),end+1))|set(np.linspace(start,end,6).round().astype(int).tolist()))
    prompt=f'''请复核候选交互{event['start_frame']}—{event['end_frame']}帧。候选不是已知正确答案。观察四视角，判断是否实际有按钮交互，哪只手、哪个按钮。
press的操作性定义：对准后的局部按压推进及随后的保持；此前大范围接近属于reach或move。结束为局部按压/保持结束、转入持续撤离之前的最后一帧，不以屏幕是否变化直接代替边界。
请重点检查候选两端连续帧；中部可能稀疏。遮挡下不能宣称精确物理接触，证据不足时保留候选并说明不确定。不要根据看到按钮被遮住就断言接触，也不要编造温度数值。
输出JSON：{{"interaction_supported":true,"start_frame":0,"end_frame":1,"arm_used":"left/right/both/unknown","action":"具体中文动作","evidence":"注明视角和帧号的证据","uncertain":true,"additional_actions":[]}}。只能在所提供帧中选边界，额外操作写additional_actions，不得忽略。'''
    result=ask(ep,'interaction',frames,prompt)
    if type(result.get('interaction_supported')) is not bool:raise ValueError('Missing interaction verdict')
    if result['interaction_supported']:
        s,e=result.get('start_frame'),result.get('end_frame')
        if type(s)!=int or type(e)!=int or s not in frames or e not in frames or s>e:raise ValueError('Invalid multiview boundary')
    return result


def run(ep):
    proposal=json.loads((annotate.OUT/'automatic/motion_refined'/f'{ep}.json').read_text(encoding='utf-8'))
    baseline=json.loads((annotate.OUT/'automatic/onset_refined/decisions'/f'{ep}.json').read_text(encoding='utf-8'))['events']
    with ThreadPoolExecutor(max_workers=2) as pool:
        coverage=list(pool.map(lambda w:survey(ep,w),windows(proposal['frame_count'])))
    annotate.save(annotate.OUT/'automatic'/STAGE/'coverage'/f'{ep}.json',coverage)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reviewed=list(pool.map(lambda e:review_event(ep,e,proposal['frame_count']),baseline))
    unresolved=[]
    for i,event in enumerate(reviewed):
        if not event['interaction_supported'] or event.get('arm_used')!='left' or event.get('additional_actions'):
            unresolved.append({'reason':'local_review_disagreement','index':i,'event':event})
    # Global observations cannot silently disappear merely because no motion peak exists.
    for window in coverage:
        for action in window['result']['actions']:
            if action['skill_id']=='wait':continue
            if action['skill_id'] in {'reach','move'} and action.get('arm_used') not in {'right','both'}:continue
            matched=any(action.get('arm_used')=='left' and action['skill_id']=='press' and
                        action['start_frame']<=e.get('end_frame',-1)+8 and action['end_frame']>=e.get('start_frame',proposal['frame_count'])-8 for e in reviewed if e['interaction_supported'])
            if not matched:unresolved.append({'reason':'unmatched_global_observation','window':window['window'],'event':action})
    # One predetermined review pass, no agent loop and no silent conflict suppression.
    requests_to_check=[]
    for issue in unresolved:
        event=issue['event']
        if issue['reason']=='local_review_disagreement' and event.get('additional_actions'):
            requests_to_check.extend({'reason':'additional_action','event':extra} for extra in event['additional_actions'])
        else:requests_to_check.append(issue)
    with ThreadPoolExecutor(max_workers=2) as pool:
        resolutions=list(pool.map(lambda x:resolve_dispute(ep,x,proposal['frame_count']),requests_to_check))
    remaining=[r for r in resolutions if r['review']['verdict']!='rejected']
    boundary_changes=[{'index':i,'baseline':[old['start_frame'],old['end_frame']],
                      'proposed':[new.get('start_frame'),new.get('end_frame')]} for i,(old,new) in enumerate(zip(baseline,reviewed))
                      if [old['start_frame'],old['end_frame']]!=[new.get('start_frame'),new.get('end_frame')]]
    audit={'baseline_events':baseline,'reviewed_events':reviewed,'initial_conflicts':unresolved,
           'dispute_reviews':resolutions,'unresolved':remaining,'boundary_changes':boundary_changes,
           'status':'review_required' if remaining or boundary_changes else 'consistent',
           'automatic_annotation_replaced':False,'reference_used':False,
           'scope':'fixed four-view coarse coverage and local review; not a short-action recall guarantee'}
    annotate.save(annotate.OUT/'automatic'/STAGE/'decisions'/f'{ep}.json',audit)
    # This experimental reviewer has not earned authority to replace fine boundaries.
    # Keep its hypotheses separate, and retain the established annotation for editing.
    print('MULTIVIEW COMPLETE',ep,audit['status'],'unresolved',len(remaining),'boundary_changes',len(boundary_changes),flush=True)
    return audit


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('episode');run(parser.parse_args().episode)
