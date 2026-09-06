"""Locate response events, then infer press onset independently of reference labels."""
import base64
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import requests

import annotate
import automatic_segment as automatic

MODEL='qwen3-vl-32b-instruct'


def ask(ep,kind,frames,prompt):
    blobs=[];fingerprints=[]
    for frame in frames:
        path=annotate.OUT/'native'/ep/f'{frame:06d}.jpg'
        raw=path.read_bytes();fingerprints.append(hashlib.sha256(raw).hexdigest())
        with Image.open(io.BytesIO(raw)) as im:
            im=im.copy();draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,300,34),fill='black')
            draw.text((5,0),f'FRAME {frame}',fill='white',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',26))
            buf=io.BytesIO();im.save(buf,format='JPEG',quality=92)
        blobs.append({'image':'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()})
    identity=hashlib.sha256(json.dumps([MODEL,prompt,frames,fingerprints],ensure_ascii=False).encode()).hexdigest()[:16]
    dest=annotate.OUT/'automatic'/'onset_evidence'/ep/f'{kind}_{identity}.json'
    if dest.exists():return json.loads(dest.read_text(encoding='utf-8'))['result']
    content=[{'text':prompt}]+blobs
    print('CALL',ep,kind,frames[0],frames[-1],flush=True)
    response=requests.post(annotate.BASE+'/services/aigc/multimodal-generation/generation',
        headers={'Authorization':'Bearer '+os.environ['DASHSCOPE_API_KEY']},
        json={'model':MODEL,'input':{'messages':[{'role':'user','content':content}]},
              'parameters':{'temperature':0.01,'max_tokens':1600}},timeout=(30,240))
    if not response.ok:raise RuntimeError(f'HTTP {response.status_code}: {response.text[:500]}')
    data=response.json();raw='\n'.join(x.get('text','') for x in data['output']['choices'][0]['message']['content']).strip()
    if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0]
    result=json.loads(raw)
    annotate.save(dest,{'model':MODEL,'prompt':prompt,'frames':frames,'image_sha256':fingerprints,
        'result':result,'usage':data.get('usage'),'reference_used_during_inference':False})
    print('DONE',kind,json.dumps(result,ensure_ascii=False),flush=True)
    return result


SCREEN='''这是墙面温控面板的腕部相机逐时图，图片FRAME标签是原始帧号。只比较屏幕可见状态，不判断机械臂动作。
找出相对于本窗口第一张图的第一次明确界面变化：数字出现/消失、背光明显切换、设定值首次改变。
不要把面板随相机移动、反光或模糊误认为内容变化；不要猜看不清的数字。只选实际提供的帧号。
输出JSON：{"changed":true,"first_changed_frame":0,"previous_state":"","new_state":"","uncertain":false}。
若没有清晰变化，changed=false，first_changed_frame=null。'''


def brightness_candidates(ep,start,end):
    """Recall cue for brief flashes in this fixed wrist-camera layout.

    Camera motion/exposure can also cause jumps: a VLM must confirm the cue.
    This does not detect contact or replace semantic response verification.
    """
    means=[]
    for n in range(start,end+1):
        with Image.open(annotate.OUT/'native'/ep/f'{n:06d}.jpg') as im:
            w,h=im.size
            region=im.convert('L').crop((round(.05*w),round(.15*h),round(.55*w),round(.85*h)))
            means.append(float(np.asarray(region).mean()))
    delta=np.diff(means)
    if not len(delta):return []
    center=float(np.median(delta));mad=float(np.median(np.abs(delta-center)))
    threshold=max(12,6*1.4826*mad)
    return [{'frame':start+i+1,'luminance_delta':round(float(v),3),'threshold':threshold}
            for i,v in enumerate(delta) if abs(v-center)>threshold]


def refine_one(ep,event,preceding_end,fps):
    first=max(preceding_end+1,event['start_frame']-round(fps))
    last=event['end_frame']
    step=max(1,int(np.ceil((last-first)/17)))
    frames=list(range(first,last+1,step))
    if frames[-1]!=last:frames.append(last)
    coarse=ask(ep,'screen_coarse',frames,SCREEN)
    response_frame=coarse.get('first_changed_frame')
    if not coarse.get('changed') or type(response_frame)!=int or response_frame not in frames:
        return {'supported':False,'reason':'No valid screen response anchor','original_event':event}
    idx=frames.index(response_frame)
    if idx==0:return {'supported':False,'reason':'No earlier unchanged frame','original_event':event}
    fine_frames=list(range(frames[idx-1],response_frame+1))
    fine=ask(ep,'screen_fine',fine_frames,SCREEN)
    anchor=fine.get('first_changed_frame')
    if not fine.get('changed') or type(anchor)!=int or anchor not in fine_frames:
        return {'supported':False,'reason':'Unresolved response anchor','original_event':event}
    # Uniform samples can skip a brief first response and pick a later redraw.
    flash_checks=[]
    for cue in brightness_candidates(ep,first,anchor):
        if cue['frame']>=anchor:continue
        local=list(range(max(first,cue['frame']-2),min(last,cue['frame']+2)+1))
        check=ask(ep,'earlier_flash',local,SCREEN+'\n请特别检查短暂闪现或从暗底变亮底，即使下一帧内容又消失，也属于一次可见响应。')
        candidate=check.get('first_changed_frame')
        flash_checks.append({'cue':cue,'check':check})
        if check.get('changed') and type(candidate)==int and candidate in local and local[0]<candidate<anchor:
            anchor=candidate;fine=check;break
    # Search the local actuation before response, not the whole approach.
    start=max(first,anchor-round(fps));end=min(last,anchor+2)
    points,_,_=annotate.telemetry(ep)
    samples=[]
    for n in range(start,end+1):
        delta=(points[n]-points[max(0,n-1)])*1000
        samples.append({'frame':n,'delta_xyz_mm':np.round(delta,3).tolist(),'step_mm':round(float(np.linalg.norm(delta)),3)})
    prompt=f'''这是左臂固定腕部相机；面板安装在墙上。黑色夹爪左尖端用于操作，会遮挡按钮。
第{anchor}帧观察到屏幕响应。这只是响应时刻，不是按压开始的答案。
请在响应之前找最后一段定向按压推进的起点：已经对准按钮后的局部推进，及随后的接触保持；之前的大范围接近/对准仍属于reach或move。
若先减速或短暂停留，再小幅推进，应辨别后面这次推进；若连续推进无可见过渡，明确写uncertain=true并给可能范围，不要假装能精确判定。
不要因为窗口从某帧开始就把该帧当作press起点。不要把屏幕响应帧直接复制为起点。
逐帧末端增量为辅助证据，不能单独证明接触。数据：{json.dumps(samples,separators=(',',':'))}
输出JSON：{{"start_frame":0,"earliest_frame":0,"latest_frame":0,"uncertain":true,"evidence":"观察到的过渡及判断依据"}}。
帧号必须位于提供的窗口内，earliest_frame <= start_frame <= latest_frame，起点不能晚于响应帧。'''
    result=ask(ep,'onset',list(range(start,end+1)),prompt)
    values=[result.get(k) for k in ['earliest_frame','start_frame','latest_frame']]
    valid=all(type(v)==int for v in values) and start<=values[0]<=values[1]<=values[2]<=anchor
    return {'supported':valid,'response_frame':anchor,'earlier_flash_checks':flash_checks,
            'response_uncertain':bool(coarse.get('uncertain') or fine.get('uncertain')),
            'onset':result,'original_event':event,'scope':'response-assisted inference, not measured contact time'}


def run(ep):
    proposal=json.loads((annotate.OUT/'automatic'/'motion_refined'/f'{ep}.json').read_text(encoding='utf-8'))
    decisions=json.loads((annotate.OUT/'automatic'/'motion_refined'/'decisions'/f'{ep}.json').read_text(encoding='utf-8'))['events']
    candidates=proposal['interaction_proposals']
    if len(candidates)!=len(decisions):raise ValueError('Event mismatch')
    jobs=[(ep,event,candidates[i-1]['end_frame'] if i else -1,proposal['fps']) for i,event in enumerate(candidates)]
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda args:refine_one(*args),jobs))
    updated=[]
    for old,new in zip(decisions,results):
        event=dict(old)
        if new['supported']:
            event['previous_start_frame']=event['start_frame'];event['start_frame']=new['onset']['start_frame']
            event['evidence']='按压开始：'+new['onset']['evidence']+' 结束边界由持续后撤信号定位。'
            event['evidence']+=f" 模型给出的待复核起点范围为{new['onset']['earliest_frame']}—{new['onset']['latest_frame']}帧，该范围不是保证误差界。"
            event['uncertain']=new['onset']['uncertain'] or new['response_uncertain']
        updated.append(event)
    annotate.save(annotate.OUT/'automatic'/'onset_refined'/'decisions'/f'{ep}.json',
        {'events':updated,'details':results,'accepted':False,'reference_used_during_inference':False})
    automatic.export_prediction(proposal,updated,stage='onset_refined')
    print(ep,'onsets',[e['start_frame'] for e in updated],flush=True)


if __name__=='__main__':
    for path in sorted((annotate.OUT/'automatic'/'motion_refined').glob('episode_*.json')):run(path.stem)
