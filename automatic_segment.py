"""Independent kinematic proposals for the current button-operation demo.

Inference never reads confirmed annotations or the hand-reviewed export table.
Kinematics proposes interactions; it cannot establish physical contact by itself.
"""
import argparse
import base64
import io
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from PIL import Image,ImageDraw,ImageFont
import requests

import annotate


def motion_intervals(points, fps):
    """Hysteresis with sustained settling; do not split pauses away from rest."""
    points=np.asarray(points)*1000
    steps=np.r_[0,np.linalg.norm(np.diff(points,axis=0),axis=1)]
    rest=np.median(points[:min(10,len(points))],axis=0)
    distance=np.linalg.norm(points-rest,axis=1)
    intervals=[]; n=len(points); i=1
    settle_frames=max(3,round(fps/3))
    while i<n:
        if steps[i]<=0.5:
            i+=1;continue
        start=i
        while start>0 and steps[start-1]>0.06:start-=1
        j=i+1
        while j<n:
            tail=steps[j:min(j+settle_frames,n)]
            if len(tail)==settle_frames and np.all(tail<0.1) and distance[j]<5:
                break
            j+=1
        intervals.append((start,min(j,n)-1))
        i=j+settle_frames
    return intervals,steps,distance


def propose(ep, horizontal_withdrawal=False, arm='left', persist=True):
    points,rows,cols=annotate.telemetry(ep)
    if arm not in {'left','right'}:raise ValueError('Unknown arm')
    pose_column=cols.index('observation.state.'+arm+'_ee_pose')
    points=np.asarray([row[pose_column][:3] for row in rows],dtype=float)
    if not np.all(np.isfinite(points)):raise ValueError('Invalid arm telemetry')
    info=json.loads((annotate.DATA/'meta'/'info.json').read_text(encoding='utf-8'))
    fps=float(info['fps']);n=len(points)
    indices=np.asarray([r[cols.index('frame_index')] for r in rows])
    timestamps=np.asarray([r[cols.index('timestamp')] for r in rows])
    if not np.array_equal(indices,np.arange(n)) or not np.allclose(timestamps,np.arange(n)/fps,atol=5e-6,rtol=0):
        raise ValueError('Frame/timestamp alignment failed')
    intervals,steps,distance=motion_intervals(points,fps)
    smooth=savgol_filter(distance,7,2)
    peaks,properties=find_peaks(smooth,prominence=5,distance=max(3,round(fps/2)))
    # Small return overshoots near the rest pose are not target interactions.
    peaks=[int(p) for p in peaks if smooth[p]>0.5*np.max(smooth)]
    events=[];segments=[];cursor=0
    for start,end in intervals:
        if cursor<start:segments.append({'start_frame':cursor,'end_frame':start-1,'skill_id':'wait','arm_used':'none'})
        local=[p for p in peaks if start<=p<=end]
        last=start
        for p in local:
            near=smooth[p]-max(1,0.01*smooth[p])
            left=p
            while left>last and smooth[left-1]>=near:left-=1
            horizon=max(4,round(fps/2))
            release=end+1
            for t in range(p+1,end+1):
                future=min(end,t+horizon)
                withdrawal_speed=(np.linalg.norm(points[t,:2]-points[t-1,:2])*1000
                                  if horizontal_withdrawal else steps[t])
                speed_threshold=0.6 if horizontal_withdrawal else 0.5
                if (withdrawal_speed>speed_threshold and smooth[t]<smooth[t-1]
                    and smooth[future]-smooth[t]<-5
                    and np.linalg.norm(points[future]-points[t])*1000>8):
                    release=t;break
            next_peak=next((v for v in local if v>p),None)
            if next_peak is not None:release=min(release,next_peak)
            release=max(left+1,release)
            if last<left:segments.append({'start_frame':last,'end_frame':left-1,'skill_id':'reach' if last==start else 'move','arm_used':arm})
            event={'start_frame':left,'end_frame':release-1,'skill_id':'press','arm_used':arm,
                   'proposal_only':True,'contact_verified':False,'peak_frame':p}
            segments.append(event);events.append(event.copy());last=release
        if last<=end:segments.append({'start_frame':last,'end_frame':end,'skill_id':'move','arm_used':arm})
        cursor=end+1
    if cursor<n:segments.append({'start_frame':cursor,'end_frame':n-1,'skill_id':'wait','arm_used':'none'})
    cursor=0
    for s in segments:
        if s['start_frame']!=cursor or s['end_frame']<cursor:raise ValueError('Invalid partition')
        cursor=s['end_frame']+1
    if cursor!=n:raise ValueError('Incomplete partition')
    result={'trajectory_id':ep,'fps':fps,'frame_count':n,'motion_intervals':intervals,
        'interaction_proposals':events,'segments':segments,'accepted':False,
        'method':'rest hysteresis and radial turning points; interaction labels need visual verification',
        'reference_annotations_used_for_inference':False,
        'source_sha256':hashlib.sha256((annotate.OUT/'telemetry'/f'{ep}.json').read_bytes()).hexdigest()}
    result['withdrawal_detector']='horizontal_displacement_with_sustained_retreat' if horizontal_withdrawal else 'total_displacement_with_sustained_retreat'
    stage='motion_refined' if horizontal_withdrawal else 'kinematic'
    if persist:
        filename=f'{ep}.json' if arm=='left' else f'{ep}.right.json'
        annotate.save(annotate.OUT/'automatic'/stage/filename,result)
    print(ep,'motion',intervals,'interactions',[(e['start_frame'],e['end_frame']) for e in events],flush=True)
    return result


def refine_event(ep,event,fps,n,scene_prior=True,cache_tag=None):
    points,_,_=annotate.telemetry(ep)
    start=max(0,event['start_frame']-12);end=min(n-1,event['end_frame']+12)
    frame_ids=sorted(set(np.linspace(start,end,min(16,end-start+1)).round().astype(int).tolist()
                         +[event['start_frame'],event['end_frame'],event['peak_frame']]))
    prompt=f'''场景：机器人左臂操作墙上的温控面板，腕部相机刚性固定在左臂；面板没有被夹爪抓住。
黑色夹爪的左侧尖端是操作端，会遮住面板按钮。腕部图中的夹爪固定并不表示手臂静止。
以下第一张是固定头部视角，随后是带原始帧号的腕部时序图。轨迹fps={fps}。
运动信号提出的交互候选为[{event['start_frame']},{event['end_frame']}]，这不是已知答案。
请核实有无按钮操作，区分：朝按钮的普通接近、最后的按压推进及接触保持、松开后离开或移向另一按钮。
按钮被遮挡时可以结合尖端与面板的相对位置变化、短距离推进/回撤和屏幕响应推断按压，注明推断依据。
不要因按钮被遮挡就把整段标为wait，也不要只凭一次屏幕变化认定整个接近过程都是press。
长按的数次数字变化并不代表数次独立按压；只有看见松开再按才拆分。
返回JSON：{{"interaction_supported":true,"start_frame":0,"end_frame":1,"action":"中文动作","evidence":"依据和遮挡说明","additional_actions":[],"uncertain":true}}。
若证据不支持按钮操作，interaction_supported=false。允许修正边界到提供窗口内；帧号必须是原始帧号。'''
    if not scene_prior:
        lines=prompt.splitlines()
        prompt='观察实际画面确定操作目标、夹爪与目标的关系，不预设物体类型、安装方式、是否抓持或使用哪根尖端。腕部相机刚性固定在左臂，夹爪在腕部图中固定不代表手臂静止。\n'+'\n'.join(lines[2:])
    motion=[{'frame':k,'xyz_mm':np.round(points[k]*1000,2).tolist()} for k in frame_ids]
    prompt+='\n同帧左末端位置：'+json.dumps(motion,separators=(',',':'))
    signature=hashlib.sha256((prompt+str(frame_ids)).encode()).hexdigest()[:12]
    dest=annotate.OUT/'automatic'/'evidence'/ep/f'event_{event["peak_frame"]}_{signature}.json'
    if cache_tag is not None:
        if not cache_tag.replace('_','').isalnum():raise ValueError('Invalid experiment cache tag')
        dest=dest.with_name(dest.stem+'_'+cache_tag+'.json')
    if dest.exists():return json.loads(dest.read_text(encoding='utf-8'))['result']
    content=[{'text':prompt}]
    files=[(annotate.OUT/'frames'/ep/'head_rgb'/f'{event["peak_frame"]:06d}.jpg',event['peak_frame'])]
    files.extend((annotate.OUT/'native'/ep/f'{k:06d}.jpg',k) for k in frame_ids)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    for path,k in files:
        with Image.open(path) as source:im=source.copy()
        draw=ImageDraw.Draw(im);draw.rectangle((0,0,250,30),fill='black');draw.text((5,0),f'FRAME {k}',fill='white',font=font)
        buf=io.BytesIO();im.save(buf,format='JPEG',quality=90)
        content.append({'image':'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()})
    payload={'model':'qwen3-vl-32b-instruct','input':{'messages':[{'role':'user','content':content}]},
        'parameters':{'temperature':0.1,'max_tokens':2500}}
    print('REFINE',ep,event['peak_frame'],flush=True)
    response=requests.post(annotate.BASE+'/services/aigc/multimodal-generation/generation',
        headers={'Authorization':'Bearer '+os.environ['DASHSCOPE_API_KEY']},json=payload,timeout=(30,240))
    if not response.ok:raise RuntimeError(f'HTTP {response.status_code}: {response.text[:700]}')
    data=response.json();raw='\n'.join(x.get('text','') for x in data['output']['choices'][0]['message']['content']).strip()
    if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0]
    result=json.loads(raw)
    if result.get('interaction_supported'):
        s,e=result.get('start_frame'),result.get('end_frame')
        if type(s)!=int or type(e)!=int or not start<=s<=e<=end:raise ValueError('Invalid refined event interval')
    annotate.save(dest,{'model':'qwen3-vl-32b-instruct','frames':frame_ids,'prompt':prompt,
        'result':result,'usage':data.get('usage'),'reference_annotations_used_for_inference':False})
    print('REFINED',ep,event['peak_frame'],json.dumps(result,ensure_ascii=False),flush=True)
    return result


def run_visual(proposal):
    ep=proposal['trajectory_id']
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda e:refine_event(ep,e,proposal['fps'],proposal['frame_count']),proposal['interaction_proposals']))
    annotate.save(annotate.OUT/'automatic'/'visual'/f'{ep}.json',{'trajectory_id':ep,'events':results,
        'accepted':False,'reference_annotations_used_for_inference':False})
    return results


def export_prediction(proposal, visual=None, stage=None):
    ep=proposal['trajectory_id'];n=proposal['frame_count'];fps=proposal['fps']
    stage=stage or ('vlm_refined' if visual is not None else 'kinematic')
    events=proposal['interaction_proposals'] if visual is None else visual
    conflicts=[]
    source_segments=proposal.get('segments',[])+proposal.get('interaction_proposals',[])
    arms={s.get('arm_used') for s in source_segments if s.get('skill_id')!='wait'}
    if not arms and proposal.get('arm_used') in {'left','right'}:arms={proposal['arm_used']}
    no_motion=not proposal['motion_intervals'] and not arms
    if not no_motion and (len(arms)!=1 or not arms.issubset({'left','right'})):
        conflicts.append({'reason':'unsupported_or_missing_single_arm','arms':sorted(str(a) for a in arms)})
    arm=next(iter(arms)) if len(arms)==1 and arms.issubset({'left','right'}) else 'unknown'
    arm_text={'left':'左臂','right':'右臂','unknown':'机械臂'}[arm]
    for i,event in enumerate(events):
        if event.get('arm_used') is not None and event['arm_used']!=arm:
            conflicts.append({'index':i,'reason':'visual_arm_conflicts_with_proposal','event':event})
        if event.get('additional_actions'):
            conflicts.append({'index':i,'reason':'unresolved_additional_actions','event':event})
        if not event.get('interaction_supported',True):continue
        s,e=event.get('start_frame'),event.get('end_frame')
        if type(s)!=int or type(e)!=int or not 0<=s<=e<n:
            conflicts.append({'index':i,'reason':'invalid_frame_interval','event':event})
        elif not any(a<=s<=e<=b for a,b in proposal['motion_intervals']):
            conflicts.append({'index':i,'reason':'interaction_outside_motion_intervals','event':event})
    audit_path=annotate.OUT/'automatic'/stage/'export_audit'/f'{ep}.json'
    if conflicts:
        annotate.save(audit_path,{'status':'blocked','conflicts':conflicts,'previous_annotation_preserved':True})
        raise ValueError('Unresolved interaction conflicts; see '+str(audit_path))
    events=[e for e in events if e.get('interaction_supported',True)]
    events=sorted(events,key=lambda e:e['start_frame'])
    overlaps=[{'previous':a,'next':b} for a,b in zip(events,events[1:]) if b['start_frame']<=a['end_frame']]
    if overlaps:
        annotate.save(audit_path,{'status':'blocked','conflicts':overlaps,'previous_annotation_preserved':True})
        raise ValueError('Overlapping interaction proposals require review')
    segments=[];cursor=0
    for start,end in proposal['motion_intervals']:
        if cursor<start:segments.append((cursor,start-1,'wait','机械臂在初始位置保持静止。',''))
        last=start
        for e in events:
            if not start<=e['start_frame']<=e['end_frame']<=end:continue
            if e['start_frame']<last:raise ValueError('Overlapping interaction proposals require review')
            if last<e['start_frame']:
                skill='reach' if last==start else 'move'
                action=arm_text+'接近温控面板按钮区域。' if skill=='reach' else arm_text+'移动并调整按钮操作位置。'
                segments.append((last,e['start_frame']-1,skill,action,''))
            segments.append((e['start_frame'],e['end_frame'],'press',e.get('action',arm_text+'操作温控面板按钮。'),e.get('evidence','运动信号候选，未经视觉确认。')))
            last=e['end_frame']+1
        if last<=end:segments.append((last,end,'move',arm_text+'撤回初始位置。',''))
        cursor=end+1
    if cursor<n:segments.append((cursor,n-1,'wait','机械臂在初始位置保持静止。',''))
    obj={'trajectory_id':ep,'trajectory_start':0,'trajectory_end':n-1,
         'task_name':'用'+arm_text+'操作温控面板并调节温度','subtasks':[]}
    if no_motion:obj['task_name']='机械臂保持静止'
    for i,(s,e,skill,action,evidence) in enumerate(segments,1):
        active=skill!='wait'
        obj['subtasks'].append({'id':i,'standard_action_text':action,'skill_id':skill,
            'arm_used':arm if active else 'none',
            'notes':'自动生成，尚未通过人工边界验收。'+evidence,
            'start_frame':s,'end_frame':e,'start_time':round(s/fps,6),'end_time':round(e/fps,6),
            'target_object_attribute':{'name':'温控面板' if active else '',
                'colors':[],'shapes':[],'materials':[],'sizes':[],'others':[],'description':''}})
    cursor=0
    for seg in obj['subtasks']:
        if seg['start_frame']!=cursor or seg['end_frame']<cursor:raise ValueError('Invalid output coverage')
        cursor=seg['end_frame']+1
    if cursor!=n:raise ValueError('Incomplete output coverage')
    annotate.save(audit_path,{'status':'passed','supported_interactions':len(events),'exported_interactions':sum(s['skill_id']=='press' for s in obj['subtasks']),'conflicts':[]})
    annotate.save(annotate.OUT/'automatic'/stage/'annotations'/f'{ep}.json',obj)
    return obj


def reconcile_withdrawal(proposal, visual):
    """Keep VLM-supported starts, use sustained retreat to end button interaction.

    Vertical settling of the wrist alone is not withdrawal in this wall-button
    scene. This detector is scene-specific, not a general contact-force sensor.
    """
    events=[]
    for candidate,semantic in zip(proposal['interaction_proposals'],visual):
        event=dict(semantic)
        if event.get('interaction_supported') and event['start_frame']<=candidate['end_frame']:
            event['vlm_end_frame']=event['end_frame']
            event['end_frame']=candidate['end_frame']
            event['evidence']=event.get('evidence','')+' 结束边界由持续后撤的水平位移信号细化。'
        events.append(event)
    if len(visual)!=len(proposal['interaction_proposals']):raise ValueError('Interaction count changed; rerun semantic analysis')
    annotate.save(annotate.OUT/'automatic'/'motion_refined'/'decisions'/f'{proposal["trajectory_id"]}.json',
        {'events':events,'accepted':False,'reference_used_during_inference':False,
         'scope':'left-arm wall-button interactions; not a general contact detector'})
    return export_prediction(proposal,events,stage='motion_refined')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--episode');parser.add_argument('--visual',action='store_true');args=parser.parse_args()
    episodes=[args.episode] if args.episode else [p.stem for p in sorted((annotate.OUT/'telemetry').glob('*.json'))]
    for ep in episodes:
        proposal=propose(ep)
        export_prediction(proposal)
        if args.visual:
            visual=run_visual(proposal)
            export_prediction(proposal,visual)
            reconcile_withdrawal(propose(ep,horizontal_withdrawal=True),visual)
