"""Independent per-frame display observations; no temporal narrative or action labels."""
import base64
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
import annotate
import re
import time

PROMPT='''只描述这一张图中显示屏的可见内容。不要推断机械臂动作、接触、温度变化趋势或前后状态。没有其他帧。
若数字方向倾斜，请尝试阅读，但模糊、反光、残缺时宁可填null。只填写直接能辨识的数字，不补全数位。不得凭常识猜测温度。
返回完整JSON：{"display_visible":true,"numeric_reading":null,"readability":"clear/partial/unreadable","visible_symbols":[],"description":"不超过100字的可见内容","uncertainty":"不超过100字"}。'''


def usable_reading(result):
    value=result.get('numeric_reading')
    if result.get('display_visible') is not True or result.get('readability')!='clear':return None
    if not isinstance(value,str) or not re.fullmatch(r'\d{1,3}(?:\.\d{1,2})?',value):return None
    return value


def summarize(entries):
    entries=sorted(entries,key=lambda e:e['frame'])
    readings=[{'frame':e['frame'],'value':usable_reading(e['result'])} for e in entries]
    transitions=[];format_conflicts=[]
    for a,b in zip(readings,readings[1:]):
        if b['frame']!=a['frame']+1 or a['value'] is None or b['value'] is None:continue
        pair={'before_frame':a['frame'],'after_frame':b['frame'],'before':a['value'],'after':b['value']}
        if [len(part) for part in a['value'].split('.')] != [len(part) for part in b['value'].split('.')]:
            format_conflicts.append(pair)
        elif float(a['value'])!=float(b['value']):transitions.append(pair)
    return {'readings':readings,'clear_adjacent_numeric_transitions':transitions,
            'reading_format_conflicts':format_conflicts,
            'excluded_unclear_frames':[r['frame'] for r in readings if r['value'] is None],
            'numeric_change_corroborated_by_model_readings':bool(transitions),
            'limitation':'Single-frame model readings may still be wrong; absent transitions do not prove absence of screen changes or button interaction.'}


def inspect(ep,frame):
    source=annotate.OUT/'native'/ep/f'{frame:06d}.jpg';raw=source.read_bytes()
    identity=hashlib.sha256(raw+PROMPT.encode()+annotate.MODEL.encode()).hexdigest()[:20]
    dest=annotate.OUT/'diagnostics/static_display'/ep/f'{frame:06d}_{identity}.json'
    if dest.exists():return json.loads(dest.read_text(encoding='utf-8'))
    for attempt in range(4):
        response=requests.post(annotate.BASE+'/services/aigc/multimodal-generation/generation',headers={'Authorization':'Bearer '+os.environ['DASHSCOPE_API_KEY']},
            json={'model':annotate.MODEL,'input':{'messages':[{'role':'user','content':[{'text':PROMPT},{'image':'data:image/jpeg;base64,'+base64.b64encode(raw).decode()}]}]},'parameters':{'temperature':0.01,'max_tokens':800}},timeout=(30,180))
        if response.status_code!=429:break
        annotate.save(dest.with_suffix('.failure.json'),{'frame':frame,'status':429,'attempt':attempt+1})
        if attempt<3:time.sleep(10*(attempt+1))
    response.raise_for_status();data=response.json();text='\n'.join(x.get('text','') for x in data['output']['choices'][0]['message']['content']).strip()
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
    result=json.loads(text)
    entry={'frame':frame,'episode':ep,'view':'left_wrist_rgb','source_image_sha256':hashlib.sha256(raw).hexdigest(),'model':annotate.MODEL,'parameters':{'temperature':0.01,'max_tokens':800},'prompt':PROMPT,'result':result,'raw_text':text,'usage':data.get('usage')}
    annotate.save(dest,entry);print(frame,result,flush=True);return entry


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('episode');p.add_argument('frames',nargs='+',type=int);a=p.parse_args()
    for frame in a.frames:inspect(a.episode,frame)
