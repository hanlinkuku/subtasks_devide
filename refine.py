"""Short native-resolution video reviews and telemetry-based motion proposals."""
import base64
import io
import json
import os
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import annotate

MODEL = 'qwen3-vl-32b-instruct'
OUT = annotate.OUT


def video_review(ep, start, end, question, stride=1):
    indices=list(range(start,end+1,stride))
    if indices[-1]!=end: indices.append(end)
    playback_fps=min(10,15/stride)
    prompt=f'''这是左腕相机的视频片段，原视频15fps，帧范围{start}至{end}。画面标注的是原始帧号。
接口播放帧率为{playback_fps}，仅供时序观察；不要使用接口播放秒数定位，必须使用画面上的原始帧号。
相机固定在左臂，黑色夹爪位置固定是正常现象，须看目标相对夹爪的运动。
只依据实际图像作答。无法确定接触的时刻时给范围，禁止凭屏幕变化断定物理接触起始。
{question}
只输出JSON对象，格式为：{{"observations":[{{"frame":0,"description":"可见事实"}}],"actions":[{{"start_frame":0,"end_frame":1,"skill_id":"reach/press/move/wait之一","description":"动作","confidence":"high/medium/low之一"}}],"uncertain_boundaries":[{{"earliest_frame":0,"latest_frame":1,"reason":"原因"}}]}}。
不要猜测品牌、无法辨认的数字或未看到的动作；动作起止采用闭区间。'''
    identity=hashlib.sha256((MODEL+prompt+str(indices)).encode()).hexdigest()[:12]
    dest=OUT/'evidence'/ep/f'video_{start:06d}_{end:06d}_{identity}.json'
    if dest.exists():return json.loads(dest.read_text(encoding='utf-8'))['result']
    frames=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',28)
    for n in indices:
        with Image.open(OUT/'native'/ep/f'{n:06d}.jpg') as original:
            im=original.copy()
        draw=ImageDraw.Draw(im);draw.rectangle((0,0,300,34),fill='black');draw.text((8,0),f'FRAME {n}',fill='white',font=font)
        buf=io.BytesIO();im.save(buf,format='JPEG',quality=92)
        frames.append('data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode())
    payload={'model':MODEL,'input':{'messages':[{'role':'user','content':[
        {'video':frames,'fps':playback_fps},{'text':prompt}]}]},'parameters':{'temperature':0.01,'max_tokens':3500}}
    print('CALL',ep,start,end,'native video frames',len(frames),flush=True)
    response=requests.post(annotate.BASE+'/services/aigc/multimodal-generation/generation',
        headers={'Authorization':'Bearer '+os.environ['DASHSCOPE_API_KEY']},json=payload,timeout=(30,240))
    if not response.ok:raise RuntimeError(f'HTTP {response.status_code}: {response.text[:1000]}')
    data=response.json();raw='\n'.join(v.get('text','') for v in data['output']['choices'][0]['message']['content']).strip()
    if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0]
    result=json.loads(raw)
    annotate.save(dest,{'model':MODEL,'prompt':prompt,'frames':indices,'usage':data.get('usage'),
        'request_id':data.get('request_id'),'result':result})
    print('DONE',ep,start,end,json.dumps(result,ensure_ascii=False)[:450],flush=True)
    return result


def batch(jobs):
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(video_review,*job) for job in jobs]
        for f in as_completed(futures):f.result()


if __name__=='__main__':
    video_review('episode_000000',72,95,'比较按钮和夹爪的相对位置。是否有按压、保持、松开？区分背光点亮与温度改变。每次变化必须给首次可见的原始帧号。')
