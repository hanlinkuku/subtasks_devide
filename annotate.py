"""Frame-indexed multiview annotation; credentials only come from the environment."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio_ffmpeg
from PIL import Image, ImageDraw
import requests

ROOT = Path(__file__).resolve().parent
DATA = next(ROOT.glob('*/Turn_On_And_Off_The_Air_Conditioner_raw_lerobot'))
OUT = ROOT / 'outputs'
MODEL = 'qwen3-vl-32b-instruct'
BASE = os.environ.get('DASHSCOPE_BASE_URL', 'https://ws-wlxbevlcz6n48rko.cn-beijing.maas.aliyuncs.com/api/v1')
VIEWS = ['head_rgb', 'head_right_rgb', 'left_wrist_rgb', 'right_wrist_rgb']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def prepare():
    for video in sorted(DATA.glob('videos/**/*.mp4')):
        dest = OUT / 'frames' / video.stem / video.parent.name.split('.')[-1]
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / '000000.jpg').exists():
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(video),
                            '-vf', 'scale=640:360', '-vsync', '0', '-start_number', '0',
                            '-q:v', '2', str(dest / '%06d.jpg')], check=True)
        print(video.stem, dest.name, len(list(dest.glob('*.jpg'))), flush=True)


def frame(ep, n):
    canvas = Image.new('RGB', (1280, 764), '#141820')
    draw = ImageDraw.Draw(canvas)
    for i, view in enumerate(VIEWS):
        x, y = (i % 2)*640, (i // 2)*382
        draw.text((x+8,y+4), f'FRAME {n} | {view} | t={n/15:.3f}s', fill='white')
        with Image.open(OUT / 'frames' / ep / view / f'{n:06d}.jpg') as im:
            canvas.paste(im, (x,y+22))
    return canvas


def call(label, prompt, ep, indices, focused=False, sheets=False):
    dest = OUT / 'evidence' / ep / (label + '.json')
    if dest.exists():
        cached=json.loads(dest.read_text(encoding='utf-8'))
        if cached.get('model')!=MODEL or cached.get('prompt')!=prompt or cached.get('frames')!=indices:
            raise ValueError(f'Cache input mismatch for {dest.name}; use a new review label')
        return cached['result']
    content = [{'text': prompt}]
    batches=[indices[i:i+12] for i in range(0,len(indices),12)] if sheets else [[n] for n in indices]
    for batch in batches:
        n=batch[0]
        buf = io.BytesIO()
        if sheets:
            im=Image.new('RGB',(1600,250*((len(batch)+3)//4)),'#141820')
            draw=ImageDraw.Draw(im)
            from PIL import ImageFont
            font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
            for i,k in enumerate(batch):
                x,y=(i%4)*400,(i//4)*250
                with Image.open(OUT/'frames'/ep/'left_wrist_rgb'/f'{k:06d}.jpg') as source:
                    source=source.resize((400,225))
                    im.paste(source,(x,y+25))
                draw.text((x+8,y),f'FRAME {k}',font=font,fill='white')
            im.save(OUT/f'{ep}_{label}_{n}.jpg')
        else:
            im = frame(ep,n)
            if focused:
                im = im.crop((0,0,640,764))
        im.save(buf, format='JPEG', quality=85)
        content.extend([{'text': f'Original frames={batch}, read left-to-right then top-to-bottom'},
                        {'image': 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()}])
    payload = {'model': MODEL, 'input': {'messages': [{'role':'user','content':content}]},
               'parameters': {'temperature': 0.1, 'max_tokens': 6000}}
    print('CALL',ep,label,'images',len(indices),flush=True)
    for attempt in range(3):
        r = requests.post(BASE.rstrip('/')+'/services/aigc/multimodal-generation/generation',
                          headers={'Authorization': 'Bearer '+os.environ['DASHSCOPE_API_KEY']},
                          json=payload, timeout=(30,240))
        if r.status_code in [429,500,502,503,504] and attempt<2:
            time.sleep(3*(attempt+1)); continue
        if not r.ok:
            raise RuntimeError(f'API HTTP {r.status_code}: {r.text[:1000]}')
        response = r.json()
        blocks = response['output']['choices'][0]['message']['content']
        raw = '\n'.join(x.get('text','') for x in blocks)
        cleaned = raw.strip()
        if cleaned.startswith('```'): cleaned = cleaned.split('\n',1)[1].rsplit('```',1)[0]
        try: result = json.loads(cleaned)
        except json.JSONDecodeError:
            save(dest.with_suffix('.unparsed.json'), {'raw':raw})
            raise
        checks=check_proposal(ep,result,indices)
        save(dest, {'model':MODEL,'prompt':prompt,'frames':indices,'usage':response.get('usage'),
                    'request_id':response.get('request_id'),'result':result,'proposal_checks':checks})
        print('DONE', label, json.dumps(result,ensure_ascii=False)[:600],flush=True)
        return result


def check_proposal(ep, result, indices):
    """Reject objectively inconsistent proposals; this is not semantic ground truth."""
    import numpy as np
    segments=result.get('segments',result.get('actions',[]))
    issues=[]
    cursor=min(indices)
    telemetry_path=OUT/'telemetry'/f'{ep}.json'
    xyz=telemetry(ep)[0] if telemetry_path.exists() else None
    for i,seg in enumerate(segments):
        start,end=seg.get('start_frame'),seg.get('end_frame')
        if type(start)!=int or type(end)!=int or not min(indices)<=start<=end<=max(indices):
            issues.append({'segment':i,'reason':'invalid_frame_interval'});continue
        if start!=cursor:issues.append({'segment':i,'reason':'gap_or_overlap','expected_start':cursor})
        cursor=end+1
        if seg.get('skill_id') not in {'reach','grasp','move','rotate','press','pull','push','release','wait'}:
            issues.append({'segment':i,'reason':'invalid_skill'})
        if xyz is not None and seg.get('skill_id')=='wait':
            displacement=float(np.max(np.linalg.norm(xyz[start:end+1]-xyz[start],axis=1))*1000)
            if displacement>10:
                issues.append({'segment':i,'reason':'wait_despite_end_effector_motion','excursion_mm':round(displacement,3)})
    if segments and cursor!=max(indices)+1:issues.append({'reason':'uncovered_tail'})
    return {'structurally_consistent':not issues,'issues':issues,
            'semantic_accuracy_verified':False,'exact_boundaries_verified':False,
            'eligible_for_final_delivery':False}


COMMON = '''你是机器人轨迹原子动作标注员。每张图是同一原始帧的四个同步视角（左上head_rgb，右上head_right_rgb，左下left_wrist_rgb，右下right_wrist_rgb），不是四个时间点。机械臂左右以机器人本体为准，不按屏幕左右。图片上标有原始frame_index，必须使用这些原始编号，不是图片序号。帧率15fps。只根据可见证据，不臆测接触、按压次数、温度读数或成功状态。技能可用reach/grasp/move/rotate/press/pull/push/release/wait；撤离用move并在动作描述中说明。一个子任务一个主要原子动作，不能把接近、按压、撤离合并。连续相同目的的运动不任意拆段。区分按压保持和新的独立按压。只输出JSON，不加代码围栏。'''


def coarse(ep):
    count = len(list((OUT/'frames'/ep/'head_rgb').glob('*.jpg')))
    indices = sorted(set(range(0,count,15))|{count-1})
    prompt = COMMON + '''\n这是全轨迹每秒抽帧，任务级参考是操作空调开关和温度，参考名称并不证明实际发生。识别全部可见动作阶段，包括有意义的等待、接近、接触/按压、移向另一个按钮、撤离。当前只给候选边界，不要声称精确。输出 {"task_name":"中文总体任务", "segments":[{"start_frame":0,"end_frame":10,"skill_id":"wait","arm_used":"none","action":"中文","evidence":"观察依据","uncertainty":"边界或动作疑点"}],"object":{"name":"","colors":[],"shapes":[],"materials":[],"sizes":[],"others":[],"description":""},"review_focus":["需要逐帧复核的内容"]}。segments按时间顺序覆盖整条轨迹，起止帧为闭区间。未知物体属性用空数组/空字符串，不猜品牌材质尺寸。'''
    return call('coarse',prompt,ep,indices)


def dense_window(ep, start, end):
    prompt = COMMON.replace('每张图是同一原始帧的四个同步视角（左上head_rgb，右上head_right_rgb，左下left_wrist_rgb，右下right_wrist_rgb），不是四个时间点。',
        '每张图上下两格是同一原始帧的头部head_rgb视角与左腕left_wrist_rgb视角，不是两个时间点。')
    prompt += f'''\n下面是原视频第{start}至{end}帧，逐帧提供，没有跳帧。请沿时间顺序追踪夹爪和面板相对位置，识别每次独立动作。任务参考为操作温控面板，但不得由目标推断实际动作。区分接近reach、按压press、按钮间移动move、离开面板撤离move、静止等待wait。按压可包括有明确证据的向按钮推进、保持接触、回弹；无接触证据时不要编造按压，可描述贴近面板调整位置。屏幕变化是辅助证据，不单凭屏幕推算接触边界。切勿把持续静止误判为持续按压，也不要把同一动作每次轻微晃动切为新任务。若窗口中间开始/结束的是持续动作，在continues_from_previous/continues_to_next中说明。\n输出 {{"segments":[{{"start_frame":{start},"end_frame":{end},"skill_id":"wait","arm_used":"none","action":"中文可观察动作","evidence":"说明支持动作与起止边界的画面变化，引用帧号","uncertain":false,"continues_from_previous":false,"continues_to_next":false}}],"possible_missed_actions":[],"screen_changes":[{{"frame":0,"observation":"只写清晰可见的变化"}}]}}。segments必须严格覆盖[{start},{end}]的所有整数帧，使用包含端点的闭区间，相邻段下一start=上一end+1。窗口外不可见内容不要断言。若边界或接触无法可靠判断，uncertain=true并在evidence说明。'''
    return call(f'dense_{start:06d}_{end:06d}',prompt,ep,list(range(start,end+1)),focused=True)


def dense(ep):
    count = len(list((OUT/'frames'/ep/'head_rgb').glob('*.jpg')))
    jobs=[(s,min(s+59,count-1)) for s in range(0,count,48)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(dense_window,ep,s,e):(s,e) for s,e in jobs}
        errors=[]
        for f in as_completed(futures):
            try: f.result()
            except Exception as exc:
                errors.append((futures[f],str(exc))); print('ERROR',errors[-1],flush=True)
        if errors: raise RuntimeError(errors)


def telemetry(ep):
    import numpy as np
    data=json.loads((OUT/'telemetry'/f'{ep}.json').read_text())
    cols=data['columns']; rows=data['rows']
    xyz=np.array([r[cols.index('observation.state.left_ee_pose')][:3] for r in rows])
    return xyz, rows, cols


def review(ep,start,end):
    import numpy as np
    xyz,rows,cols=telemetry(ep)
    motion=[]
    for n in range(start,end+1):
        motion.append({'frame':n,'left_xyz_mm':[round(float(v)*1000,2) for v in xyz[n]],
                       'step_mm':round(float(np.linalg.norm(xyz[n]-xyz[max(0,n-1)]))*1000,3)})
    prompt=COMMON.replace('每张图是同一原始帧的四个同步视角（左上head_rgb，右上head_right_rgb，左下left_wrist_rgb，右下right_wrist_rgb），不是四个时间点。',
        '每张图上下两格是同一原始帧的头部head_rgb视角与左腕left_wrist_rgb视角，不是两个时间点。')
    prompt+='''\n关键：腕部相机固定在机械臂上！因此夹爪在腕部图中的像素位置固定完全不意味着机械臂静止。必须看墙面、门框、面板相对腕部相机的变化，并交叉看固定头部视角。以下还提供同帧左末端位置，单位毫米，是实际运动的辅助证据（不直接证明接触或任务成功）。不要把手臂返回途中判成wait。持续向同一按钮靠近是一个reach，不要分成move再reach。屏幕数值变化不等于新按压：长按自动连增温度可能属于一次press，不得按每次数字跳变切片。只在实际松开再按或换按钮等证据下拆分。接触不清晰则明确uncertain=true。\n'''
    prompt+=f'逐帧审核范围[{start},{end}]，所有帧均提供。运动数据：'+json.dumps(motion,separators=(',',':'))
    prompt+='''\n输出 {"segments":[{"start_frame":0,"end_frame":1,"skill_id":"reach","arm_used":"left","action":"中文","evidence":"具体相对位置、运动和接触证据及帧号","uncertain":true}],"boundary_uncertainty":[{"frame":0,"earliest":0,"latest":0,"reason":"不确定依据"}],"screen_changes":[],"missed_action_check":"是否有遗漏独立技能"}。严格覆盖本窗口所有帧，闭区间连续无重叠。不要根据任务文字补全未观察的动作，不要编造精度。'''
    return call(f'review_{start:06d}_{end:06d}',prompt,ep,list(range(start,end+1)),focused=True)


def review_all(ep):
    windows=([(0,35),(48,95),(90,137),(132,179),(168,207),(272,291),(324,351),(400,447),(436,463),(554,579)]
             if ep.endswith('0') else [(0,35),(80,127),(116,159),(152,191),(184,227),(348,373)])
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(review,ep,s,e):(s,e) for s,e in windows}
        errors=[]
        for f in as_completed(jobs):
            try:f.result()
            except Exception as exc:errors.append((jobs[f],str(exc)));print('ERROR',errors[-1],flush=True)
        if errors:raise RuntimeError(errors)


def sheet_review(ep,start,end):
    prompt='''你要读机器人腕部相机的逐帧接触图。图片是时间序列拼图，按从左到右、从上到下顺序，每格FRAME数字是原始帧号，不要混淆。相机固定于左臂，因此黑色夹爪固定在画面中；需要看面板相对夹爪的距离和遮挡。逐格比较，不要套用预想的动作。先列出屏幕背光或内容变化的首次帧号，再判断是否有明显的靠近/触碰/退离。屏幕变化本身不确定机械接触起始时刻；无法确定必须给不确定区间。不要从反光把数字读错，不清晰就写看不清。不要把长按的每次数字跳变拆成独立按压。输出JSON {"observations":[{"frame":0,"observation":"具体看到什么变化"}],"segments":[{"start_frame":0,"end_frame":1,"skill_id":"reach/press/move/wait中选一","action":"中文动作","evidence":"可见证据","uncertain":true}],"boundary_uncertainty":[{"frame":0,"earliest":0,"latest":0,"reason":""}]}。范围内闭区间连续覆盖，不能臆测窗口外的动作。'''
    return call(f'sheet_{start:06d}_{end:06d}',prompt,ep,list(range(start,end+1)),sheets=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','coarse','dense','review'])
    parser.add_argument('--episode',default='episode_000000')
    args=parser.parse_args()
    if args.stage=='prepare': prepare()
    elif args.stage=='coarse': coarse(args.episode)
    elif args.stage=='dense': dense(args.episode)
    else: review_all(args.episode)
