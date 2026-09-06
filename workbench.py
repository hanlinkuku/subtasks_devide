"""Local frame-accurate review workbench. Run: python workbench.py"""
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import annotate

ROOT = Path(__file__).resolve().parent
MANUAL = annotate.OUT / 'manual_edits'
SKILLS = {'reach', 'grasp', 'move', 'rotate', 'press', 'pull', 'push', 'release', 'wait'}
ARMS = {'left', 'right', 'both', 'none', 'unknown'}
app = FastAPI(title='轨迹标注工作台', docs_url=None, redoc_url=None)
lock = threading.Lock()
jobs = {}


def episodes():
    fps = json.loads((annotate.DATA / 'meta/info.json').read_text(encoding='utf-8'))['fps']
    metadata = [json.loads(line) for line in (annotate.DATA / 'meta/episodes.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    return [{'id': f'episode_{m["episode_index"]:06d}', 'frames': m['length'], 'fps': fps,
             'automatic': automatic_path(f'episode_{m["episode_index"]:06d}').exists(),
             'manual': (MANUAL / f'episode_{m["episode_index"]:06d}.json').exists()} for m in metadata]


def episode(ep):
    if not re.fullmatch(r'episode_\d{6}', ep):
        raise HTTPException(404, '轨迹不存在')
    result = next((item for item in episodes() if item['id'] == ep), None)
    if result is None:
        raise HTTPException(404, '轨迹不存在')
    return result


def automatic_path(ep):
    return annotate.OUT / 'automatic/onset_refined/annotations' / f'{ep}.json'


def revision(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def validate_annotation(value, meta):
    """Enforce inclusive, complete coverage; timestamps are derived, never trusted."""
    obj = copy.deepcopy(value)
    if obj.get('trajectory_id') != meta['id']:
        raise ValueError('轨迹 ID 不一致')
    if obj.get('trajectory_start') != 0 or obj.get('trajectory_end') != meta['frames'] - 1:
        raise ValueError('轨迹范围不可改变')
    if not isinstance(obj.get('task_name'), str):
        raise ValueError('缺少任务名称')
    segments = obj.get('subtasks')
    if not isinstance(segments, list) or not segments:
        raise ValueError('至少需要一个子任务')
    cursor = 0
    for i, seg in enumerate(segments, 1):
        if not isinstance(seg, dict):
            raise ValueError('子任务格式错误')
        s, e = seg.get('start_frame'), seg.get('end_frame')
        if type(s) is not int or type(e) is not int or s != cursor or e < s or e >= meta['frames']:
            raise ValueError(f'子任务 {i} 存在重叠、漏帧或无效边界')
        if seg.get('skill_id') not in SKILLS or seg.get('arm_used') not in ARMS:
            raise ValueError(f'子任务 {i} 的动作或机械臂无效')
        if not all(isinstance(seg.get(k), str) for k in ['standard_action_text', 'notes']):
            raise ValueError('动作描述和备注必须是文本')
        target = seg.get('target_object_attribute')
        if not isinstance(target, dict) or not all(isinstance(target.get(k), str) for k in ['name', 'description']):
            raise ValueError('目标属性格式错误')
        for key in ['colors', 'shapes', 'materials', 'sizes', 'others']:
            if not isinstance(target.get(key), list) or not all(isinstance(v, str) for v in target[key]):
                raise ValueError('目标属性必须为文本列表')
        seg['id'] = i
        seg['start_time'], seg['end_time'] = round(s / meta['fps'], 6), round(e / meta['fps'], 6)
        cursor = e + 1
    if cursor != meta['frames']:
        raise ValueError('子任务未覆盖完整轨迹')
    return obj


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@app.get('/api/episodes')
def list_episodes():
    return episodes()


@app.get('/api/episodes/{ep}/annotation')
def get_annotation(ep: str, source: str = 'latest', experimental_review: bool = False):
    meta = episode(ep)
    if source not in {'latest', 'automatic'}:
        raise HTTPException(400, '无效来源')
    manual = MANUAL / f'{ep}.json'
    path = manual if source == 'latest' and manual.exists() else automatic_path(ep)
    if not path.exists():
        raise HTTPException(404, '尚无标注，请运行自动划分')
    review_path=annotate.OUT/'automatic/multiview_refined/decisions'/f'{ep}.json'
    review=json.loads(review_path.read_text(encoding='utf-8')) if experimental_review and review_path.exists() else None
    facts_path=annotate.OUT/'automatic/fact_review'/f'{ep}.json'
    if review and facts_path.exists():
        review['fact_review']=json.loads(facts_path.read_text(encoding='utf-8'))
    return {'annotation': validate_annotation(json.loads(path.read_text(encoding='utf-8')), meta), 'review':review,
            'source': 'manual' if path == manual else 'automatic', 'revision': revision(manual)}


class SaveRequest(BaseModel):
    annotation: dict
    revision: str | None = None


@app.put('/api/episodes/{ep}/annotation')
def save_annotation(ep: str, body: SaveRequest):
    meta = episode(ep)
    try:
        value = validate_annotation(body.annotation, meta)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc))
    path = MANUAL / f'{ep}.json'
    with lock:
        if revision(path) != body.revision:
            raise HTTPException(409, '另一页面已保存修改，请先导出当前草稿，再重新加载')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        atomic_json(MANUAL / 'history' / ep / f'{stamp}.json', value)
        atomic_json(path, value)
        atomic_json(MANUAL / f'{ep}.meta.json', {'saved_at': stamp, 'source': 'human_edited',
                    'exact_frame_truth_claimed': False, 'automatic_sha256': revision(automatic_path(ep))})
        return {'revision': revision(path), 'annotation': value}


@app.get('/api/episodes/{ep}/frames/{view}/{frame}')
def get_frame(ep: str, view: str, frame: int):
    meta = episode(ep)
    if view not in annotate.VIEWS or not 0 <= frame < meta['frames']:
        raise HTTPException(404, '画面不存在')
    path = annotate.OUT / 'frames' / ep / view / f'{frame:06d}.jpg'
    if not path.exists():
        raise HTTPException(404, '逐帧图像尚未准备完成，请运行自动划分')
    return FileResponse(path, media_type='image/jpeg', headers={'Cache-Control': 'private, max-age=3600'})


class JobRequest(BaseModel):
    episode: str
    api_key: str = Field(default='', max_length=256, repr=False)


def run_job(job_id, ep, key):
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    if key:
        env['DASHSCOPE_API_KEY'] = key
    try:
        with subprocess.Popen([sys.executable, '-u', str(ROOT / 'run_workbench_pipeline.py'), ep],
                              cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              encoding='utf-8', errors='replace') as proc:
            for line in proc.stdout:
                if line.startswith('PROGRESS:'):
                    with lock:
                        jobs[job_id]['message'] = line.strip().removeprefix('PROGRESS:')
            code = proc.wait()
        if code:
            raise RuntimeError('推理未完成。请检查 API Key、网络和源数据；已有标注保持可用。')
        result=get_annotation(ep, 'automatic')
        needs_review=result.get('review') and result['review'].get('status')=='review_required'
        with lock:
            jobs[job_id].update(status='complete', message='自动划分完成；多视角复核存在分歧，请加载结果查看报告' if needs_review else '自动划分完成；可加载结果，人工修订已保留')
    except Exception:
        with lock:
            jobs[job_id].update(status='failed', message='推理未完成。请检查 API Key、网络和源数据；已有标注保持可用。')
    finally:
        env.pop('DASHSCOPE_API_KEY', None)


@app.post('/api/jobs')
def start_job(body: JobRequest):
    episode(body.episode)
    with lock:
        if any(job['status'] == 'running' for job in jobs.values()):
            raise HTTPException(409, '已有自动划分正在运行')
        job_id = uuid.uuid4().hex
        jobs[job_id] = {'id': job_id, 'episode': body.episode, 'status': 'running', 'message': '准备开始'}
    threading.Thread(target=run_job, args=(job_id, body.episode, body.api_key), daemon=True).start()
    return {'id': job_id}


@app.get('/api/jobs/current')
def current_job():
    with lock:
        return copy.deepcopy(next(reversed(jobs.values()), None))


@app.get('/api/jobs/{job_id}')
def job_status(job_id: str):
    with lock:
        if job_id not in jobs:
            raise HTTPException(404, '任务不存在')
        return copy.deepcopy(jobs[job_id])


app.mount('/', StaticFiles(directory=ROOT / 'workbench_ui', html=True), name='ui')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8765, access_log=False)
