"""Replay the current pipeline offline and capture the actual constructed VL payloads.

Cached parsed answers replace HTTP; all reconstructed stages go into a new bundle.
No production annotations or caches are overwritten and no API key is exported.
"""
import argparse
import base64
import copy
import hashlib
import html
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import threading
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont
import annotate
import automatic_segment as automatic
import refine_onsets as onset


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(ep):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export = annotate.OUT / 'replays' / f'{ep}_{stamp}'
    export.mkdir(parents=True, exist_ok=False)
    state = threading.local()
    calls = []
    calls_lock = threading.Lock()
    original_save = annotate.save
    original_post = automatic.requests.post
    old_key = os.environ.get('DASHSCOPE_API_KEY')
    # A placeholder satisfies request construction. HTTP is disabled below.
    os.environ['DASHSCOPE_API_KEY'] = 'OFFLINE_REPLAY_PLACEHOLDER'

    def capture_cache(path):
        if not path.is_file():
            raise RuntimeError(f'Offline replay requires cache: {path.name}')
        state.cache = path

    def capture_save(path, value):
        write_json(export / 'stages' / path.relative_to(annotate.OUT), value)

    class ReplayResponse:
        ok = True
        status_code = 200

        def __init__(self, cached):
            self.cached = cached

        def json(self):
            # Historical caches retained parsed JSON, not the original HTTP envelope.
            return {'output': {'choices': [{'message': {'content': [
                {'text': json.dumps(self.cached['result'], ensure_ascii=False)}]}}]},
                    'usage': self.cached.get('usage')}

    def offline_post(url, *, headers, json, timeout):
        path = state.cache
        cached = __import__('json').loads(path.read_text(encoding='utf-8'))
        prompt = json['input']['messages'][0]['content'][0]['text']
        assert cached['prompt'] == prompt and cached['model'] == json['model']
        with calls_lock:
            calls.append({'cache': path, 'cached': cached, 'payload': copy.deepcopy(json), 'url': url})
        return ReplayResponse(cached)

    # Compile only the three functions needing interception. Their image construction,
    # prompt text, parameters and branching remain the production implementation.
    replaced = []
    try:
        annotate.save = capture_save
        automatic.requests.post = offline_post
        for module, name in [(automatic, 'refine_event'), (onset, 'ask')]:
            function = getattr(module, name)
            source = inspect.getsource(function)
            old = "if dest.exists():return json.loads(dest.read_text(encoding='utf-8'))['result']"
            assert old in source
            source = source.replace(old, 'capture_cache(dest)')
            module.__dict__['capture_cache'] = capture_cache
            exec(compile(source, str(export / f'{name}_instrumented.py'), 'exec'), module.__dict__)
            replaced.append((module, name, function))
        run_source = inspect.getsource(onset.run)
        run_source = run_source.replace("annotate.OUT/'automatic'/'motion_refined'", "REPLAY_EXPORT/'stages'/'automatic'/'motion_refined'")
        onset.__dict__['REPLAY_EXPORT'] = export
        replaced.append((onset, 'run', onset.run))
        exec(compile(run_source, str(export / 'run_instrumented.py'), 'exec'), onset.__dict__)

        proposal = automatic.propose(ep)
        automatic.export_prediction(proposal)
        visual = automatic.run_visual(proposal)
        automatic.export_prediction(proposal, visual)
        automatic.reconcile_withdrawal(automatic.propose(ep, horizontal_withdrawal=True), visual)
        onset.run(ep)
    finally:
        annotate.save = original_save
        automatic.requests.post = original_post
        for module, name, function in replaced:
            setattr(module, name, function)
        if old_key is None:
            os.environ.pop('DASHSCOPE_API_KEY', None)
        else:
            os.environ['DASHSCOPE_API_KEY'] = old_key

    # Stable pedagogical order: semantic checks first, then response/onset per event.
    # Calls within each production stage can execute concurrently.
    def order(call):
        kind = call['cache'].name
        if kind.startswith('event_'):
            return (0, int(kind.split('_')[1]), 0)
        frame = call['cached']['frames'][0]
        rank = 0 if kind.startswith('screen_coarse') else 1 if kind.startswith('screen_fine') else 2 if kind.startswith('earlier_flash') else 3
        events = proposal['interaction_proposals']
        event_index = min(range(len(events)), key=lambda i: abs(events[i]['peak_frame'] - sum(call['cached']['frames']) / len(call['cached']['frames'])))
        return (1, event_index, rank, frame)

    calls.sort(key=order)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
    manifest_calls = []
    context_sections = []
    html_sections = []
    labels = {'event':'交互语义核实', 'screen_coarse':'屏幕响应粗定位', 'screen_fine':'屏幕响应逐帧复查',
              'earlier_flash':'更早闪屏补查', 'onset':'按压起点逐帧细化'}
    for index, call in enumerate(calls, 1):
        cached = call['cached']
        kind = next((k for k in labels if call['cache'].name.startswith(k + '_')), 'unknown')
        name = f'{index:02d}_{kind}'
        folder = export / 'calls' / name
        folder.mkdir(parents=True)
        payload = call['payload']
        write_json(folder / 'request.json', payload)
        readable = copy.deepcopy(payload)
        images = []
        frame_ids = cached['frames']
        semantic = kind == 'event'
        if semantic:
            peak = int(call['cache'].name.split('_')[1])
            image_frames = [peak] + frame_ids
        else:
            image_frames = frame_ids
        content = payload['input']['messages'][0]['content']
        prompt = content[0]['text']
        assert len(content) - 1 == len(image_frames)
        for pos, (item, frame) in enumerate(zip(content[1:], image_frames), 1):
            view = 'head_rgb' if semantic and pos == 1 else 'left_wrist_rgb'
            data = base64.b64decode(item['image'].split(',', 1)[1])
            filename = f'{pos:02d}_{view}_frame_{frame:06d}.jpg'
            image_path = folder / 'images' / filename
            image_path.parent.mkdir(exist_ok=True)
            image_path.write_bytes(data)
            readable['input']['messages'][0]['content'][pos]['image'] = f'images/{filename}'
            with Image.open(image_path) as im:
                width, height = im.size
            source_path = (annotate.OUT / 'frames' / ep / 'head_rgb' / f'{frame:06d}.jpg'
                           if view == 'head_rgb' else annotate.OUT / 'native' / ep / f'{frame:06d}.jpg')
            if not semantic:
                assert sha(source_path) == cached['image_sha256'][pos-1]
            images.append({'position': pos, 'frame': frame, 'view': view,
                           'file': f'images/{filename}', 'width': width, 'height': height,
                           'sha256': sha(image_path), 'source_sha256': sha(source_path)})
        write_json(folder / 'request_readable.json', readable)
        write_json(folder / 'response_parsed.json', cached['result'])
        write_json(folder / 'historical_cache.json', cached)
        write_json(folder / 'images.json', images)
        (folder / 'context.txt').write_text(prompt + '\n', encoding='utf-8')
        sheet = Image.new('RGB', (1280, 208 * ((len(images)+3)//4)), '#17231f')
        draw = ImageDraw.Draw(sheet)
        for i, record in enumerate(images):
            x, y = i % 4 * 320, i // 4 * 208
            with Image.open(folder / record['file']) as im:
                sheet.paste(im.resize((320,180)), (x,y+28))
            draw.text((x+6,y+4), f"{i+1:02d}  FRAME {record['frame']}  {'HEAD' if record['view']=='head_rgb' else 'WRIST'}", font=font, fill='white')
        sheet.save(folder / 'contact_sheet.jpg', quality=90)
        meta = {'id': index, 'kind': kind, 'label': labels[kind], 'directory': f'calls/{name}',
                'mode': 'offline_cached_answer_replay', 'images': len(images), 'frames_in_image_order': image_frames,
                'model': payload['model'], 'parameters': payload['parameters'],
                'cache_source': str(call['cache'].relative_to(annotate.ROOT)), 'cache_sha256': sha(call['cache']),
                'historical_source_image_hashes_available': not semantic}
        manifest_calls.append(meta)
        result_text = json.dumps(cached['result'],ensure_ascii=False,indent=2)
        context_sections.append(f"## {index:02d} {labels[kind]}\n\n图片顺序：{image_frames}\n\n参数：{payload['parameters']}\n\n### 完整 context\n\n{prompt}\n\n### 缓存返回\n\n```json\n{result_text}\n```\n")
        image_links = ''.join(f'<a href="calls/{name}/{r["file"]}" target="_blank">{r["position"]:02d} · 帧 {r["frame"]} · {r["view"]}</a>' for r in images)
        html_sections.append(f'''<section id="call-{index}"><h2>{index:02d} / {labels[kind]}</h2>
<p>{len(images)} 张图片 · 帧顺序 {html.escape(str(image_frames))} · 缓存返回回放</p>
<p class="hint">下图为阅读总览，未作为拼图输入模型。点击下方链接查看实际输入的单张图片。</p>
<a href="calls/{name}/contact_sheet.jpg"><img class="sheet" loading="lazy" src="calls/{name}/contact_sheet.jpg" alt="第 {index} 次调用图片总览"></a>
<div class="image-links">{image_links}</div><details><summary>完整 context（含逐帧运动数据）</summary><pre>{html.escape(prompt)}</pre></details>
<details><summary>模型缓存返回 JSON</summary><pre>{html.escape(result_text)}</pre></details>
<p><a href="calls/{name}/context.txt">context.txt</a> · <a href="calls/{name}/request_readable.json">可读请求</a> · <a href="calls/{name}/request.json">完整请求（含图片 Base64）</a> · <a href="calls/{name}/response_parsed.json">返回 JSON</a></p></section>''')

    final_path = export / 'stages/automatic/onset_refined/annotations' / f'{ep}.json'
    final = json.loads(final_path.read_text(encoding='utf-8'))
    baseline_path = annotate.OUT / 'automatic/onset_refined/annotations' / f'{ep}.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    assert final == baseline, 'Replay output differs from current automatic output'
    shutil.copy2(final_path, export / 'final_annotation.json')
    (export / 'all_contexts.md').write_text('# 完整调用 context 与缓存返回\n\n展示顺序按阶段和事件整理，不表示并发请求的实际完成顺序。\n\n'+'\n'.join(context_sections), encoding='utf-8')
    inputs = export / 'source'
    inputs.mkdir()
    shutil.copy2(annotate.OUT / 'telemetry' / f'{ep}.json', inputs / 'telemetry.json')
    shutil.copy2(annotate.DATA / 'meta/info.json', inputs / 'dataset_info.json')
    for filename in ['annotate.py','automatic_segment.py','refine_onsets.py','run_workbench_pipeline.py','export_annotation_replay.py']:
        shutil.copy2(annotate.ROOT / filename, inputs / filename)
    source_videos = [{'path':str(p.relative_to(annotate.DATA)), 'sha256':sha(p)} for p in sorted(annotate.DATA.glob(f'videos/**/{ep}.mp4'))]
    manifest = {'trajectory_id':ep,'created_at':datetime.now(timezone.utc).isoformat(),'mode':'offline_cached_answer_replay',
                'network_requests':0,'frame_count':proposal['frame_count'],'fps':proposal['fps'],
                'calls':manifest_calls,'image_count':sum(c['images'] for c in manifest_calls),
                'final_matches_current_automatic_output':True,'subtasks':len(final['subtasks']),
                'source_videos':source_videos,'telemetry_sha256':sha(inputs/'telemetry.json'),
                'notes':['Request payloads and labeled images reconstructed by current production functions.',
                         'Parsed cached answers replayed; original raw HTTP responses were not retained.',
                         'Semantic-check caches have no historical input-image hashes; exact historical pixel identity cannot be proven.',
                         'Onset-stage source-image hashes verified against cached hashes.',
                         'No human reference annotations used; no exact-contact accuracy claim.']}
    write_json(export / 'manifest.json', manifest)
    rows=''.join(f'<tr><td>{s["id"]}</td><td>{html.escape(s["standard_action_text"])}</td><td>{s["skill_id"]}</td><td>{s["start_frame"]}–{s["end_frame"]}</td></tr>' for s in final['subtasks'])
    navigation=''.join(f'<a href="#call-{c["id"]}">{c["id"]:02d} {c["label"]}</a>' for c in manifest_calls)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整标注回放 · {ep}</title>
<style>body{{margin:0;background:#f4f3ed;color:#233d32;font:15px/1.8 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1100px;padding:35px 24px;margin:auto}}h1{{font-size:30px}}h2{{font-size:23px}}a{{color:#246856}}section{{padding:24px 0;border-top:1px solid #ced7ca}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef0e8;color:#25362d;padding:18px;font:14px/1.8 Consolas,"Microsoft YaHei",monospace}}.sheet{{width:100%;border-radius:5px}}.hint{{color:#6c7467;font-size:13px}}.image-links,nav{{display:flex;flex-wrap:wrap;gap:8px 18px;margin:16px 0}}.image-links a,nav a{{font-size:12px}}summary{{cursor:pointer;padding:12px 0;font-weight:600}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #d6dcd0}}.notice{{border-left:3px solid #bda264;padding:10px 20px;background:#f5edd9}}</style><main>
<p>ANNOTATION REPLAY / 完整轨迹</p><h1>{ep} · 从运动候选到帧级标注</h1>
<p>{proposal['frame_count']} 帧 / {proposal['fps']:g} FPS / {len(manifest_calls)} 次逻辑 VL 调用 / {manifest['image_count']} 张输入图片 / {len(final['subtasks'])} 个子任务</p>
<div class="notice">本材料是离线缓存回放，实际网络调用为 0。重新运行候选检测、图片与 context 构造、边界细化及输出组装，用已保存的模型回答代替 HTTP 响应。结果与当前自动标注一致，不代表精确接触帧已被验证。</div>
<p>流程：解码与帧号对齐 → 运动候选 → 视觉交互核实 → 持续后撤确定结束 → 屏幕响应定位 → 局部逐帧定位开始 → 连续区间输出。</p>
<p>本次复用已有解码图片及运动数据，附源视频哈希与运动数据副本；没有重新解码视频。原始场景为四路视频，模型交互核实用头部单帧＋左腕序列，后续细化用左腕序列。</p>
<p><a href="all_contexts.md">全部 context 和回答</a> · <a href="manifest.json">总清单</a> · <a href="final_annotation.json">最终标注 JSON</a> · <a href="stages/automatic/kinematic/{ep}.json">初始运动候选</a> · <a href="stages/automatic/motion_refined/{ep}.json">后撤细化候选</a></p>
<nav>{navigation}</nav>{''.join(html_sections)}<section><h2>最终子任务</h2><table><tr><th>序号</th><th>动作</th><th>类型</th><th>闭区间帧号</th></tr>{rows}</table></section>
<section><h2>复现说明</h2><p>request.json 保留本次重建的完整请求体（包含 Base64 图片），request_readable.json 将图片替换成本地相对路径，便于阅读，不能原样提交 API。context.txt 是完整文字输入，无省略。response_parsed.json 是历史缓存解析后的结果，不是原始 HTTP 返回。</p><p>起点阶段的源图片哈希已与历史缓存核对。较早的交互核实缓存没有保存源图片哈希，因此只能保证图片按当前代码与当前源文件重建，不能证明与历史调用逐字节一致。实际网络调用并发执行，本文按阶段与事件顺序排列。</p></section></main></html>'''
    (export/'index.html').write_text(page,encoding='utf-8')
    # Validate every packed request against its image files, and every final interval.
    cursor=0
    for seg in final['subtasks']:
        assert seg['start_frame']==cursor and seg['end_frame']>=cursor
        cursor=seg['end_frame']+1
    assert cursor==proposal['frame_count']
    for call in manifest_calls:
        folder=export/call['directory']
        request=json.loads((folder/'request.json').read_text(encoding='utf-8'))
        records=json.loads((folder/'images.json').read_text(encoding='utf-8'))
        for part,record in zip(request['input']['messages'][0]['content'][1:],records):
            assert base64.b64decode(part['image'].split(',',1)[1])==(folder/record['file']).read_bytes()
    archive=shutil.make_archive(str(export),'zip',root_dir=export.parent,base_dir=export.name)
    print(json.dumps({'directory':str(export),'archive':archive,'calls':len(calls),'images':manifest['image_count'],'subtasks':len(final['subtasks'])},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',default='episode_000000')
    main(parser.parse_args().episode)
