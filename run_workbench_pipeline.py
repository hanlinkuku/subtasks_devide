"""Prepare source data and run independent inference for one episode."""
import argparse
import json
import subprocess
import sys

import imageio_ffmpeg
import annotate
import automatic_segment as automatic
import refine_onsets
import multiview_review
import fact_review


def progress(message):
    print('PROGRESS:' + message, flush=True)


def prepare_episode(ep):
    progress('读取轨迹与运动数据')
    sys.path.insert(0, str(annotate.ROOT / '.runtime'))
    import duckdb
    source = next(annotate.DATA.glob(f'data/**/{ep}.parquet'))
    relation = duckdb.sql('SELECT * FROM read_parquet(?) ORDER BY frame_index', params=[str(source)])
    rows = relation.fetchall()
    columns = [column[0] for column in relation.description]
    annotate.save(annotate.OUT / 'telemetry' / f'{ep}.json', {'columns': columns, 'rows': rows})
    count = len(rows)
    progress('检查四视角逐帧图像')
    for view in annotate.VIEWS:
        video = next(annotate.DATA.glob(f'videos/**/observation.images.{view}/{ep}.mp4'))
        destinations = [(annotate.OUT / 'frames' / ep / view, ['-vf', 'scale=640:360'])]
        if view == 'left_wrist_rgb':
            destinations.append((annotate.OUT / 'native' / ep, []))
        for dest, filters in destinations:
            if all((dest / f'{n:06d}.jpg').is_file() for n in range(count)):
                continue
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-i', str(video),
                            *filters, '-vsync', '0', '-start_number', '0', '-q:v', '2',
                            str(dest / '%06d.jpg')], check=True, capture_output=True)
            if not all((dest / f'{n:06d}.jpg').is_file() for n in range(count)):
                raise ValueError('视频与轨迹帧数不一致')


def run(ep, experimental_review=False, scene_prior=False):
    prepare_episode(ep)
    progress('检测运动区间与候选交互')
    proposal = automatic.propose(ep)
    progress('Qwen3-VL-32B 识别交互；相同输入复用推理缓存')
    visual = automatic.run_visual(proposal, scene_prior=scene_prior)
    progress('根据持续后撤细化结束边界')
    automatic.reconcile_withdrawal(automatic.propose(ep, horizontal_withdrawal=True), visual)
    progress('逐帧检查画面响应与按压起点')
    refine_onsets.run(ep)
    if not experimental_review:
        progress('自动划分完成；实验复核未参与正式结果')
        return
    progress('固定四视角全程观察与局部分歧复查')
    audit=multiview_review.run(ep)
    progress('提取分歧窗口的可见事实，检查交互证据充分性')
    fact_review.run(ep)
    progress('自动划分完成；多视角报告含待复核分歧，精细边界未被覆盖' if audit['status']=='review_required' else '自动划分与多视角复核完成')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('episode')
    parser.add_argument('--experimental-review', action='store_true')
    args=parser.parse_args()
    run(args.episode, args.experimental_review)
