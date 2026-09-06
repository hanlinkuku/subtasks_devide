import base64
import copy
import io
import json
import numpy as np
import pytest
from PIL import Image
import annotate
import arm_context as ctx
import automatic_segment as automatic
import refine_onsets


@pytest.mark.parametrize('ep',['episode_000000','episode_000001'])
def test_current_data_selects_left(ep):
    assert ctx.select_translation_arm(ep,15)['selected_arm']=='left'


@pytest.mark.parametrize('mode,expected',[('swap','right'),('both',None),('none',None)])
def test_selection_has_no_left_fallback(mode,expected,monkeypatch):
    points,rows,cols=copy.deepcopy(annotate.telemetry('episode_000000'))
    li=cols.index('observation.state.left_ee_pose');ri=cols.index('observation.state.right_ee_pose')
    for row in rows:
        if mode=='swap':row[li],row[ri]=row[ri],row[li]
        elif mode=='both':row[ri]=row[li]
        else:row[li]=row[ri]
    monkeypatch.setattr(annotate,'telemetry',lambda _: (points,rows,cols))
    assert ctx.select_translation_arm('synthetic',15)['selected_arm']==expected


@pytest.fixture
def routing_input(tmp_path,monkeypatch):
    monkeypatch.setattr(annotate,'OUT',tmp_path)
    monkeypatch.setenv('DASHSCOPE_API_KEY','unit-test-placeholder')
    rows=[[[0,0,0],[n/1000,0,0]] for n in range(30)]
    monkeypatch.setattr(annotate,'telemetry',lambda _: (np.zeros((30,3)),rows,
        ['observation.state.left_ee_pose','observation.state.right_ee_pose']))
    for arm,color in [('left','blue'),('right','red')]:
        for frame in range(30):
            path=ctx.wrist_path('case',frame,arm);path.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(640,360),color).save(path)
    head=tmp_path/'frames/case/head_rgb/000014.jpg';head.parent.mkdir(parents=True)
    Image.new('RGB',(640,360),'black').save(head)
    return {'start_frame':10,'end_frame':16,'peak_frame':14,'arm_used':'right'}


def response(result):
    class Response:
        ok=True
        def json(self):return {'output':{'choices':[{'message':{'content':[{'text':json.dumps(result)}]}}]}}
    return Response()


def assert_red_images(items):
    for item in items:
        raw=base64.b64decode(item['image'].split(',',1)[1])
        with Image.open(io.BytesIO(raw)) as im:
            red,_,blue=im.getpixel((320,100));assert red>200 and blue<40


def test_visual_request_uses_right_image_prompt_and_coordinates(routing_input,monkeypatch):
    captured=[]
    def post(*args,**kwargs):
        captured.append(kwargs['json'])
        return response({'interaction_supported':True,'start_frame':10,'end_frame':16})
    monkeypatch.setattr(automatic.requests,'post',post)
    automatic.refine_event('case',routing_input,15,30,scene_prior=False)
    content=captured[0]['input']['messages'][0]['content'];prompt=content[0]['text']
    assert '固定在右臂' in prompt and '同帧右末端位置' in prompt
    motion=json.loads(prompt.split('同帧右末端位置：')[1])
    assert all(row['xyz_mm'][0]==row['frame'] for row in motion)
    assert_red_images(content[2:])


def test_onset_request_uses_right_images_and_delta(routing_input,monkeypatch):
    captured=[]
    def post(*args,**kwargs):
        payload=kwargs['json'];captured.append(payload)
        text=payload['input']['messages'][0]['content'][0]['text']
        if text.startswith(refine_onsets.SCREEN):return response({'changed':True,'first_changed_frame':12})
        return response({'start_frame':10,'earliest_frame':9,'latest_frame':11,'uncertain':True})
    monkeypatch.setattr(refine_onsets.requests,'post',post)
    result=refine_onsets.refine_one('case',routing_input,-1,15)
    assert result['supported']
    for payload in captured:assert_red_images(payload['input']['messages'][0]['content'][1:])
    prompt=captured[-1]['input']['messages'][0]['content'][0]['text']
    assert prompt.startswith('这是右臂固定腕部相机')
    samples=json.loads(prompt.split('数据：')[1].split('\n输出JSON')[0])
    assert all(s['delta_xyz_mm']==[1.0,0.0,0.0] for s in samples[1:])


def test_missing_right_frame_never_falls_back_left(routing_input):
    ctx.wrist_path('case',0,'right').unlink()
    with pytest.raises(FileNotFoundError):refine_onsets.ask('case','check',[0],'inspect',arm='right')


def test_pipeline_passes_selected_arm_to_both_candidate_passes(tmp_path,monkeypatch):
    import run_workbench_pipeline as pipeline
    monkeypatch.setattr(annotate,'OUT',tmp_path)
    monkeypatch.setattr(pipeline,'prepare_episode',lambda _: 'right')
    seen=[]
    def propose(ep,horizontal_withdrawal=False,arm='left',persist=True):
        seen.append(arm)
        return {'trajectory_id':ep,'interaction_proposals':[{'arm_used':arm}]}
    monkeypatch.setattr(automatic,'propose',propose)
    monkeypatch.setattr(automatic,'run_visual',lambda proposal,**kwargs: [])
    monkeypatch.setattr(automatic,'reconcile_withdrawal',lambda *args: None)
    def onset(ep,**kwargs):
        saved=json.loads((tmp_path/'automatic/motion_refined'/f'{ep}.json').read_text(encoding='utf-8'))
        assert saved['interaction_proposals'][0]['arm_used']=='right'
    monkeypatch.setattr(refine_onsets,'run',onset)
    pipeline.run('case')
    assert seen==['right','right']
