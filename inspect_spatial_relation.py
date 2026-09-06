"""Fixed single-frame four-view spatial diagnostic; never confirms contact."""
import argparse
import annotate
from multiview_review import ask

PROMPT='''只记录这一帧中机器人夹爪与可能操作的物体表面的空间关系，不知道前后帧，不判断动作类别、成功与否或显示屏读数。
分别检查四个视角。能同时看到夹爪尖端和物体表面时，描述它们之间是否有直接可见的空隙。只能报告二维投影：投影重叠不代表物理接触，遮挡不代表按压。
对每个视角输出一条记录：arm_used为left/right/both/unknown；relation为visible_gap/projected_overlap/occluded/not_visible/unclear。
visible_gap必须能看见尖端与目标之间的背景空隙；projected_overlap仅表示轮廓在图像中重叠；无法分辨就unclear。不要补全被遮挡的按钮位置。图中多个尖端或目标不唯一时说明歧义。
输出完整JSON：{"views":[{"view":"head_rgb","arm_used":"unknown","target_description":"可见物体","relation":"unclear","evidence":"不超过100字，具体说明尖端和目标的位置、空隙或遮挡"}],"limitations":"不超过100字"}。views必须包含四个视角各一次。'''


def inspect(ep,frame):
    result=ask(ep,'spatial_single_frame',[frame],PROMPT)
    observations=result.get('views')
    if not isinstance(observations,list) or sorted(o.get('view','') for o in observations)!=sorted(annotate.VIEWS):
        raise ValueError('Spatial observation must cover each view exactly once')
    if any(o.get('relation') not in {'visible_gap','projected_overlap','occluded','not_visible','unclear'} for o in observations):
        raise ValueError('Invalid spatial relation')
    record={'episode':ep,'frame':frame,'observations':observations,'limitations':result.get('limitations',''),
            'contact_confirmed':False,'automatic_annotation_replaced':False,'candidate_labels_provided':False}
    annotate.save(annotate.OUT/'diagnostics/spatial'/ep/f'{frame:06d}.json',record)
    print(frame,[(o['view'],o['relation']) for o in observations],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('episode');parser.add_argument('frames',nargs='+',type=int)
    args=parser.parse_args()
    for frame in args.frames:inspect(args.episode,frame)
