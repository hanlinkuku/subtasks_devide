"""Symmetric arm candidate experiment, separate from scene-specific VL refinement."""
import argparse
import annotate
from automatic_segment import propose


def run(ep):
    arms={arm:propose(ep,horizontal_withdrawal=True,arm=arm,persist=False) for arm in ['left','right']}
    result={'trajectory_id':ep,'arms':arms,'arms_with_translation_intervals':[a for a,p in arms.items() if p['motion_intervals']],
            'reference_used_during_inference':False,'automatic_annotation_replaced':False,
            'limitation':'Translation-based button candidates only. Absence does not exclude rotation, gripper or stationary interaction. Right-arm visual refinement is not implemented.'}
    annotate.save(annotate.OUT/'experiments/001_bilateral_candidates'/f'{ep}.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('episode');run(parser.parse_args().episode)
