"""Explicit arm identity shared by telemetry, images and model context."""
import numpy as np
import annotate


def require_arm(arm):
    if arm not in {'left','right'}:raise ValueError('Missing or invalid arm identity')
    return arm


def arm_label(arm):return {'left':'左臂','right':'右臂'}[require_arm(arm)]


def xyz(ep,arm):
    _,rows,cols=annotate.telemetry(ep)
    index=cols.index('observation.state.'+require_arm(arm)+'_ee_pose')
    points=np.asarray([row[index][:3] for row in rows],dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Invalid arm XYZ data')
    return points


def wrist_path(ep,frame,arm):
    require_arm(arm)
    folder=annotate.OUT/'native'/ep if arm=='left' else annotate.OUT/'native_views'/ep/'right_wrist_rgb'
    return folder/f'{frame:06d}.jpg'


def select_translation_arm(ep,fps):
    from automatic_segment import motion_intervals
    arms={}
    for arm in ['left','right']:
        intervals,steps,distance=motion_intervals(xyz(ep,arm),fps)
        arms[arm]={'motion_intervals':intervals,'max_step_mm':float(steps.max()),'max_distance_mm':float(distance.max())}
    active=[arm for arm,data in arms.items() if data['motion_intervals']]
    return {'trajectory_id':ep,'selected_arm':active[0] if len(active)==1 else None,
        'status':'single_translation_arm' if len(active)==1 else 'requires_review',
        'arms':arms,'limitation':'Translation only; no interval does not rule out rotation, gripper motion or static interaction.'}
