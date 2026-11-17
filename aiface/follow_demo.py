"""Deterministic, network-free face-follow simulation from synthetic observations."""
import csv
from math import pi, sin
from pathlib import Path
from .core import MockMotion
from .head_pose import MonoCalibration, FixedCameraRig, SimulatedFollower, HeadObservation


def run(output=Path('results/follow_demo.csv')):
    motion=MockMotion()
    follower=SimulatedFollower(MonoCalibration(.65,64),FixedCameraRig())
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i in range(250):
        t=i*.02
        # A reproducible synthetic head in front of a FIXED camera.
        distance=.65+.03*sin(2*pi*.22*t)
        eye=64*.65/distance
        obs=HeadObservation(.5,.5+.025*sin(2*pi*.16*t),eye,
                            5*sin(2*pi*.17*t),0,0)
        plan=follower.plan(obs,tuple(motion.q))
        if plan.status == 'TRACK_SIMULATION':
            if motion.halted:
                motion.reset()
            motion.step(plan.joint_angles,.02)
        else:
            motion.emergency_stop()
        rows.append((t,plan.status,distance,plan.distance_m,
                     *(v*180/pi for v in motion.q)))
    with output.open('w',newline='',encoding='utf-8-sig') as out:
        w=csv.writer(out)
        w.writerow(('time_s','status','synthetic_face_distance_m','estimated_distance_m',
                    'q1_deg','q2_deg','q3_deg'))
        w.writerows(rows)
    count=sum(r[1]=='TRACK_SIMULATION' for r in rows)
    print('SYNTHETIC FACE FOLLOW DEMO:',count,'/',len(rows),'valid plans')
    print('CSV:',output)
    if count == 0:
        raise RuntimeError('No valid simulation commands generated')
    print('NO CAMERA / NO MOTOR OUTPUT / NOT A MEASURED ACCURACY RESULT')
    return rows
