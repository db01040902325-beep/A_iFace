"""A_iFace v0.1: simulated 3R arm only. Run: python main.py --demo"""
import argparse
import csv
import math
from pathlib import Path
from aiface.core import ArmConfig, MockMotion, forward, inverse, required_torque


def demo(output: Path):
    cfg=ArmConfig()
    motion=MockMotion(cfg)
    target=(0.24,0.12,math.radians(10))
    q_target=inverse(*target,cfg,current=tuple(motion.q))
    output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i in range(500):
        q=motion.step(q_target,0.02)
        x,z,phi=forward(q,cfg)
        tau=required_torque(q,tuple(motion.dq), (0.,0.,0.),cfg)
        rows.append([round((i+1)*.02,3),x,z,math.degrees(phi),*(math.degrees(v) for v in q),*tau])
    with output.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['t_s','x_m','z_m','phone_pitch_deg','q1_deg','q2_deg','q3_deg','tau1_Nm','tau2_Nm','tau3_Nm'])
        writer.writerows(rows)
    pose=forward(motion.q,cfg)
    print('Simulated pose:',tuple(round(v,4) for v in pose))
    print('Target pose:',tuple(round(v,4) for v in target))
    print('CSV:',output)
    print('CAUTION: No motor controls, no head-distance measurement.')


def main():
    p=argparse.ArgumentParser(description='3R planar arm simulation / optional webcam preview')
    p.add_argument('--demo',action='store_true')
    p.add_argument('--camera',action='store_true')
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--out',type=Path,default=Path('results/demo.csv'))
    a=p.parse_args()
    if a.camera:
        from aiface.vision import preview_camera
        preview_camera(a.device)
    if a.demo or not a.camera:
        demo(a.out)


if __name__ == '__main__': main()
