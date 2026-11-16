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
    p=argparse.ArgumentParser(description='3R arm simulation or pretrained YuNet face detection')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--demo',action='store_true',help='3R simulated arm and CSV')
    mode.add_argument('--camera',action='store_true',help='webcam YuNet face preview')
    mode.add_argument('--video',type=Path,help='face preview from existing video file')
    mode.add_argument('--download-model',action='store_true',help='download SHA-256-checked pretrained YuNet')
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--face-log',type=Path,help='opt-in CSV tracking logs, no images')
    p.add_argument('--out',type=Path,default=Path('results/demo.csv'))
    a=p.parse_args()
    if a.download_model:
        from aiface.vision import install_model
        print('Verified YuNet model:',install_model())
    elif a.camera or a.video is not None:
        from aiface.vision import preview_camera
        preview_camera(a.device,video=a.video,log_path=a.face_log)
    else:
        demo(a.out)


if __name__ == '__main__': main()
