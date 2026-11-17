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
    p=argparse.ArgumentParser(description='A_iFace simulated arm, face detection and calibrated head-pose preview')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--demo',action='store_true',help='3R simulation')
    mode.add_argument('--follow-demo',action='store_true',help='synthetic face-follow and 3R simulation CSV')
    mode.add_argument('--camera',action='store_true',help='YuNet 2D webcam detection')
    mode.add_argument('--video',type=Path,help='YuNet 2D video detection')
    mode.add_argument('--download-model',action='store_true',help='YuNet ONNX download')
    mode.add_argument('--download-pose-model',action='store_true',help='MediaPipe Face Landmarker bundle')
    mode.add_argument('--pose-camera',action='store_true',help='MediaPipe 3D pose + simulated 3R follow')
    mode.add_argument('--pose-video',type=Path,help='MediaPipe 3D pose + simulated follow from video')
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--face-log',type=Path,help='YuNet numeric CSV, no video saved')
    p.add_argument('--pose-log',type=Path,help='MediaPipe pose and simulation numeric CSV')
    p.add_argument('--out',type=Path,default=Path('results/demo.csv'))
    p.add_argument('--calibration',type=Path,default=Path('calibration/face.json'))
    p.add_argument('--reference-distance-cm',type=float,default=None,
                   help='Manually measured face distance; hold C to calibrate during pose camera')
    p.add_argument('--desired-distance-cm',type=float,default=45.)
    p.add_argument('--horizontal-fov-deg',type=float,default=60.,
                   help='Approximate HFOV for demo geometry, NOT calibrated intrinsics')
    a=p.parse_args()
    if a.download_model:
        from aiface.vision import install_model
        print('Verified YuNet model:',install_model())
    elif a.camera or a.video is not None:
        from aiface.vision import preview_camera
        preview_camera(a.device,video=a.video,log_path=a.face_log)
    elif a.download_pose_model:
        from aiface.pose_camera import download_model
        print('MediaPipe model:',download_model())
    elif a.pose_camera or a.pose_video is not None:
        from aiface.pose_camera import preview_pose
        preview_pose(a.device,video=a.pose_video,reference_distance_cm=a.reference_distance_cm,
                     calibration_path=a.calibration,desired_distance_cm=a.desired_distance_cm,
                     horizontal_fov_deg=a.horizontal_fov_deg,log_path=a.pose_log)
    elif a.follow_demo:
        from aiface.follow_demo import run
        run(a.out if a.out != Path('results/demo.csv') else Path('results/follow_demo.csv'))
    else:
        demo(a.out)


if __name__ == '__main__':
    main()
