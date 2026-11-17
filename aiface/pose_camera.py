"""MediaPipe 3D face pose + *simulated* 3R follow on a fixed-camera test rig.

No motor interface. Camera is assumed to be FIXED to robot base (not phone).
Single-webcam distance is an approximate personally calibrated estimate only.
"""
import csv
from collections import deque
from dataclasses import asdict
import json
from math import pi, radians, tan
from pathlib import Path
from statistics import median
import time
from urllib.request import Request, urlopen
import zipfile
from io import BytesIO

from .core import ArmConfig, MockMotion
from .head_pose import FixedCameraRig, MonoCalibration, SimulatedFollower, observation_from_landmarks

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'face_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'


def download_model(path=MODEL_PATH):
    """Explicit download from Google's versioned HTTPS endpoint; max 24 MB."""
    p = Path(path)
    if p.exists():
        with zipfile.ZipFile(p) as z:
            if z.testzip() is not None:
                raise ValueError('Invalid model bundle')
        return p
    with urlopen(Request(MODEL_URL, headers={'User-Agent':'A_iFace/0.3'}), timeout=40) as r:
        data = r.read(24*1024*1024+1)
    if len(data) > 24*1024*1024 or data[:2] != b'PK':
        raise ValueError('Model download invalid or too large')
    with zipfile.ZipFile(BytesIO(data)) as z:
        if z.testzip() is not None:
            raise ValueError('Model ZIP integrity failure')
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp = p.with_suffix('.download')
    try:
        tmp.write_bytes(data)
        tmp.replace(p)
    finally:
        tmp.unlink(missing_ok=True)
    return p


def _read_calibration(path):
    p=Path(path)
    if not p.exists():
        return None
    obj=json.loads(p.read_text(encoding='utf-8'))
    if set(obj) != {'reference_distance_m','reference_eye_spacing_px'}:
        raise ValueError('Invalid calibration JSON')
    return MonoCalibration(**obj)


def _save_calibration(path, cal):
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({
        'reference_distance_m':cal.reference_distance_m,
        'reference_eye_spacing_px':cal.reference_eye_spacing_px
    },indent=2)+'\n',encoding='utf-8')


def _draw_robot(cv2, np, size, motion, planned, status):
    h=size
    panel=np.zeros((h,480,3),dtype=np.uint8)
    panel[:]=(29,29,29)
    ox,oz=35,int(h*.83)
    scale=min(650., (h-65)/.55)
    cfg=motion.cfg
    q=motion.q
    a=[q[0],q[0]+q[1],sum(q)]
    from math import cos,sin
    pts=[(0.,0.)]
    for length,ang in zip(cfg.lengths,a):
        pts.append((pts[-1][0]+length*cos(ang),pts[-1][1]+length*sin(ang)))
    xy=lambda p:(int(ox+p[0]*scale),int(oz-p[1]*scale))
    for p,t in zip(pts[:-1],pts[1:]):
        cv2.line(panel,xy(p),xy(t),(60,225,140),5)
    for p in pts:
        cv2.circle(panel,xy(p),6,(245,180,80),-1)
    if planned and planned.face_point_xz is not None:
        cv2.circle(panel,xy(planned.face_point_xz),8,(50,80,250),-1)
    cv2.putText(panel,'SIMULATION ONLY - MOTORS OFF',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.57,(0,210,255),2)
    cv2.putText(panel,status,(12,53),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
    cv2.putText(panel,'fixed-base camera assumption',(12,h-16),cv2.FONT_HERSHEY_SIMPLEX,.46,(180,180,180),1)
    return panel


def preview_pose(device=0, *, video=None, reference_distance_cm=None, calibration_path='calibration/face.json',
                 desired_distance_cm=45., horizontal_fov_deg=60., log_path=None, model_path=MODEL_PATH):
    try:
        import cv2
        import numpy as np
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError('Install: python -m pip install -r requirements-pose.txt') from exc
    if not Path(model_path).is_file():
        raise RuntimeError('Face Landmarker missing. Run python main.py --download-pose-model')
    if reference_distance_cm is not None and not 35 <= reference_distance_cm <= 120:
        raise ValueError('Reference distance 35..120 cm required')
    if not 35 <= desired_distance_cm <= 65:
        raise ValueError('Desired standoff 35..65 cm only')
    if not 30 <= horizontal_fov_deg <= 110:
        raise ValueError('Camera horizontal FOV must be 30..110 deg')
    if video is not None and not Path(video).is_file():
        raise FileNotFoundError(video)
    source = device if video is None else str(video)
    cap=cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('Cannot open camera or video')
    base=mp.tasks.BaseOptions
    vision=mp.tasks.vision
    opts=vision.FaceLandmarkerOptions(
        base_options=base(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
        min_face_detection_confidence=.65,
        min_face_presence_confidence=.65,
        min_tracking_confidence=.65,
        output_facial_transformation_matrixes=True
    )
    cal=_read_calibration(calibration_path)
    motion=MockMotion(ArmConfig())
    samples=deque(maxlen=35)
    csv_file=None
    writer=None
    t0=time.monotonic()
    prev_t=t0
    last_ts=0
    if log_path is not None:
        p=Path(log_path); p.parent.mkdir(parents=True,exist_ok=True)
        csv_file=p.open('w',newline='',encoding='utf-8')
        writer=csv.writer(csv_file)
        writer.writerow(['time_s','status','eye_px','x_norm','y_norm','pitch_deg','yaw_deg','roll_deg',
                         'estimated_distance_m','q1_deg','q2_deg','q3_deg'])
    try:
        with vision.FaceLandmarker.create_from_options(opts) as landmarker:
            while True:
                ok,frame=cap.read()
                if not ok: break
                height,width=frame.shape[:2]
                now=time.monotonic()
                dt=min(.05,max(.001,now-prev_t))
                prev_t=now
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                rgb=np.ascontiguousarray(rgb)
                mpimage=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                ts=max(int((now-t0)*1000),last_ts+1)
                last_ts=ts
                outcome=landmarker.detect_for_video(mpimage,ts)
                status='HOLD_NO_FACE'
                plan=None
                obs=None
                count=len(outcome.face_landmarks)
                try:
                    if count>1:
                        status='HOLD_MULTIPLE_FACES'
                    elif count==1:
                        if not outcome.facial_transformation_matrixes:
                            status='HOLD_NO_FACE_MATRIX'
                        else:
                            obs=observation_from_landmarks(
                                outcome.face_landmarks[0],outcome.facial_transformation_matrixes[0],width,height)
                            samples.append(obs.eye_spacing_px)
                            cv2.circle(frame,(int(obs.cx*width),int(obs.cy*height)),5,(0,200,255),-1)
                            if cal is None:
                                status='HOLD_NEEDS_CALIBRATION'
                            else:
                                focal=width/(2*tan(radians(horizontal_fov_deg)/2))
                                # Camera must be fixed to arm base; only a demo geometry.
                                rig=FixedCameraRig(width,height,focal,0.,.12)
                                follow=SimulatedFollower(cal,rig,desired_standoff_m=desired_distance_cm/100)
                                plan=follow.plan(obs,tuple(motion.q))
                                status=plan.status
                    if status=='TRACK_SIMULATION':
                        if motion.halted:
                            motion.reset()
                        motion.step(plan.joint_angles,dt)
                    else:
                        motion.emergency_stop()
                except ValueError:
                    status='HOLD_INVALID_GEOMETRY'
                    motion.emergency_stop()
                h1=f'FACE {count} | {status}'
                cv2.putText(frame,h1,(8,27),cv2.FONT_HERSHEY_SIMPLEX,.57,(0,255,255),2)
                if obs:
                    caption=f'Pitch={obs.pitch_deg:+.1f} Yaw={obs.yaw_deg:+.1f} Roll={obs.roll_deg:+.1f}'
                    cv2.putText(frame,caption,(8,55),cv2.FONT_HERSHEY_SIMPLEX,.52,(245,245,245),2)
                dist=plan.distance_m if plan else None
                cv2.putText(frame,f'approx depth: {dist:.2f}m' if dist is not None else 'depth: NOT CALIBRATED / UNRELIABLE',
                            (8,82),cv2.FONT_HERSHEY_SIMPLEX,.52,(230,230,230),2)
                cv2.putText(frame,'C: measure reference distance first | Q: quit',
                            (8,height-12),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
                if writer:
                    writer.writerow([round(now-t0,3),status,obs.eye_spacing_px if obs else '',
                                     obs.cx if obs else '',obs.cy if obs else '',
                                     obs.pitch_deg if obs else '',obs.yaw_deg if obs else '',
                                     obs.roll_deg if obs else '',dist if dist is not None else '',
                                     *(v*180/pi for v in motion.q)])
                panel=_draw_robot(cv2,np,height,motion,plan,status)
                combined=np.hstack([frame,panel])
                cv2.imshow('A_iFace 3D face pose + simulated 3R (NO MOTORS)',combined)
                key=cv2.waitKey(1)&0xff
                if key==ord('q'):
                    break
                if key==ord('c') and reference_distance_cm is not None and obs and len(samples)>=10 and abs(obs.yaw_deg)<10:
                    cal=MonoCalibration(reference_distance_cm/100,median(samples))
                    _save_calibration(calibration_path,cal)
                    print('Calibration saved at manually measured distance:',calibration_path)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()
