"""Live webcam + six-joint *virtual* face following preview.

NO commands are sent to actuators. Fixed-base camera and rough FOV required.
Press SPACE to latch virtual stop, R to release (after five good observations),
C to calibrate at measured reference distance, Q to quit.
"""
import csv
from collections import Counter, deque
from math import degrees, pi, radians, tan
from pathlib import Path
from statistics import median
import time

from .arm6 import forward6
from .follow6 import FaceFollower6, SimulatedFace6Controller
from .head_pose import FixedCameraRig, observation_from_landmarks, MonoCalibration
from .pose_camera import MODEL_PATH, _read_calibration, _save_calibration


def _panel(cv2,np,height,controller,plan,status):
    width=510
    panel=np.full((height,width,3),25,dtype=np.uint8)
    pose=forward6(tuple(controller.motion.q))
    pts=pose.points_m
    face=plan.face_xyz_m
    # 3D location is presented as two complementary fixed projections.
    px_per_m = min(380.,max(180.,height*.67))
    ztop=max(45,int(height*.28))
    zside=max(60,int(height*.76))
    origin_top=(50,ztop)
    origin_side=(50,zside)
    top=lambda p:(int(origin_top[0]+p[0]*px_per_m),
                   int(origin_top[1]-p[1]*px_per_m))
    side=lambda p:(int(origin_side[0]+p[0]*px_per_m),
                    int(origin_side[1]-(p[2]-.12)*px_per_m))
    for projection in (top,side):
        for a,b in zip(pts[:-1],pts[1:]):
            cv2.line(panel,projection(a),projection(b),(70,215,140),4)
        for point in pts:
            cv2.circle(panel,projection(point),4,(255,210,100),-1)
        if face is not None:
            cv2.circle(panel,projection(face),6,(70,80,235),-1)
    cv2.putText(panel,'6R SIMULATION - MOTORS OFF',(8,21),
                cv2.FONT_HERSHEY_SIMPLEX,.55,(25,220,255),2)
    cv2.putText(panel,status[:38],(8,43),cv2.FONT_HERSHEY_SIMPLEX,.42,
                (235,235,235),1)
    cv2.putText(panel,'TOP (X/Y) - lateral motion', (8,max(61,ztop-110)),
                cv2.FONT_HERSHEY_SIMPLEX,.43,(210,210,210),1)
    cv2.putText(panel,'SIDE (X/Z) - height and depth', (8,max(65,zside-110)),
                cv2.FONT_HERSHEY_SIMPLEX,.43,(210,210,210),1)
    qdeg=tuple(round(degrees(q),1) for q in controller.motion.q)
    ylabels=(height-55,height-36)
    cv2.putText(panel,f'J1..J3: {qdeg[0]:+.1f}, {qdeg[1]:+.1f}, {qdeg[2]:+.1f}',
                (8,ylabels[0]),cv2.FONT_HERSHEY_SIMPLEX,.47,(235,235,235),1)
    cv2.putText(panel,f'J4..J6: {qdeg[3]:+.1f}, {qdeg[4]:+.1f}, {qdeg[5]:+.1f}',
                (8,ylabels[1]),cv2.FONT_HERSHEY_SIMPLEX,.47,(235,235,235),1)
    cv2.putText(panel,'SPACE: STOP | R: reset | C: calibrate | Q: quit',
                (8,height-10),cv2.FONT_HERSHEY_SIMPLEX,.4,(25,220,255),1)
    return panel,pose


def preview_pose6(device=0, *, video=None, reference_distance_cm=None,
                  calibration_path='calibration/face.json',
                  desired_distance_cm=45., horizontal_fov_deg=60.,
                  log_path=None, model_path=MODEL_PATH):
    try:
        import cv2
        import numpy as np
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError('Run: python -m pip install -r requirements-pose.txt') from exc

    if not Path(model_path).is_file():
        raise RuntimeError('Face Landmarker missing. Use --download-pose-model')
    if reference_distance_cm is not None and not 35 <= reference_distance_cm <= 120:
        raise ValueError('Reference distance must be 35..120cm')
    if not 35 <= desired_distance_cm <= 65:
        raise ValueError('Desired distance must be 35..65cm')
    if not 30 <= horizontal_fov_deg <= 110:
        raise ValueError('Unsupported horizontal FOV')
    if video is not None and not Path(video).is_file():
        raise FileNotFoundError(video)
    cap=cv2.VideoCapture(device if video is None else str(video))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('Cannot open video/webcam')
    vision=mp.tasks.vision
    opts=vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
        min_face_detection_confidence=.65,
        min_face_presence_confidence=.65,
        min_tracking_confidence=.65,
        output_facial_transformation_matrixes=True,
    )
    cal=_read_calibration(calibration_path)
    controller=None
    last=0
    recent=deque(maxlen=35)
    previous_t=time.monotonic()
    first_t=previous_t
    counts=Counter()
    csvfile=None
    writer=None
    if log_path is not None:
        out=Path(log_path)
        out.parent.mkdir(parents=True,exist_ok=True)
        csvfile=out.open('w',newline='',encoding='utf-8-sig')
        writer=csv.writer(csvfile)
        writer.writerow([
            'time_s','status','faces','estimated_distance_m',
            'pitch_deg','yaw_deg','roll_deg','face_x_m','face_y_m','face_z_m',
            'target_x_m','target_y_m','target_z_m',
            'actual_x_m','actual_y_m','actual_z_m',
            *[f'q{j}_deg' for j in range(1,7)],
        ])
    try:
        with vision.FaceLandmarker.create_from_options(opts) as landmark:
            while True:
                success,frame=cap.read()
                if not success:
                    break
                h,w=frame.shape[:2]
                now=time.monotonic()
                dt=max(.001,min(.05,now-previous_t))
                previous_t=now
                fpx=w/(2*tan(radians(horizontal_fov_deg)/2))
                rig=FixedCameraRig(w,h,fpx,0.,.12)
                if controller is None:
                    controller=SimulatedFace6Controller(
                        FaceFollower6(cal,rig,desired_standoff_m=desired_distance_cm/100)
                    )
                # Camera frame dimensions may change during capture.
                controller.follower.rig=rig
                rgb=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                ms=max(int((now-first_t)*1000),last+1)
                last=ms
                detected=landmark.detect_for_video(image,ms)
                faces=len(detected.face_landmarks)
                observation=None
                if faces==1 and detected.facial_transformation_matrixes:
                    try:
                        observation=observation_from_landmarks(
                            detected.face_landmarks[0],
                            detected.facial_transformation_matrixes[0],w,h
                        )
                        recent.append(observation.eye_spacing_px)
                        cv2.circle(frame,(int(observation.cx*w),int(observation.cy*h)),
                                   5,(0,225,255),-1)
                    except ValueError:
                        observation=None
                plan=controller.update(observation,faces=faces,dt=dt)
                status=plan.status
                counts[status]+=1
                header=f'FACE {faces} | {status}'
                cv2.putText(frame,header[:60],(8,26),
                            cv2.FONT_HERSHEY_SIMPLEX,.55,(0,225,255),2)
                if observation is not None:
                    angles=(observation.pitch_deg,observation.yaw_deg,
                            observation.roll_deg)
                    cv2.putText(frame,'Pitch %+5.1f Yaw %+5.1f Roll %+5.1f'%angles,
                                (8,54),cv2.FONT_HERSHEY_SIMPLEX,.48,(235,235,235),1)
                estimated=('depth approx %.2fm'%plan.distance_m
                           if plan.distance_m is not None else 'depth: UNAVAILABLE')
                cv2.putText(frame,estimated,(8,79),
                            cv2.FONT_HERSHEY_SIMPLEX,.53,(245,245,245),1)
                if controller.manual_latched:
                    cv2.putText(frame,'MANUAL STOP LATCHED: press R after inspection',
                                (8,105),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),2)
                panel, pose=_panel(cv2,np,h,controller,plan,status)
                if writer is not None:
                    face=plan.face_xyz_m or ('','','')
                    target=plan.target_xyz_m or ('','','')
                    writer.writerow([
                        round(now-first_t,4),status,faces,
                        plan.distance_m if plan.distance_m is not None else '',
                        observation.pitch_deg if observation else '',
                        observation.yaw_deg if observation else '',
                        observation.roll_deg if observation else '',
                        *face,*target,*pose.position_m,
                        *(degrees(q) for q in controller.motion.q)
                    ])
                cv2.imshow('A_iFace 6R FACE FOLLOW - SIMULATION ONLY',
                           np.hstack((frame,panel)))
                key=cv2.waitKey(1)&0xff
                if key in (ord('q'),27):
                    break
                if key==32:
                    controller.manual_stop()
                elif key==ord('r'):
                    controller.reset_manual()
                elif (key==ord('c') and reference_distance_cm is not None
                      and observation is not None and len(recent)>=10
                      and abs(observation.yaw_deg)<10):
                    cal=MonoCalibration(reference_distance_cm/100,median(recent))
                    _save_calibration(calibration_path,cal)
                    controller.follower.calibration=cal
                    controller.motion.emergency_stop()
                    controller.valid_streak=0
                    print('6R preview calibration saved:',calibration_path)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if csvfile is not None:
            csvfile.close()
    print('6R webcam tracking state counts:',dict(counts))
    print('NO ACTUATORS WERE COMMANDED')
