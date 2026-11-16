"""YuNet OpenCV face detection (WIDER FACE pretrained), webcam/video preview only.
No images are stored or transmitted, and no motor hardware is accessed.
"""
import csv
import hashlib
from pathlib import Path
import time
from urllib.request import Request, urlopen
from .face_tracking import FaceTracker, parse_yunet

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models' / 'face_detection_yunet_2023mar.onnx'
MODEL_URL = 'https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx'
MODEL_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'

def verified(path=MODEL):
    p = Path(path)
    return p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == MODEL_SHA256

def install_model(path=MODEL):
    """User-triggered HTTPS download with pinned SHA-256, not auto-downloaded."""
    p = Path(path)
    if verified(p):
        return p
    req = Request(MODEL_URL, headers={'User-Agent': 'A_iFace-FaceModel/0.2'})
    with urlopen(req, timeout=30) as response:
        data = response.read(1024*1024+1)
    if len(data) > 1024*1024 or hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError('Invalid or untrusted model download; aborted')
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp')
    try:
        temp.write_bytes(data)
        temp.replace(p)
    finally:
        temp.unlink(missing_ok=True)
    return p

def preview_camera(device=0, *, video=None, log_path=None, model_path=MODEL):
    if not verified(model_path):
        raise RuntimeError('YuNet model missing/invalid; run: python main.py --download-model')
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('Install opencv-python: python -m pip install -r requirements-face.txt') from exc
    if not hasattr(cv2, 'FaceDetectorYN'):
        raise RuntimeError('OpenCV FaceDetectorYN unavailable: update opencv-python')
    source = int(device) if video is None else str(video)
    if video is not None and not Path(video).is_file():
        raise FileNotFoundError(video)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError('Cannot open camera/video source')
    detector = cv2.FaceDetectorYN.create(str(model_path), '', (320,320), .80, .3, 5000)
    tracker = FaceTracker()
    start = time.monotonic()
    file = None
    writer = None
    try:
        if log_path is not None:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            file = p.open('w', newline='', encoding='utf-8')
            writer = csv.writer(file)
            writer.writerow(['time_s','status','faces','x_norm','y_norm','roll_deg','confidence'])
        print('YuNet face position + 2D roll only. NO metric distance / motor control. Press q.')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height,width = frame.shape[:2]
            detector.setInputSize((width,height))
            _, raw = detector.detect(frame)
            faces = parse_yunet(raw)
            now = time.monotonic()
            result = tracker.update(faces,width,height,now)
            o = result.observation
            if o:
                x,y,w,h = (int(n) for n in o.bbox)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.circle(frame,(int(o.cx*width),int(o.cy*height)),5,(0,255,255),-1)
                cv2.putText(frame,f'X={o.cx:.3f} Y={o.cy:.3f} ROLL={o.roll_deg:.1f}',
                            (8,min(54,height-5)),cv2.FONT_HERSHEY_SIMPLEX,.57,(255,255,255),2)
            cv2.putText(frame,f'{result.status} | faces {result.faces_seen} | MOTORS OFF',
                        (8,26),cv2.FONT_HERSHEY_SIMPLEX,.60,(0,230,255),2)
            if writer:
                writer.writerow([round(now-start,3),result.status,result.faces_seen,
                                 f'{o.cx:.5f}' if o else '',f'{o.cy:.5f}' if o else '',
                                 f'{o.roll_deg:.2f}' if o else '',f'{o.confidence:.3f}' if o else ''])
            cv2.imshow('A_iFace Face Detector - Motors Disabled',frame)
            if cv2.waitKey(1)&0xff==ord('q'):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if file is not None:
            file.close()
