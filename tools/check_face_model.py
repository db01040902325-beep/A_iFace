"""Check YuNet ONNX loads and runs on synthetic data (not face accuracy)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aiface.vision import MODEL, verified

def main():
    if not verified(MODEL):
        raise RuntimeError('YuNet ONNX hash verification failed')
    import cv2
    import numpy as np
    detector=cv2.FaceDetectorYN.create(str(MODEL),'',(320,320),.80,.3,5000)
    image=np.zeros((320,320,3),dtype=np.uint8)
    _, faces=detector.detect(image)
    if faces is not None and (len(faces.shape)!=2 or faces.shape[1]<15):
        raise AssertionError('Invalid face detector output shape')
    print('YuNet ONNX load and inference: PASS (blank synthetic frame)')

if __name__=='__main__':main()
