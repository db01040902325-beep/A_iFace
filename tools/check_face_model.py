"""Smoke-test real OpenCV YuNet ONNX on a synthetic frame (not accuracy)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aiface.vision import MODEL, verified


def main():
    if not verified(MODEL):
        raise RuntimeError('ONNX model SHA-256 verification failed')
    import cv2
    import numpy as np
    detector = cv2.FaceDetectorYN.create(str(MODEL), '', (320,320), .80, .3, 5000)
    image = np.zeros((320,320,3),dtype=np.uint8)
    _, faces = detector.detect(image)
    if faces is not None and (len(faces.shape) != 2 or faces.shape[1] < 15):
        raise AssertionError('Invalid detector output shape')
    print('YuNet ONNX load/inference: PASS (synthetic frame, no accuracy claim)')


if __name__=='__main__':main()
