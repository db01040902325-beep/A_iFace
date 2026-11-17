"""Head-pose model smoke test without webcam or identifiable images."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from aiface.pose_camera import MODEL_PATH
import mediapipe as mp
import numpy as np

def main():
    assert MODEL_PATH.is_file(), 'Missing Face Landmarker model bundle'
    v=mp.tasks.vision
    opts=v.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=v.RunningMode.VIDEO,
        num_faces=2, output_facial_transformation_matrixes=True
    )
    with v.FaceLandmarker.create_from_options(opts) as landmarker:
        image=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.zeros((320,320,3),dtype=np.uint8))
        result=landmarker.detect_for_video(image,1)
        assert len(result.face_landmarks)==0, 'False detection on blank frame'
    print('MediaPipe Face Landmarker load and inference PASS (synthetic blank frame only).')

if __name__=='__main__':main()
