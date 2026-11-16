# v0.2 — YuNet 얼굴 검출 및 2D 추적

## 모델 및 데이터

- 모델: OpenCV Zoo YuNet, face_detection_yunet_2023mar.onnx
- 사전학습 데이터: WIDER FACE (COCO는 얼굴 전용 데이터셋이 아님)
- 원본: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- 모델 라이선스: 해당 폴더의 MIT LICENSE
- 모델 SHA-256: 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
- 별도 라벨링이나 재학습 필요 없음

## 실행

    python -m pip install -r requirements-face.txt
    python main.py --download-model
    python main.py --camera --device 0

녹화된 영상에서 실행:

    python main.py --video sample.mp4

원본 영상은 저장하지 않고 추적 수치만 CSV로 저장하려면:

    python main.py --camera --face-log results/face_tracking.csv

q 키로 종료. 카메라 접근 권한과 장치 번호를 확인하십시오.

## 검출 내용

- 얼굴 사각형, 정규화 중심좌표(X/Y 0~1), 눈 선의 2D 기울기(roll °)
- 0명/2명 이상: NO_FACE/MULTIPLE_FACES로 표시하며 추적 좌표 차단
- 낮은 신뢰도 또는 갑작스런 위치 변화 시 추적 좌표 차단
- 원본 영상의 외부 업로드 및 자동 저장 기능 없음
- 실제 얼굴 거리(cm), 3D yaw/pitch, 얼굴 신원 식별 기능 없음
- 모터는 움직이지 않음

## 다음 단계

1. ToF 거리 센서(또는 보정된 스테레오) 실측값 확보
2. 얼굴 3D 자세 추정과 카메라-로봇 좌표계 보정
3. 목표 좌표를 3R 역기구학으로 변환
4. 더미 시험 및 기계·전자 안전성 검증 후 모터 제어
