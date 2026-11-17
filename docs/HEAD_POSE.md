# v0.3 — MediaPipe 머리 방향과 가상 3R 추적

## 사전학습 모델

- Google MediaPipe Face Landmarker Tasks Python
- 모델 URL: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- 478개 얼굴 특징점 및 canonical-to-observed 4x4 얼굴 변환행렬
- 기존 OpenCV YuNet v0.2는 유지하며 MediaPipe는 별도 선택 실행
- 재학습/수동 라벨링 없이 사용
- 모델 다운로드는 명시적으로만 실행; 버전 URL과 SHA-256(64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff)을 검증

## 실행

    python -m pip install -r requirements-pose.txt
    python main.py --download-pose-model
    python main.py --follow-demo
    python main.py --pose-camera --reference-distance-cm 65

C 키: 사용자가 줄자로 정확하게 65cm를 맞춘 상태에서 보정합니다.
Q 키: 종료. 실제 얼굴 영상은 저장하지 않습니다. 별도 옵션으로 수치 CSV만 저장 가능.

## 좌표 및 한계

- Euler Z-Y-X: Pitch=카메라 X축, Yaw=Y축, Roll=Z축 방향 추정
- 실제 고개 끄덕임의 부호는 카메라 설치 방향에 따라 교정해야 합니다.
- 추정 거리 = 기준 거리 × (기준 눈 사이 픽셀 / 현재 눈 사이 픽셀)
- 깊이는 정확한 실측값이 아니며, 머리가 회전하면 눈 간격과 추정 깊이가 달라집니다. Yaw 22° 이상은 차단합니다.
- 고정 웹캠을 로봇암 어깨 좌표계 기준에 설치했고 광축이 +X라고 가정합니다.
- 스마트폰 거치대에 움직이는 카메라를 달면 카메라 변환행렬을 FK로 갱신해야 합니다. 이 기능은 아직 없습니다.
- 3R 로봇암은 2D 평면에서만 움직이고 좌우 이동/크게 회전한 얼굴은 따라가지 못합니다.
- 가상 추적은 실물 정확도/안전 성능을 입증하지 않습니다.

## 추적 상태

- TRACK_SIMULATION: 가상 관절 목표 생성
- HOLD_NO_FACE: 얼굴 없음
- HOLD_MULTIPLE_FACES: 얼굴 2개 이상
- HOLD_NEEDS_CALIBRATION: 기준 거리 보정 전
- HOLD_UNSUPPORTED_HEAD_POSE: 머리의 큰 각도
- HOLD_UNRELIABLE_DEPTH: 거리값 불확실/범위 초과
- HOLD_OUTSIDE_2D_PLANE: 좌우 이동으로 3R 평면 벗어남
- HOLD_IK_UNREACHABLE: 관절 제한 또는 작업공간 초과
- HOLD_SAFETY_ENVELOPE: 최소 안전거리/작업구간 실패
- HOLD_INVALID_GEOMETRY: 잘못된 입력

## 다음 단계

1. 실물 카메라 내부 화각과 어깨 좌표계 외부 변환 보정
2. ToF 거리센서 실측값과 단안 추정값 비교
3. 스마트폰 거치부 장착 카메라를 사용한다면 동적 외부 변환 계산
4. 더미 기반 물리 안전 검증 후 실물 모터 구현
