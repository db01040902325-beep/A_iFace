# A_iFace — AI 얼굴 위치 기반 스마트폰 로봇암

3인 동역학 텀프로젝트: 침대 머리판 고정형 **3자유도(어깨·팔꿈치·손목)** 스마트폰 거치 로봇암입니다.

## 구현 현황 (v0.1)
- 3R 평면 로봇암 정기구학 및 역기구학, 관절각 제한
- 3개 링크 및 스마트폰 질량·관성 기반 역동역학: M(q)q̈ + C(q,q̇)q̇ + G(q) = τ
- 속도·가속도 제한, 모의 비상정지, **실제 모터 출력 없는** 궤적 시뮬레이션
- CSV 실험 기록, Python 유닛 테스트, GitHub Actions 테스트
- 노트북 카메라 사전학습 얼굴 검출 시연 (OpenCV 선택 설치, 신원 식별 없음)
- ESP32 모든 모터 명령 거부 안전 프리뷰

**미구현:** 실제 얼굴-휴대폰 거리 측정, 3D 얼굴 방향 추정, 3D 좌표 변환, AI 기반 실제 자동 추적, 실물 모터 구동, 장애물 감지, 노트북 없는 독립 실행.

## 실행
Python 3.10 이상. 프로젝트 루트에서 실행합니다.

    python main.py --demo
    python -m unittest discover -s tests -v

출력: results/demo.csv (시간별 관절각·스마트폰 위치·필요 토크)

카메라 미리보기 (모터 제어 없음):

    python -m pip install opencv-python
    python main.py --camera

q 키로 종료. 이 얼굴 검출기는 신원, 실제 얼굴 거리, 3D 머리 자세를 추정하지 않습니다.

## 파일 구성
- aiface/core.py — 기구학, 질량행렬과 역동역학, 안전 모의제어
- aiface/vision.py — 얼굴 검출 영상 미리보기
- main.py — 시뮬레이션과 CSV 기록
- tests/test_core.py — 역기구학·동역학·제어 한계 자동 테스트
- firmware/ — ESP32 안전 프리뷰 (모터 미구현)
- docs/ — 구현계획서 사양 및 실험계획

## 설계 제한 및 안전 요구
- 어깨·팔꿈치·손목은 모두 같은 수직 2D 평면 내 회전하는 것으로 모델링합니다. **좌우 방향 추적은 불가능**하며 별도 자유도가 필요합니다.
- 링크 치수와 스마트폰 무게는 임시 가정이며 실제 Inventor 설계와 실측 후 교정해야 합니다.
- 얼굴의 화면상 좌표만으로 실제 거리를 알 수 없으므로, 거리센서 및 카메라-로봇 보정 전에는 실제 자동 구동을 허용하지 않습니다.
- 사용자 얼굴 근처의 구동 장치이므로 하중 낙하 방지, 전원 차단 가능한 물리적 비상정지, 기계적 회전 스토퍼, 충분한 안전거리 검증이 필요합니다. 초기 실험은 더미만 사용하십시오.

## 향후 개발
v0.2: 카메라+거리센서 융합 및 좌표 보정 → v0.3: 안전한 실제 관절 구동 → v0.4: AI 얼굴 추적 통합 → v1.0: 노트북 없는 SBC+ESP32 실행.

학교 시제품으로, 사용자 신원 식별이나 의료기기용이 아닙니다.


## v0.2: 공개 사전학습 모델을 활용한 얼굴 검출

OpenCV Zoo YuNet(WIDER FACE 사전학습)을 이용하며, 별도 수동 라벨링 없이 얼굴 중심(X/Y)과 눈 기울기(2D roll)를 표시합니다.

실행:

    python -m pip install -r requirements-face.txt
    python main.py --download-model
    python main.py --camera --device 0

모의 로봇암: python main.py --demo
자동 테스트: python -m unittest discover -s tests -v

녹화된 영상 입력: python main.py --video sample.mp4
선택적 CSV 기록: python main.py --camera --face-log results/face_tracking.csv

얼굴이 없거나 2명 이상이면 추적 결과를 차단합니다. q 키로 종료합니다.

**현재 실제 거리(cm), 3D 머리 방향, 실제 모터 추종은 구현하지 않았습니다.**
모델은 명시적으로 다운로드할 때만 설치하고 SHA-256을 확인합니다.
원본 얼굴 영상은 저장하거나 외부로 전송하지 않습니다.

자세한 내용: docs/FACE_DETECTION.md
