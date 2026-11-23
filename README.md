# A_iFace — AI 얼굴 위치 기반 스마트폰 로봇암

3인 동역학 텀프로젝트: 침대 머리판 고정형 스마트폰 거치 로봇암입니다. 기존 3자유도 시뮬레이션을 유지하면서 6자유도(6R) 기구학 프로토타입을 개발 중입니다.

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


## v0.3: 3D 머리 방향 추정 + 보정된 거리 + 가상 로봇암 추적

공개 사전학습 **MediaPipe Face Landmarker**(478개 얼굴 특징점·4x4 얼굴 변환행렬)를 이용합니다. 카메라에서 얼굴 방향 Pitch/Yaw/Roll을 추정하고, **사용자가 실측한 기준 거리**에서 눈 사이 픽셀 간격을 저장하여 상대적인 거리를 근사합니다. **측정 장비 없이 절대 거리를 정확히 알아내는 기능이 아닙니다.**

설치:

    python -m pip install -r requirements-pose.txt
    python main.py --download-pose-model

먼저 카메라·모터 없이 3자유도 얼굴 추종 시뮬레이션:

    python main.py --follow-demo

실제 웹캠으로 얼굴 방향 및 가상 로봇암 확인:

    python main.py --pose-camera --device 0 --reference-distance-cm 65

1. 카메라에서 얼굴까지 실제 거리를 줄자로 **65cm** 측정한 다음 해당 위치에 서십시오.
2. 정면을 바라본 상태에서 잠시 기다린 뒤 C 키를 눌러 기준 눈 사이 픽셀 값을 저장하십시오.
3. 얼굴 위치·방향·추정 거리와 우측의 **가상 3자유도 로봇암**을 관찰하십시오.
4. 종료는 Q 키입니다.

영상 파일로 실행:

    python main.py --pose-video sample.mp4 --reference-distance-cm 65

수치 CSV 기록 옵션:

    --pose-log results/pose.csv

거리 보정 파일 calibration/face.json은 로컬에만 저장하고 Git에는 올리지 않습니다.

**안전 및 기계적 제약**
- 로봇 추적 시뮬레이션은 카메라가 로봇암 어깨 관절 옆에 **고정**되고 +X를 바라본다고 가정합니다. 스마트폰 거치대에 카메라를 부착할 경우 이 기하학은 틀리므로 실제 구동에 사용하면 안 됩니다.
- 3개 관절은 2D 수직 평면만 움직입니다. 머리가 좌우로 이동하거나 큰 Yaw/Roll이 생기면 추적을 중단합니다.
- 카메라 수평 화각 기본 60°는 **임시 가정**입니다. 실제 카메라 내부·외부 보정 없이 실제 위치 정확도를 보장하지 않습니다.
- 추정 거리 불확실, 3자유도 역기구학 불가능 시 HOLD 상태로 가상 정지합니다.
- 실제 모터 제어 없음. 사람 대신 더미로 물리적 비상정지·낙하 방지·충돌 방지 시험을 마친 뒤 실물 이동 제어를 검토하십시오.

모델 출처: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
개발 상세: docs/HEAD_POSE.md


## 6R 확장 기구학 (추가 프로토타입)

기존 3R의 테스트와 데모를 보존한 채 별도의 6축 정기구학(FK)과 역기구학(IK)을 구현했습니다.
축은 1번 Z(베이스 좌우), 2~4번 Y(어깨·팔꿈치·손목 상하),
5번 X(손목 롤), 6번 Z(도구 좌표계 말단 방향)입니다.

```bash
python -m aiface.arm6
python -m aiface.ik6
python -m unittest discover -s tests -v
```

현재 **6R 정기구학 및 기본 해석적 역기구학**만 구현되어 있습니다. 특이점 일부는 안전하게 거부하며, 6축 모터 구동·실시간 얼굴 추종·물리적 안전성은 미구현입니다.
기존 `main.py --pose-camera`는 3R 가상 로봇암을 계속 사용합니다.
기하학 가정과 단계별 계획: `docs/ARM6_DESIGN.md`.


## 6R 가상 관절 궤적 검증 (모터 OFF)

기본값: 관절별 최대 각속도 20°/s, 각가속도 80°/s².
기구학 모델에 6축 독립 관절 궤적 플래너를 추가했습니다.
예시 목표 3개를 차례로 이동·재지정·홈 복귀하면서 CSV로 기록합니다.

```bash
python -m aiface.motion6
python tools/check_motion6_limits.py results/motion6_demo.csv
```

CSV 파일은 `results/motion6_demo.csv`입니다. 6개 관절 위치 차분을 이용해
각속도·각가속도 제한을 검증합니다. 별도 단위 테스트에서 가상 비상정지,
정지 유지, 명시적 리셋, 관절 한계와 목표 변경도 검증합니다.

**실제 모터 출력은 전혀 없으며**, 현재 웹캠 화면의 가상 추적 기능은
여전히 3R만 사용합니다. 6R 자기충돌·인체 충돌·실제 모터 토크,
물리적인 전원 차단은 검증되지 않았습니다.


## 6R 웹캠 얼굴 추종 — 가상 로봇암

6R **별도 기능**이며 기존 3R의 \`--pose-camera\`에는 영향을 주지 않습니다.

\`\`\`bash
python main.py --pose-camera6 --device 0 --pose-log6 results/face6_live.csv
\`\`\`

- MediaPipe 사전학습 얼굴 감지/방향 추정, 기존 \`calibration/face.json\`의 거리 보정값 재사용
- 얼굴의 가로/세로 위치·단안 거리 추정으로 3D 좌표를 근사하고, 6R IK 및 속도 제한 모의 관절을 따라 이동
- 6개 관절값, 얼굴 위치 및 가상 말단 좌표를 CSV로 기록
- 실시간 화면 오른쪽에 **상면(X/Y)과 측면(X/Z)** 가상 로봇암 표시
- 얼굴 0명/2명 이상·거리 불확실·IK 도달 불가일 때 가상 관절 HOLD
- 복귀 시 5프레임 연속 정상 검출 후 가상 추적 재개
- \`SPACE\` 수동 가상 정지 잠금 / \`R\` 잠금 해제 / \`Q\` 종료
- 기준 거리에서 신규 보정 시에만 \`--reference-distance-cm 50\` 추가 후 \`C\`를 누릅니다.

네트워크·카메라 없이 재현 가능한 통합 회귀 시뮬레이션:
\`\`\`bash
python -m aiface.follow6
python -m unittest discover -s tests -v
\`\`\`

**주의:** 웹캠의 광축이 로봇 베이스 +X와 일치하고 0.12m 높이에 고정되었다는
가정에서만 유효한 가상 좌표입니다. 렌즈/카메라 위치, 얼굴의 회전 방향
부호 및 스마트폰 장착 방향은 교정되지 않았습니다. 화면 방향·Yaw/Roll 매핑은
설계용 예시일 뿐이며 실제 물리적 움직임의 검증이 아닙니다.
실제 거리센서, 6R 충돌 검사, 모터, 물리적 E-Stop은 미구현입니다.
