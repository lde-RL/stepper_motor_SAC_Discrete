# Quick Start Guide

30분 안에 시작하기

## 사전 준비

✅ Teensy 4.1 + Ethernet 모듈
✅ Stepper motor + 4-bit switch driver
✅ Encoder (훈련 중에만 필요)
✅ Ubuntu 22.04 + ROS 2 Jazzy
✅ Python 3.10+

## 5단계로 시작하기

### Step 1: 하드웨어 연결 (5분)

```
Teensy 4.1:
├─ Pin 2-5    → 4-bit Switch Driver (IN0-IN3)
├─ Pin 25-26  → Encoder (A, B channels)
├─ Pin A0     → Current sensing (optional)
└─ Ethernet   → 192.168.1.10 (static IP)

PC/Jetson:
└─ Ethernet   → 192.168.1.113 (agent IP)
```

### Step 2: 소프트웨어 설치 (10분)

```bash
# ROS 2 Jazzy (이미 설치되어 있다고 가정)
source /opt/ros/jazzy/setup.bash

# 프로젝트 클론 (이미 완료)
cd ~/Desktop/stepper_motor

# 의존성 설치
pip install torch numpy pyyaml matplotlib tensorboard

# ROS 패키지 빌드
cd ros2_ws
colcon build
source install/setup.bash
```

### Step 3: Teensy 펌웨어 업로드 (5분)

```bash
# Arduino IDE 또는 PlatformIO 사용
# teensy_firmware/main_rl.cpp 열기

# ⚠️ 설정 확인:
# - agent_ip(192,168,1,113)
# - local_ip(192,168,1,10)
# - POLE_PAIRS (모터에 맞게 조정)

# 업로드 후 Serial Monitor 확인
# "Teensy RL Control Node Ready" 메시지 확인
```

### Step 4: 통신 테스트 (5분)

```bash
# Terminal 1: micro-ROS agent 실행
ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888

# Terminal 2: 토픽 확인
ros2 topic list
# 출력 예:
# /Hello_world
# /health
# /encoder_angle
# /ref_target
# /action_cmd

# Terminal 3: 엔코더 데이터 확인
ros2 topic echo /encoder_angle
# 모터를 손으로 돌려서 값 변화 확인
```

### Step 5: 훈련 시작! (5분 설정 + 학습)

```bash
# Terminal 1: micro-ROS agent (이미 실행 중)

# Terminal 2: 훈련 시작
cd ~/Desktop/stepper_motor
source ros2_ws/install/setup.bash
ros2 run stepper_rl_control train

# Terminal 3: Tensorboard (optional)
tensorboard --logdir logs/
# 브라우저: http://localhost:6006

# 모터가 움직이기 시작하면 성공! 🎉
```

## 훈련 진행 확인

터미널 출력:
```
[INFO] StepperRLTrainer initialized
[INFO] Observation dimension: 50
[INFO] Episode 1 ended | Steps: 2543 | Return: -12.34
[INFO] Episode 2 ended | Steps: 3012 | Return: -10.56
...
[INFO] Step 10000 | Q-loss: 0.0234 | Policy-loss: 0.0156 | Alpha: 0.15
```

## 학습 시간 설정

### 1주일 학습 후 자동 중단
```bash
timeout 168h ros2 run stepper_rl_control train
```

### N 에피소드 후 중단
`config/training_config.yaml` 수정:
```yaml
episode:
  max_episodes: 10000  # 추가
```

## 모델 저장 위치

```
models/
├── model_ep10_20250106_143022.pt          # PyTorch 모델
├── model_ep10_20250106_143022_quant.bin   # INT8 양자화 (Teensy용)
├── model_ep20_...
└── ...
```

## 다음 단계

1. **모니터링**: Tensorboard에서 학습 곡선 확인
2. **하이퍼파라미터 튜닝**: `config/training_config.yaml` 조정
3. **정책 평가**: 최적 모델로 테스트
4. **배포**: 엔코더 제거 후 실제 사용

## 문제가 생기면?

### 모터가 안 움직임
- [ ] Teensy Serial Monitor에서 에러 메시지 확인
- [ ] `/health` 토픽에서 fault_flags 확인
- [ ] 전원 공급 확인 (12V)
- [ ] 4-bit switch driver 연결 확인

### 학습이 발산함
- [ ] `learning_rate`를 1e-4로 낮추기
- [ ] `reward.error_weight` 증가
- [ ] `traj.max_frequency` 감소 (쉬운 목표부터 학습)

### 통신 끊김
- [ ] 네트워크 연결 확인 (`ping 192.168.1.10`)
- [ ] micro-ROS agent 재시작
- [ ] Teensy 리셋

## 도움말

README.md 전체 문서 참조
GitHub Issues로 문의

---
**Happy Learning! 🤖🎓**
