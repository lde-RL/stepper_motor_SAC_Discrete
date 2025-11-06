# Stepper Motor Reinforcement Learning Control

강화학습 기반 스테퍼 모터 제어 시스템 - Teensy 4.1 + Jetson Nano + ROS 2 Jazzy

## 프로젝트 개요

실환경에서 장기간(1주일+) 자율 학습하는 강화학습 시스템:
- **Teensy 4.1**: 1 kHz 실시간 제어, 안전 가드, 온보드 정책 추론
- **Jetson Nano/PC**: SAC-Discrete 학습, 목표 궤적 생성, 보상 계산
- **ROS 2 Jazzy**: 통신 프레임워크
- **핵심 특징**: 엔코더는 **훈련 중 보상 계산용으로만** 사용, **관측에는 미포함** → 배포 시 엔코더 제거 가능

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         PC / Jetson Nano                        │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ Trajectory Gen │→│ Training Node │←│ Replay Buffer (PER) │  │
│  │  (Sin waves)   │  │  SAC-Discrete │  │  100k transitions  │  │
│  └────────────────┘  └──────┬───────┘  └────────────────────┘  │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │ Policy Quantizer │                          │
│                    │   (INT8)         │                          │
│                    └────────┬────────┘                          │
└─────────────────────────────┼───────────────────────────────────┘
                              │ ROS 2 (UDP)
                              │ /ref, /action, /obs, /encoder
┌─────────────────────────────▼───────────────────────────────────┐
│                         Teensy 4.1                              │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  1 kHz Control │→│ Safety Guard │→│ 4-bit Switch Driver │  │
│  │  Loop (Timer)  │  │ (Illegal comb│  │  (16 actions)      │  │
│  └────────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌────────────────┐  ┌──────────────┐                          │
│  │ Policy Inference│  │ Encoder      │ (Training only)        │
│  │ (On-board NN)  │  │ (Reward calc)│                          │
│  └────────────────┘  └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

## 주요 기능

### 1. 관측 설계 (No Encoder in Observation!)
- **목표 정보**: `theta_ref_now`, `dtheta_ref_now`, `theta_ref_preview[K]`
- **행동 이력**: `last_action`, `action_hist[N]`
- **엔코더 미포함** → 배포 시 센서 의존성 없음

### 2. 훈련 모드 (가변 Sin 추적)
- 진폭, 주파수, 위상 랜덤화
- 미래 목표 preview 제공
- 에피소드 시작/종료 시 램프 적용

### 3. 배포 모드 (정적 목표)
- `dtheta_ref_now = 0.0`
- `theta_ref_preview[]` = 모두 동일한 값으로 채움
- 같은 정책 네트워크 사용 가능

### 4. 안전 시스템
- 불법 조합 방지 (Short-through)
- 과전류/과열/과전압 모니터링
- 통신 타임아웃 처리
- 안전 액션(0x00) 자동 적용

## 설치

### 1. 시스템 요구사항
- **Teensy 4.1** with Ethernet
- **PC/Jetson Nano** with Ubuntu 22.04
- **ROS 2 Jazzy** (또는 Humble/Iron)
- **Python 3.10+**
- **PyTorch 2.0+**

### 2. ROS 2 설치 (Ubuntu 22.04)
```bash
# ROS 2 Jazzy 설치 (공식 가이드 참조)
sudo apt update && sudo apt install ros-jazzy-desktop
source /opt/ros/jazzy/setup.bash
```

### 3. 프로젝트 빌드
```bash
cd ~/Desktop/stepper_motor/ros2_ws

# 커스텀 메시지 빌드
colcon build --packages-select stepper_rl_msgs
source install/setup.bash

# 제어 패키지 빌드
colcon build --packages-select stepper_rl_control
source install/setup.bash

# Python 의존성 설치
pip install -r ../requirements.txt
```

### 4. Teensy 펌웨어 업로드
```bash
# PlatformIO 사용 (권장)
cd teensy_firmware
pio run --target upload

# 또는 Arduino IDE 사용
# teensy_firmware/main_rl.cpp를 열고 업로드
```

## 사용 방법

### Phase 1: 통신 테스트
```bash
# Terminal 1: micro-ROS agent
ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888

# Terminal 2: Teensy 상태 모니터링
ros2 topic echo /health

# Terminal 3: 수동 목표 전송 테스트
ros2 topic pub /ref_target std_msgs/msg/Float32 "data: 1.57"
```

### Phase 2: 훈련 시작
```bash
# Terminal 1: micro-ROS agent
ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888

# Terminal 2: 훈련 노드 실행
ros2 run stepper_rl_control train

# Terminal 3: Tensorboard 모니터링
tensorboard --logdir logs/
```

**훈련 시간 설정**:
- 기본 설정: 무한 실행 (Ctrl+C로 중단)
- 에피소드 제한: `config/training_config.yaml`에서 `max_episodes` 추가
- 시간 제한: cron job 또는 systemd timer로 자동 중단 설정

```bash
# 예: 7일(168시간) 후 자동 중단
timeout 168h ros2 run stepper_rl_control train
```

### Phase 3: 정책 평가
```bash
# 최신 모델 테스트 (정적 목표)
python3 -c "
from stepper_rl_control.training_node import StepperRLTrainer
# 평가 스크립트 실행
"
```

### Phase 4: 배포
```bash
# 최적 정책을 Teensy로 전송
ros2 run stepper_rl_control deploy --policy models/best_policy_quant.bin
```

## 설정 파일

### `config/training_config.yaml`
주요 하이퍼파라미터:
```yaml
rl:
  hidden_dim: 64
  learning_rate: 0.0003
  batch_size: 256
  buffer_capacity: 100000
  preview_length: 32      # K (미래 목표 스텝 수)
  action_hist_length: 16  # N (과거 액션 히스토리)
```

### `config/deployment_config.yaml`
배포 설정:
```yaml
target:
  mode: "static"
  static_angle: 1.57  # 90도

policy:
  use_onboard_inference: true
```

## 파일 구조

```
stepper_motor/
├── ros2_ws/
│   └── src/
│       ├── stepper_rl_msgs/          # 커스텀 ROS 메시지
│       │   └── msg/
│       │       ├── RefTarget.msg
│       │       ├── Observation.msg
│       │       ├── Action.msg
│       │       └── ...
│       └── stepper_rl_control/       # 제어 패키지
│           └── stepper_rl_control/
│               ├── sac_discrete.py         # SAC 알고리즘
│               ├── replay_buffer.py        # PER 버퍼
│               ├── policy_quantization.py  # INT8 양자화
│               ├── training_node.py        # 훈련 노드
│               └── monitoring.py           # 시각화
├── teensy_firmware/
│   └── main_rl.cpp                   # Teensy 펌웨어
├── config/
│   ├── training_config.yaml
│   └── deployment_config.yaml
├── models/                           # 저장된 정책
├── logs/                             # 학습 로그
└── README.md
```

## 핵심 알고리즘: SAC-Discrete

- **Policy**: `π(a|s)` - Categorical distribution over 16 actions
- **Q-networks**: Twin Q-functions with target networks
- **Entropy regularization**: `J = E[R + αH(π)]`
- **Automatic temperature tuning**: α 자동 조정

**학습 업데이트**:
1. 리플레이 버퍼에서 배치 샘플링 (PER)
2. Q-네트워크 업데이트 (TD error)
3. 정책 업데이트 (entropy-regularized objective)
4. 온도(α) 업데이트 (target entropy)
5. 타깃 네트워크 소프트 업데이트

## 문제 해결

### 통신 안 됨
```bash
# Teensy IP 확인
ping 192.168.1.10

# micro-ROS agent 재시작
ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888 -v6
```

### 학습이 안정적이지 않음
- `learning_rate`를 1e-4로 낮추기
- `preview_length`를 64로 증가
- `reward.error_weight`를 2.0으로 증가

### 엔코더 값이 이상함
- 엔코더 배선 확인 (A, B 채널)
- `CPR` 상수 확인 (main_rl.cpp)
- 정/역회전 시 count 증가/감소 확인

### Teensy가 멈춤
- 안전 가드 트리거 확인 (`/health` 토픽)
- 전류/온도 센서 캘리브레이션
- `safety` 설정값 조정

## 성능 최적화

### 학습 속도 향상
- GPU 사용 (CUDA)
- `train_frequency` 증가 (200 → 500 Hz)
- `batch_size` 증가 (256 → 512)

### 메모리 절약
- `buffer_capacity` 감소 (100k → 50k)
- `use_per: false` (PER 비활성화)

### 추적 성능 향상
- `preview_length` 증가 (32 → 64)
- `reward.derivative_weight` 조정
- 목표 궤적 커리큘럼 학습 (느린 주파수 → 빠른 주파수)

## 참고 문헌

- **SAC**: Haarnoja et al. (2018) "Soft Actor-Critic"
- **SAC-Discrete**: Christodoulou (2019) "Soft Actor-Critic for Discrete Action Settings"
- **PER**: Schaul et al. (2016) "Prioritized Experience Replay"

## 라이센스

MIT License

## 문의

Issues: GitHub repository
