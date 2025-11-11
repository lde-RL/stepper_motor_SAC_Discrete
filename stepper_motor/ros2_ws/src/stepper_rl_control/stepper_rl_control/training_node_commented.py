"""
===================================================================================
PC 훈련 노드 (Training Node) - 전체 시스템의 두뇌!
===================================================================================

이 파일의 역할:
  1. 목표 궤적 생성 (가변 sin 파형)
  2. Teensy와 ROS 통신 (목표/액션 송신, 엔코더/상태 수신)
  3. 보상 계산 (엔코더 사용 - 훈련 전용!)
  4. 에피소드 관리 (시작/종료, 리셋 없음)
  5. 강화학습 (SAC-Discrete 훈련)
  6. 모델 저장 및 배포

핵심 설계:
  - 1 kHz 제어 루프: 매 1ms마다 목표 생성 → 액션 선택 → 보상 계산
  - 비동기 학습: 제어와 독립적으로 200 Hz로 네트워크 학습
  - 엔코더는 보상 계산에만 사용, 관측에는 미포함!

===================================================================================
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import json

# ROS 메시지 (나중에 custom message로 교체)
from std_msgs.msg import Float32, UInt8, Int32

# 강화학습 구성요소
from .sac_discrete import SACDiscrete
from .replay_buffer import PrioritizedReplayBuffer, SimpleReplayBuffer
from .policy_quantization import QuantizedPolicy


# ===================================================================================
# 1. 궤적 생성기 (TrajectoryGenerator)
# ===================================================================================
class TrajectoryGenerator:
    """
    가변 Sin 파형 목표 궤적 생성

    역할:
      - 훈련 시: 랜덤 진폭/주파수/위상의 sin 파형 생성
      - 정책이 다양한 궤적을 학습하도록 함
      - 미래 K개 스텝의 목표도 제공 (preview)

    왜 Sin 파형?
      - 실제 모터 제어에서 흔한 궤적
      - 속도/가속도가 연속적 (부드러움)
      - 주기적 → 에피소드 여러 번 반복 학습 가능

    배포 시:
      - sin 파형 대신 정적 목표 사용
      - dtheta_ref = 0
      - preview = 모두 동일한 값
    """

    def __init__(
        self,
        min_amplitude: float = 0.5,      # 최소 진폭 (rad) - 작은 움직임
        max_amplitude: float = 2.0,      # 최대 진폭 (rad) - 큰 움직임
        min_frequency: float = 0.5,      # 최소 주파수 (Hz) - 느린 진동
        max_frequency: float = 3.0,      # 최대 주파수 (Hz) - 빠른 진동
        ramp_duration: float = 0.1       # 램프 시간 (sec) - 부드러운 시작/끝
    ):
        """
        궤적 생성기 초기화

        파라미터 설명:
          - amplitude: 진폭이 클수록 모터가 크게 움직임
            예: 0.5 rad ≈ 28도, 2.0 rad ≈ 115도

          - frequency: 주파수가 높을수록 빠르게 진동
            예: 0.5 Hz = 2초 주기, 3.0 Hz = 0.33초 주기

          - ramp_duration: 에피소드 시작/끝에서 급격한 변화 방지
            예: 0.1초 동안 진폭을 0 → 목표값으로 증가
        """
        self.min_amp = min_amplitude
        self.max_amp = max_amplitude
        self.min_freq = min_frequency
        self.max_freq = max_frequency
        self.ramp_duration = ramp_duration

        # 현재 에피소드의 sin 파라미터 (reset()에서 랜덤 설정)
        self.amplitude = 0.0
        self.frequency = 0.0
        self.phase = 0.0
        self.start_time = 0.0

    def reset(self):
        """
        새 에피소드 시작 시 호출 - sin 파라미터 랜덤화

        각 에피소드마다:
          - 진폭: 0.5~2.0 rad 중 랜덤
          - 주파수: 0.5~3.0 Hz 중 랜덤
          - 위상: 0~2π 중 랜덤

        왜 랜덤화?
          - 다양한 궤적 경험 → 일반화 능력 향상
          - 특정 궤적에 과적합 방지
        """
        self.amplitude = np.random.uniform(self.min_amp, self.max_amp)
        self.frequency = np.random.uniform(self.min_freq, self.max_freq)
        self.phase = np.random.uniform(0, 2 * np.pi)
        self.start_time = 0.0

    def get_target(self, t: float, preview_steps: int = 32, dt: float = 0.001) -> tuple:
        """
        현재 시간 t에서의 목표 각도 및 미래 preview 생성

        Args:
            t: 에피소드 시작 후 경과 시간 (초)
            preview_steps: 미래 몇 스텝을 볼지 (K, 기본 32)
            dt: 시간 간격 (1 kHz = 0.001초)

        Returns:
            theta_now: 현재 목표 각도 (rad)
            dtheta_now: 현재 목표 각속도 (rad/s)
            theta_preview: 미래 K개 목표 각도 (배열)

        예시:
            t=0.5초, amplitude=1.0, frequency=1.0, phase=0
            theta_now = 1.0 * sin(2π*1.0*0.5 + 0) = 0 rad
            dtheta_now = 1.0 * 2π*1.0 * cos(...) = 2π rad/s

            preview_steps=32, dt=0.001
            theta_preview[0] = sin(2π*1.0*0.501) ≈ 0.006 rad
            theta_preview[1] = sin(2π*1.0*0.502) ≈ 0.013 rad
            ...
            theta_preview[31] = sin(2π*1.0*0.532) ≈ 0.2 rad

        왜 Preview 필요?
          - 엔코더 없는 시스템에서 "미래 목표"가 정책의 눈 역할
          - 정책이 "앞으로 목표가 어디로 갈지" 보고 미리 준비
        """
        # 램프 적용 (에피소드 시작 0.1초 동안 진폭 증가)
        if t < self.ramp_duration:
            ramp = t / self.ramp_duration  # 0 → 1로 증가
        else:
            ramp = 1.0  # 이후엔 전체 진폭

        amp = self.amplitude * ramp

        # 현재 목표 각도 (sin 파형)
        # θ(t) = A * sin(2πft + φ)
        theta_now = amp * np.sin(2 * np.pi * self.frequency * t + self.phase)

        # 현재 목표 각속도 (sin의 미분 = cos)
        # dθ/dt = A * 2πf * cos(2πft + φ)
        dtheta_now = amp * 2 * np.pi * self.frequency * np.cos(
            2 * np.pi * self.frequency * t + self.phase
        )

        # 미래 preview 생성 (t+dt, t+2dt, ..., t+K*dt)
        theta_preview = []
        for i in range(1, preview_steps + 1):
            t_future = t + i * dt  # 미래 시간
            theta_future = amp * np.sin(2 * np.pi * self.frequency * t_future + self.phase)
            theta_preview.append(theta_future)

        return theta_now, dtheta_now, np.array(theta_preview)


# ===================================================================================
# 2. 보상 계산기 (RewardCalculator)
# ===================================================================================
class RewardCalculator:
    """
    보상 계산 - ⭐ 엔코더 사용 (훈련 전용!)

    역할:
      - 엔코더로 실제 각도 측정
      - 목표와 비교해서 오차 계산
      - 오차 기반 보상 반환

    중요!
      - 엔코더는 여기서만 사용!
      - 관측(observation)에는 절대 넣지 않음!
      - 이유: 배포 시 엔코더 제거할 수 있도록

    보상 구성:
      1. 오차 페널티: 목표와 실제 각도 차이
      2. 오차 변화율 페널티: 오차가 커지는 속도
      3. 스위칭 페널티: 액션을 자주 바꾸면 페널티
      4. 결함 페널티: 안전 문제 발생 시 큰 페널티
    """

    def __init__(
        self,
        error_weight: float = 1.0,              # 오차 가중치
        error_derivative_weight: float = 0.5,   # 오차 변화율 가중치
        switch_penalty: float = 0.01,           # 스위칭 페널티
        fault_penalty: float = 10.0             # 결함 페널티
    ):
        """
        보상 계산기 초기화

        가중치 설명:
          - error_weight: 높을수록 정확한 추적 중요
          - error_derivative_weight: 높을수록 빠른 수렴 중요
          - switch_penalty: 높을수록 부드러운 제어 선호
          - fault_penalty: 안전 위반 시 강한 페널티
        """
        self.k_error = error_weight
        self.k_deriv = error_derivative_weight
        self.k_switch = switch_penalty
        self.k_fault = fault_penalty

        # 이전 스텝 정보 (미분 계산용)
        self.last_error = 0.0
        self.last_action = 0

    def reset(self):
        """
        에피소드 시작 시 내부 상태 리셋
        """
        self.last_error = 0.0
        self.last_action = 0

    def compute(
        self,
        theta_actual: float,   # 실제 각도 (엔코더 측정!)
        theta_ref: float,      # 목표 각도
        action: int,           # 선택한 액션
        fault_flags: int,      # 안전 결함 플래그 (0=정상)
        dt: float = 0.001      # 시간 간격 (1 kHz)
    ) -> tuple:
        """
        보상 계산

        Args:
            theta_actual: 엔코더로 측정한 실제 각도 (rad)
            theta_ref: 목표 각도 (rad)
            action: 이번 스텝에서 선택한 액션 (0~15)
            fault_flags: Teensy에서 보낸 결함 플래그
            dt: 시간 간격

        Returns:
            (reward, done)
            reward: 이번 스텝의 보상 (높을수록 좋음)
            done: 에피소드 종료 여부 (True=종료)

        보상 예시:
            theta_ref = 1.0, theta_actual = 1.05
            error = -0.05 rad (거의 정확!)
            r_error = -1.0 * 0.05 = -0.05

            error_deriv = (error - last_error) / dt
                       = (-0.05 - (-0.1)) / 0.001 = 50 rad/s (개선 중!)
            r_deriv = -0.5 * 50 = -25

            action이 바뀜 (3 → 5)
            r_switch = -0.01

            총 보상 = -0.05 - 25 - 0.01 = -25.06
            (음수지만 이전 스텝보다 나아짐)
        """
        # 1. 각도 오차 계산 (wrap to [-π, π])
        # 왜 wrap? 각도는 순환 (360도 = 0도)
        # 예: ref=350도, actual=10도 → error=-340도(X) → 20도(O)
        error = theta_ref - theta_actual
        while error > np.pi:
            error -= 2 * np.pi
        while error < -np.pi:
            error += 2 * np.pi

        # 2. 오차 변화율 (미분)
        # 양수: 오차가 커지는 중 (나쁨)
        # 음수: 오차가 줄어드는 중 (좋음)
        error_deriv = (error - self.last_error) / dt

        # 3. 보상 구성요소 계산
        # (a) 오차 페널티 - 작을수록 좋음
        r_error = -self.k_error * abs(error)

        # (b) 오차 변화율 페널티 - 빠르게 줄어들수록 좋음
        r_deriv = -self.k_deriv * abs(error_deriv)

        # (c) 스위칭 페널티 - 액션 바꾸면 페널티
        # 이유: 빈번한 스위칭은 하드웨어에 부담, 불안정
        r_switch = -self.k_switch * (action != self.last_action)

        # 4. 총 보상
        reward = r_error + r_deriv + r_switch

        # 5. 종료 조건 체크
        done = False
        if fault_flags != 0:
            # 안전 결함 발생 → 큰 페널티 + 에피소드 종료
            reward -= self.k_fault
            done = True

        # 6. 상태 업데이트 (다음 스텝용)
        self.last_error = error
        self.last_action = action

        return reward, done


# ===================================================================================
# 3. 메인 훈련 노드 (StepperRLTrainer)
# ===================================================================================
class StepperRLTrainer(Node):
    """
    ⭐⭐⭐ 전체 시스템의 핵심! ⭐⭐⭐

    역할:
      1. ROS 노드로서 Teensy와 통신
      2. 1 kHz 제어 루프 실행 (목표 생성, 액션 선택, 보상 계산)
      3. 비동기 학습 루프 (SAC 업데이트)
      4. 에피소드 관리
      5. 모델 저장 및 로깅

    동작 흐름:
      [제어 루프 1 kHz]
        목표 생성 → Teensy로 전송
        ↓
        관측 구성 (목표 + 행동이력)
        ↓
        정책으로 액션 선택 → Teensy로 전송
        ↓
        엔코더 수신 → 보상 계산
        ↓
        경험 저장 (리플레이 버퍼)

      [학습 루프 200 Hz (비동기)]
        리플레이 버퍼에서 배치 샘플링
        ↓
        SAC 네트워크 업데이트
        ↓
        PER 우선순위 업데이트

      [에피소드 종료 시]
        통계 로깅
        ↓
        모델 저장 (주기적)
        ↓
        다음 에피소드 시작 (리셋 없음!)
    """

    def __init__(self, config: dict):
        """
        훈련 노드 초기화

        Args:
            config: 설정 딕셔너리 (training_config.yaml에서 로드)
        """
        super().__init__('stepper_rl_trainer')  # ROS 노드 이름
        self.config = config

        # ===== ROS 통신 설정 =====
        # QoS: Quality of Service (메시지 전달 보장 수준)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,  # 메시지 손실 없이 전달
            history=HistoryPolicy.KEEP_LAST,         # 최신 N개만 보관
            depth=10                                  # 버퍼 크기
        )

        # Publisher (PC → Teensy)
        self.ref_pub = self.create_publisher(Float32, 'ref_target', qos)   # 목표 각도
        self.action_pub = self.create_publisher(UInt8, 'action_cmd', qos)  # 액션 명령

        # Subscriber (Teensy → PC)
        self.encoder_sub = self.create_subscription(
            Float32, 'encoder_angle', self.encoder_callback, qos
        )  # 엔코더 각도 (보상 계산용!)

        self.obs_sub = self.create_subscription(
            Float32, 'observation', self.obs_callback, qos
        )  # 관측 (현재는 미사용, 나중에 custom message로)

        # ===== 내부 상태 변수 =====
        self.current_encoder_angle = 0.0  # 최신 엔코더 값
        self.current_obs = None            # 최신 관측
        self.episode_start_time = self.get_clock().now()  # 에피소드 시작 시각
        self.episode_step = 0              # 현재 에피소드 스텝 수
        self.episode_count = 0             # 총 에피소드 수
        self.total_steps = 0               # 총 스텝 수 (모든 에피소드 합)

        # ===== 궤적 생성기 초기화 =====
        self.traj_gen = TrajectoryGenerator(
            min_amplitude=config['traj']['min_amplitude'],
            max_amplitude=config['traj']['max_amplitude'],
            min_frequency=config['traj']['min_frequency'],
            max_frequency=config['traj']['max_frequency']
        )
        self.traj_gen.reset()  # 첫 에피소드 파라미터 랜덤화

        # ===== 보상 계산기 초기화 =====
        self.reward_calc = RewardCalculator(
            error_weight=config['reward']['error_weight'],
            error_derivative_weight=config['reward']['derivative_weight'],
            switch_penalty=config['reward']['switch_penalty'],
            fault_penalty=config['reward']['fault_penalty']
        )

        # ===== SAC Agent 초기화 =====
        obs_dim = self._calculate_obs_dim()  # 관측 벡터 차원 계산
        self.agent = SACDiscrete(
            obs_dim=obs_dim,
            num_actions=config['rl']['num_actions'],
            hidden_dim=config['rl']['hidden_dim'],
            lr=config['rl']['learning_rate'],
            gamma=config['rl']['gamma'],
            tau=config['rl']['tau'],
            alpha=config['rl']['alpha'],
            auto_entropy_tuning=config['rl']['auto_entropy_tuning']
        )

        # ===== Replay Buffer 초기화 =====
        if config['rl']['use_per']:
            # PER (Prioritized Experience Replay) 사용
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=config['rl']['buffer_capacity'],
                obs_dim=obs_dim,
                alpha=config['rl']['per_alpha'],
                beta=config['rl']['per_beta'],
                n_step=config['rl']['n_step'],
                gamma=config['rl']['gamma']
            )
        else:
            # 단순 Uniform 샘플링
            self.replay_buffer = SimpleReplayBuffer(
                capacity=config['rl']['buffer_capacity'],
                obs_dim=obs_dim,
                n_step=config['rl']['n_step'],
                gamma=config['rl']['gamma']
            )

        # ===== 훈련 상태 변수 =====
        self.last_obs = None               # 이전 관측 (transition 저장용)
        self.last_action = None            # 이전 액션
        self.episode_rewards = []          # 각 에피소드의 총 보상
        self.episode_return = 0.0          # 현재 에피소드 누적 보상

        # ===== 로깅 디렉토리 =====
        self.log_dir = Path(config['logging']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = Path(config['logging']['model_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # ===== ROS Timer 설정 =====
        # (1) 제어 타이머: 1 kHz (0.001초마다)
        self.control_dt = 0.001
        self.control_timer = self.create_timer(self.control_dt, self.control_loop)

        # (2) 학습 타이머: 200 Hz (0.005초마다, 설정 가능)
        self.train_freq = config['rl']['train_frequency']
        self.train_timer = self.create_timer(1.0 / self.train_freq, self.train_loop)

        # (3) 로깅 타이머: 10초마다
        self.log_timer = self.create_timer(10.0, self.log_stats)

        # ===== 에피소드 설정 =====
        self.max_episode_steps = config['episode']['max_steps']
        self.min_episode_steps = config['episode']['min_steps']

        self.get_logger().info('StepperRLTrainer 초기화 완료!')
        self.get_logger().info(f'관측 차원: {obs_dim}')
        self.get_logger().info(f'리플레이 버퍼 크기: {config["rl"]["buffer_capacity"]}')

    def _calculate_obs_dim(self) -> int:
        """
        관측 벡터 차원 계산

        관측 구성:
          1. theta_ref_now (1)      - 현재 목표 각도
          2. dtheta_ref_now (1)     - 현재 목표 각속도
          3. theta_ref_preview (K)  - 미래 K개 목표 각도
          4. last_action_mask (1)   - 직전 액션
          5. action_hist (N)        - 과거 N개 액션 히스토리

        ⭐ 엔코더는 포함 안 됨! ⭐

        예: K=32, N=16
          obs_dim = 1 + 1 + 32 + 1 + 16 = 51
        """
        K = self.config['rl']['preview_length']      # Preview 길이
        N = self.config['rl']['action_hist_length']  # 액션 히스토리 길이
        return 1 + 1 + K + 1 + N

    def encoder_callback(self, msg: Float32):
        """
        엔코더 각도 수신 콜백 (Teensy → PC)

        ⭐ 중요: 엔코더는 보상 계산에만 사용! ⭐
        관측에는 절대 넣지 않음!

        Args:
            msg: 엔코더 각도 메시지 (rad)
        """
        self.current_encoder_angle = msg.data

    def obs_callback(self, msg: Float32):
        """
        관측 수신 콜백 (Teensy → PC)

        현재는 placeholder (나중에 custom Observation message 사용)
        """
        pass  # TODO: Custom message 구현 후 파싱

    def control_loop(self):
        """
        ⭐⭐⭐ 1 kHz 제어 루프 - 시스템의 심장! ⭐⭐⭐

        매 1ms마다 실행되며 다음 작업 수행:
          1. 에피소드 시간 계산
          2. 목표 궤적 생성 (sin wave + preview)
          3. 목표를 Teensy로 전송
          4. 관측 구성 (엔코더 제외!)
          5. 정책으로 액션 선택
          6. 액션을 Teensy로 전송
          7. 보상 계산 (엔코더 사용!)
          8. 경험 저장 (리플레이 버퍼)
          9. 에피소드 종료 체크

        동작 흐름:
          [제어 루프] → 목표 생성 → Teensy
                                      ↓
          [제어 루프] ← 엔코더 ← Teensy

          [제어 루프] → 관측 구성 (목표 + 행동이력)
                     → 액션 선택
                     → Teensy로 전송

          [제어 루프] → 보상 계산 (엔코더 사용)
                     → 버퍼 저장
        """
        # ===== 1. 에피소드 경과 시간 계산 =====
        current_time = self.get_clock().now()
        episode_elapsed = (current_time - self.episode_start_time).nanoseconds / 1e9  # 초 단위

        # ===== 2. 목표 궤적 생성 =====
        theta_ref, dtheta_ref, theta_preview = self.traj_gen.get_target(
            episode_elapsed,
            preview_steps=self.config['rl']['preview_length'],
            dt=self.control_dt
        )
        # theta_ref: 현재 목표 (예: 0.5 rad)
        # dtheta_ref: 현재 목표 속도 (예: 1.2 rad/s)
        # theta_preview: 미래 32개 목표 (예: [0.51, 0.52, ..., 0.7])

        # ===== 3. 목표를 Teensy로 전송 =====
        # TODO: Custom RefTarget message로 교체
        ref_msg = Float32()
        ref_msg.data = float(theta_ref)
        self.ref_pub.publish(ref_msg)

        # ===== 4. 관측 구성 (⭐ 엔코더 제외!) =====
        obs = self._build_observation(theta_ref, dtheta_ref, theta_preview)
        # obs = [theta_ref, dtheta_ref, preview[0], ..., preview[31],
        #        last_action, action_hist[0], ..., action_hist[15]]

        # ===== 5. 액션 선택 =====
        if len(self.replay_buffer) < self.config['rl']['warmup_steps']:
            # Warmup 단계: 랜덤 탐험
            # 이유: 초기에는 정책이 엉망이므로 랜덤 데이터 수집
            action = np.random.randint(0, self.config['rl']['num_actions'])
        else:
            # 훈련 단계: 정책 사용 (확률적 샘플링)
            action = self.agent.select_action(obs, deterministic=False)

        # ===== 6. 액션을 Teensy로 전송 =====
        action_msg = UInt8()
        action_msg.data = int(action)  # 0~15
        self.action_pub.publish(action_msg)

        # ===== 7. 보상 계산 (⭐ 엔코더 사용!) =====
        reward, done = self.reward_calc.compute(
            self.current_encoder_angle,  # 실제 각도 (엔코더)
            theta_ref,                   # 목표 각도
            action,                      # 선택한 액션
            fault_flags=0                # TODO: Teensy에서 받기
        )

        self.episode_return += reward  # 누적 보상

        # ===== 8. 경험 저장 (Transition) =====
        # Transition: (s, a, r, s', done)
        if self.last_obs is not None and self.last_action is not None:
            self.replay_buffer.add(
                self.last_obs,    # 이전 관측
                self.last_action, # 이전 액션
                reward,           # 받은 보상
                obs,              # 현재 관측 (다음 상태)
                done              # 종료 플래그
            )

        # ===== 9. 에피소드 종료 체크 =====
        self.episode_step += 1
        self.total_steps += 1

        if done or self.episode_step >= self.max_episode_steps:
            # 종료 조건:
            #   - 안전 결함 발생 (done=True)
            #   - 최대 스텝 도달 (timeout)
            self._end_episode()

        # ===== 10. 상태 업데이트 (다음 스텝용) =====
        self.last_obs = obs
        self.last_action = action

    def _build_observation(
        self,
        theta_ref: float,
        dtheta_ref: float,
        theta_preview: np.ndarray
    ) -> np.ndarray:
        """
        관측 벡터 구성 - ⭐ 엔코더 미포함! ⭐

        관측 = [목표 정보] + [행동 이력]
             = [theta_ref, dtheta_ref, preview[K], last_action, action_hist[N]]

        Args:
            theta_ref: 현재 목표 각도
            dtheta_ref: 현재 목표 각속도
            theta_preview: 미래 K개 목표

        Returns:
            obs: 관측 벡터 (numpy array, 차원: 1+1+K+1+N)

        예시:
            theta_ref = 0.5
            dtheta_ref = 1.2
            theta_preview = [0.51, 0.52, ..., 0.7] (32개)
            last_action = 3
            action_hist = [3, 2, 3, 5, ...] (16개)

            obs = [0.5, 1.2, 0.51, 0.52, ..., 0.7, 3, 3, 2, 3, 5, ...]
            차원 = 1 + 1 + 32 + 1 + 16 = 51
        """
        obs = [theta_ref, dtheta_ref]
        obs.extend(theta_preview.tolist())  # 미래 목표 추가

        # 직전 액션 추가
        if self.last_action is not None:
            obs.append(float(self.last_action))
        else:
            obs.append(0.0)  # 첫 스텝

        # 액션 히스토리 추가 (현재는 placeholder)
        # TODO: Teensy에서 실제 히스토리 받아오기
        N = self.config['rl']['action_hist_length']
        obs.extend([0.0] * N)

        return np.array(obs, dtype=np.float32)

    def train_loop(self):
        """
        학습 루프 - 비동기로 200 Hz 실행

        제어 루프와 독립적으로 실행되며:
          1. 리플레이 버퍼에서 배치 샘플링
          2. SAC 네트워크 업데이트
          3. PER 우선순위 업데이트

        왜 비동기?
          - 제어는 1 kHz로 빠르게 (실시간성)
          - 학습은 느려도 됨 (정확성)
          - GPU 사용 가능 (제어와 분리)
        """
        # Warmup 완료 확인
        if len(self.replay_buffer) < self.config['rl']['warmup_steps']:
            return

        # 배치 크기만큼 데이터 있는지 확인
        if len(self.replay_buffer) < self.config['rl']['batch_size']:
            return

        # ===== 1. 배치 샘플링 =====
        batch = self.replay_buffer.sample(self.config['rl']['batch_size'])
        obs, actions, rewards, next_obs, dones, weights, indices = batch

        # ===== 2. GPU로 이동 =====
        obs = obs.to(self.agent.device)
        actions = actions.to(self.agent.device)
        rewards = rewards.to(self.agent.device)
        next_obs = next_obs.to(self.agent.device)
        dones = dones.to(self.agent.device)
        if weights is not None:
            weights = weights.to(self.agent.device)

        # ===== 3. SAC 업데이트 =====
        metrics = self.agent.update(obs, actions, rewards, next_obs, dones, weights)

        # ===== 4. PER 우선순위 업데이트 =====
        if self.config['rl']['use_per'] and indices is not None:
            self.replay_buffer.update_priorities(indices, metrics['td_errors'])

        # ===== 5. 로깅 (1000 스텝마다) =====
        if self.total_steps % 1000 == 0:
            self.get_logger().info(
                f"Step {self.total_steps} | "
                f"Q-loss: {metrics['q1_loss']:.4f} | "
                f"Policy-loss: {metrics['policy_loss']:.4f} | "
                f"Alpha: {metrics['alpha']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f}"
            )

    def _end_episode(self):
        """
        에피소드 종료 처리

        작업:
          1. 에피소드 통계 기록
          2. 로깅
          3. 모델 저장 (주기적)
          4. 다음 에피소드 준비 (리셋!)
        """
        # ===== 1. 통계 기록 =====
        self.episode_rewards.append(self.episode_return)
        self.episode_count += 1

        # ===== 2. 로깅 =====
        self.get_logger().info(
            f"Episode {self.episode_count} 종료 | "
            f"Steps: {self.episode_step} | "
            f"Return: {self.episode_return:.4f} | "
            f"평균 Return (100): {np.mean(self.episode_rewards[-100:]):.4f}"
        )

        # ===== 3. 모델 저장 (주기적) =====
        if self.episode_count % self.config['logging']['save_interval'] == 0:
            self._save_model()

        # ===== 4. 다음 에피소드 준비 =====
        self.episode_step = 0
        self.episode_return = 0.0
        self.episode_start_time = self.get_clock().now()

        # 궤적 파라미터 랜덤화 (새로운 sin 파형)
        self.traj_gen.reset()

        # 보상 계산기 리셋
        self.reward_calc.reset()

        # 주의: 물리적 리셋 없음! 모터는 계속 움직임

    def _save_model(self):
        """
        모델 체크포인트 저장

        저장 내용:
          1. PyTorch 모델 (.pt) - 학습 재개용
          2. 양자화 모델 (.bin) - Teensy 배포용
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # ===== 1. PyTorch 모델 저장 =====
        model_path = self.model_dir / f'model_ep{self.episode_count}_{timestamp}.pt'
        self.agent.save(str(model_path))
        self.get_logger().info(f'모델 저장: {model_path}')

        # ===== 2. 양자화 모델 저장 (Teensy용) =====
        quant_policy = QuantizedPolicy(self.agent.policy)
        quant_policy.quantize()
        quant_path = self.model_dir / f'model_ep{self.episode_count}_{timestamp}_quant.bin'
        quant_policy.save(str(quant_path))

    def log_stats(self):
        """
        주기적 통계 로깅 (10초마다)

        로깅 내용:
          - 에피소드 수
          - 총 스텝 수
          - 버퍼 크기
          - 평균 리턴 (최근 10/100 에피소드)
        """
        if len(self.episode_rewards) == 0:
            return

        stats = {
            'episode': self.episode_count,
            'total_steps': self.total_steps,
            'buffer_size': len(self.replay_buffer),
            'avg_return_10': float(np.mean(self.episode_rewards[-10:])),
            'avg_return_100': float(np.mean(self.episode_rewards[-100:])),
            'max_return': float(np.max(self.episode_rewards)),
        }

        self.get_logger().info(f"통계: {stats}")

        # 파일에 저장 (JSONL 형식)
        stats_file = self.log_dir / 'training_stats.jsonl'
        with open(stats_file, 'a') as f:
            f.write(json.dumps(stats) + '\n')


# ===================================================================================
# 메인 함수
# ===================================================================================
def main(args=None):
    """
    훈련 노드 실행

    사용법:
        ros2 run stepper_rl_control train
    """
    rclpy.init(args=args)

    # ===== 설정 로드 =====
    # TODO: YAML 파일에서 로드
    config = {
        'traj': {
            'min_amplitude': 0.5,
            'max_amplitude': 2.0,
            'min_frequency': 0.5,
            'max_frequency': 3.0
        },
        'reward': {
            'error_weight': 1.0,
            'derivative_weight': 0.5,
            'switch_penalty': 0.01,
            'fault_penalty': 10.0
        },
        'rl': {
            'num_actions': 16,
            'hidden_dim': 64,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'auto_entropy_tuning': True,
            'buffer_capacity': 100000,
            'use_per': True,
            'per_alpha': 0.6,
            'per_beta': 0.4,
            'n_step': 5,
            'batch_size': 256,
            'warmup_steps': 1000,
            'train_frequency': 200,
            'preview_length': 32,
            'action_hist_length': 16
        },
        'episode': {
            'min_steps': 2000,
            'max_steps': 5000
        },
        'logging': {
            'log_dir': 'logs',
            'model_dir': 'models',
            'save_interval': 10
        }
    }

    # ===== 노드 생성 및 실행 =====
    node = StepperRLTrainer(config)

    try:
        rclpy.spin(node)  # 무한 루프 (Ctrl+C로 종료)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
