"""
===================================================================================
SAC-Discrete (Soft Actor-Critic for Discrete Actions) 알고리즘 구현
===================================================================================

이 파일의 목적:
  - 스테퍼 모터의 4비트 스위치(16개 조합)를 제어하기 위한 강화학습 알고리즘
  - DDPG의 Discrete 버전 + Entropy Regularization (자동 탐험)

핵심 아이디어:
  1. Policy(정책): 상태를 보고 "어떤 액션이 좋을지 확률로" 출력
     예: [액션0: 5%, 액션1: 10%, 액션2: 30%, ..., 액션15: 3%]

  2. Q-network(가치): 상태를 보고 "각 액션의 가치(Q-value)" 출력
     예: [액션0: 0.5, 액션1: 1.2, 액션2: 2.3, ..., 액션15: 0.1]

  3. Entropy(엔트로피): 정책이 얼마나 "불확실한가" (탐험 정도)
     - 높음: 여러 액션을 골고루 선택 → 탐험 많이
     - 낮음: 특정 액션에 집중 → 활용 많이

  4. Temperature(α): 탐험과 활용의 균형을 조절하는 온도 파라미터
     - 자동으로 학습됨! (사용자가 조절 불필요)

DDPG와 비교:
  - DDPG: Continuous action (예: 모터 전압 0~12V)
  - SAC-Discrete: Discrete action (예: 16개 스위치 조합 중 하나)
  - DDPG: 결정론적 정책 (같은 상태 → 항상 같은 액션)
  - SAC-Discrete: 확률론적 정책 (같은 상태 → 여러 액션 가능, 확률적 선택)
  - DDPG: 노이즈 수동 추가 (탐험)
  - SAC-Discrete: Entropy 자동 조절 (탐험)
===================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional


# ===================================================================================
# 1. Policy Network (정책 네트워크)
# ===================================================================================
class PolicyNetwork(nn.Module):
    """
    정책 네트워크: π(a|s) - "상태를 보고 어떤 액션을 선택할지 확률 분포 출력"

    역할:
      - 입력: 관측(observation) - 목표 각도, 미래 목표, 과거 액션 등
      - 출력: 16개 액션에 대한 logits (확률로 변환 전 값)

    DDPG의 Actor와 비교:
      - DDPG Actor: obs → 단일 액션 값 (예: 전압 5.3V)
      - SAC Policy: obs → 16개 액션의 확률 (예: [0.05, 0.1, 0.3, ...])

    왜 확률로 출력?
      - 같은 상태에서도 여러 액션 시도 가능 → 탐험
      - 훈련 중: 확률 높은 액션 주로 선택하지만 낮은 것도 가끔 시도
      - 배포 시: 가장 확률 높은 액션만 선택 (deterministic)
    """

    def __init__(self, obs_dim: int, num_actions: int = 16, hidden_dim: int = 64):
        """
        네트워크 초기화

        Args:
            obs_dim: 관측 벡터의 차원 (예: 50)
                    = 1(theta_ref) + 1(dtheta_ref) + 32(preview) + 1(last_action) + 16(action_hist)
            num_actions: 액션 개수 (4비트 스위치 = 16개 조합)
            hidden_dim: 은닉층 뉴런 수 (기본 64)
        """
        super().__init__()

        # 3층 MLP (Multi-Layer Perceptron) 구조
        # obs_dim → 64 → 64 → 16
        self.fc1 = nn.Linear(obs_dim, hidden_dim)      # 첫 번째 층: 입력 → 은닉층1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)   # 두 번째 층: 은닉층1 → 은닉층2
        self.logits_out = nn.Linear(hidden_dim, num_actions)  # 출력층: 은닉층2 → 액션 logits

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        순전파: 관측을 받아서 액션 logits 출력

        Args:
            obs: 관측 텐서 (batch_size, obs_dim)
                예: [[0.5, 1.2, 0.3, ...], [0.2, 0.9, ...]]  # 배치 크기만큼

        Returns:
            logits: 액션에 대한 logits (batch_size, 16)
                예: [[2.3, 1.5, 3.2, ...], ...]

        Logits란?
            - Softmax 적용 전의 값
            - 큰 값 = 그 액션이 좋음
            - Softmax(logits) = 확률 분포
            예: logits = [2.0, 1.0, 3.0] → softmax → [0.24, 0.09, 0.67]
        """
        x = F.relu(self.fc1(obs))      # 첫 번째 층 + ReLU 활성화
        x = F.relu(self.fc2(x))        # 두 번째 층 + ReLU 활성화
        logits = self.logits_out(x)    # 출력층 (활성화 함수 없음!)
        return logits

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        액션 선택 함수 - 실제 환경에서 사용

        Args:
            obs: 관측 텐서 (1, obs_dim) - 단일 상태
            deterministic: True면 가장 좋은 액션만 선택 (배포 시)
                          False면 확률적으로 샘플링 (훈련 시)

        Returns:
            action: 선택된 액션 ID (0~15 중 하나)
            log_prob: 선택한 액션의 로그 확률 (학습에 사용)

        예시:
            obs = [0.5, 1.2, ...]
            logits = [2.3, 1.5, 3.2, 1.8, ...] (16개)
            probs = [0.15, 0.06, 0.35, 0.09, ...] (softmax 적용)

            deterministic=False: 확률적 샘플링
                → 35% 확률로 액션2, 15% 확률로 액션0, ...
                → action = 2 (샘플 결과)

            deterministic=True: argmax
                → action = 2 (가장 높은 확률)
        """
        # 1. Logits → 확률 변환
        logits = self.forward(obs)                    # (1, 16)
        probs = F.softmax(logits, dim=-1)             # (1, 16) 확률 분포

        if deterministic:
            # 배포 시: 가장 확률 높은 액션 선택
            action = torch.argmax(probs, dim=-1)      # 예: 2 (가장 큰 확률)
            log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)) + 1e-8)
        else:
            # 훈련 시: 확률 분포에서 샘플링 (탐험)
            dist = torch.distributions.Categorical(probs)  # 카테고리 분포 생성
            action = dist.sample()                    # 확률적 샘플링
            log_prob = dist.log_prob(action)          # log P(action)

        return action.item(), log_prob  # item(): 텐서 → 파이썬 정수

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        주어진 액션의 확률과 엔트로피 계산 - 정책 업데이트 시 사용

        Args:
            obs: 관측 배치 (batch_size, obs_dim)
            actions: 액션 배치 (batch_size,) - 이미 선택된 액션들

        Returns:
            log_probs: 각 액션의 로그 확률 (batch_size,)
            entropy: 정책의 엔트로피 (batch_size,) - 불확실성

        언제 사용?
            - 리플레이 버퍼에서 과거 경험을 샘플링했을 때
            - "그때 선택한 액션이 현재 정책에서는 얼마나 확률 높은가?" 확인

        Entropy(엔트로피)란?
            - H = -Σ p(a) * log p(a)
            - 정책이 얼마나 "고르게 분산되어 있는가"
            - 높음: 여러 액션 고르게 선택 (탐험 많이)
              예: probs = [0.25, 0.25, 0.25, 0.25] → H 높음
            - 낮음: 특정 액션에 집중 (활용 많이)
              예: probs = [0.9, 0.03, 0.03, 0.04] → H 낮음
        """
        logits = self.forward(obs)                # (batch, 16)
        probs = F.softmax(logits, dim=-1)         # (batch, 16)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)        # (batch,) - 각 액션의 log P
        entropy = dist.entropy()                  # (batch,) - 각 상태의 엔트로피

        return log_probs, entropy


# ===================================================================================
# 2. Q-Network (가치 네트워크)
# ===================================================================================
class QNetwork(nn.Module):
    """
    Q-네트워크: Q(s, a) - "상태에서 각 액션을 선택하면 얼마나 좋은지 평가"

    역할:
      - 입력: 관측(observation)
      - 출력: 16개 액션 각각의 Q-value (예상 누적 보상)

    DDPG의 Critic과 비교:
      - DDPG Critic: Q(s, a) → 단일 스칼라 (주어진 액션의 가치)
                    입력에 액션도 포함됨
      - SAC Q-network: Q(s, ·) → 16개 Q-values (모든 액션의 가치)
                      입력은 상태만! 액션은 출력으로

    왜 모든 액션의 Q-value를 출력?
      - Discrete action space에서는 가능한 액션이 적음 (16개)
      - 한 번에 모두 계산하면 효율적
      - Policy 업데이트 시 "모든 액션의 가치"를 알아야 하므로

    Q-value란?
      - "이 상태에서 이 액션을 선택하면 앞으로 받을 총 보상의 기댓값"
      - 예: Q(s, 액션3) = 5.2 → 액션3 선택 시 앞으로 평균 5.2의 보상
      - 높을수록 좋은 액션
    """

    def __init__(self, obs_dim: int, num_actions: int = 16, hidden_dim: int = 64):
        """
        네트워크 초기화

        Args:
            obs_dim: 관측 벡터 차원
            num_actions: 액션 개수 (16)
            hidden_dim: 은닉층 크기 (64)
        """
        super().__init__()

        # Policy와 동일한 3층 MLP 구조
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, num_actions)  # 출력: 16개 Q-values

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        순전파: 관측 → 모든 액션의 Q-value

        Args:
            obs: 관측 텐서 (batch_size, obs_dim)

        Returns:
            q_values: 각 액션의 Q-value (batch_size, 16)
                예: [[2.3, 1.5, 3.2, ...], ...]
                    첫 번째 샘플의 액션0 Q-value는 2.3
                    첫 번째 샘플의 액션2 Q-value는 3.2 (가장 높음!)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q_values = self.q_out(x)
        return q_values


# ===================================================================================
# 3. SAC-Discrete 메인 클래스
# ===================================================================================
class SACDiscrete:
    """
    SAC-Discrete 알고리즘의 메인 클래스

    이 클래스가 하는 일:
      1. 네트워크 관리: Policy, Q1, Q2, Q1_target, Q2_target
      2. 액션 선택: 정책을 사용해서 환경에서 액션 선택
      3. 학습 업데이트: 리플레이 버퍼의 경험으로 네트워크들 학습
      4. 모델 저장/로드

    왜 Q 네트워크가 2개 (Twin Q)?
      - Q-value 과대평가 방지 (Overestimation bias)
      - Q1, Q2 중 작은 값을 사용 → 보수적 추정
      - DDPG의 TD3에서 차용한 기법

    왜 Target 네트워크?
      - 학습 안정성
      - 학습 목표(target)가 계속 변하면 불안정
      - Target은 천천히 업데이트 (soft update)
      - DDPG와 동일한 기법
    """

    def __init__(
        self,
        obs_dim: int,                 # 관측 벡터 차원
        num_actions: int = 16,        # 액션 개수
        hidden_dim: int = 64,         # 은닉층 크기
        lr: float = 3e-4,             # 학습률 (learning rate)
        gamma: float = 0.99,          # 할인율 (discount factor)
        tau: float = 0.005,           # Target 네트워크 업데이트 비율
        alpha: float = 0.2,           # 초기 엔트로피 계수 (온도)
        auto_entropy_tuning: bool = True,  # 온도 자동 조절 여부
        target_entropy: Optional[float] = None,  # 목표 엔트로피
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        SAC-Discrete 알고리즘 초기화

        하이퍼파라미터 설명:
          - lr (learning rate): 얼마나 빨리 학습할지
              너무 크면 불안정, 너무 작으면 느림
              3e-4 (0.0003)는 일반적인 기본값

          - gamma (discount factor): 미래 보상을 얼마나 중요하게 볼지
              0.99 = 미래 보상도 매우 중요
              0.9 = 가까운 미래만 중요
              0 = 현재 보상만 중요

          - tau: Target 네트워크 업데이트 속도
              0.005 = 매 스텝마다 0.5%씩 업데이트 (천천히)
              1.0 = 즉시 복사 (사용 안 함)

          - alpha: 탐험-활용 균형 (Temperature)
              높음 = 탐험 많이 (여러 액션 시도)
              낮음 = 활용 많이 (좋은 액션에 집중)
              auto_entropy_tuning=True면 자동 조절!
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.num_actions = num_actions

        # ===== 네트워크 생성 =====
        # 1. Policy 네트워크 (액션 선택용)
        self.policy = PolicyNetwork(obs_dim, num_actions, hidden_dim).to(device)

        # 2. Q 네트워크 2개 (Twin Q)
        self.q1 = QNetwork(obs_dim, num_actions, hidden_dim).to(device)
        self.q2 = QNetwork(obs_dim, num_actions, hidden_dim).to(device)

        # 3. Target Q 네트워크 (학습 안정화용)
        self.q1_target = QNetwork(obs_dim, num_actions, hidden_dim).to(device)
        self.q2_target = QNetwork(obs_dim, num_actions, hidden_dim).to(device)

        # Target을 현재 Q로 초기화 (가중치 복사)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Target 네트워크는 학습 안 함 (gradient 계산 안 함)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # ===== Optimizer (최적화기) 생성 =====
        # 각 네트워크마다 독립적인 optimizer
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # ===== Entropy Tuning (온도 조절) =====
        self.auto_entropy_tuning = auto_entropy_tuning
        if self.auto_entropy_tuning:
            # 목표 엔트로피 계산
            # -log(1/|A|) = 균일 분포의 엔트로피
            # 16개 액션 균일 분포: -log(1/16) ≈ 2.77
            # 0.7 곱하면 약 1.94 → "적당히 탐험하라"
            if target_entropy is None:
                self.target_entropy = -np.log(1.0 / num_actions) * 0.7
            else:
                self.target_entropy = target_entropy

            # log(α)를 학습 파라미터로 (α는 항상 양수여야 하므로)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()  # α = exp(log α)
        else:
            # 자동 조절 안 하면 고정값 사용
            self.alpha = torch.tensor(alpha).to(device)

        self.training_step = 0  # 현재 학습 스텝 카운터

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        환경과 상호작용할 때 액션 선택

        Args:
            obs: 현재 관측 (numpy 배열)
                예: [0.5, 1.2, 0.3, ...] (50차원)
            deterministic: True면 최고 확률 액션만, False면 샘플링

        Returns:
            action: 선택된 액션 ID (0~15)

        사용 예:
            # 훈련 중
            action = agent.select_action(obs, deterministic=False)

            # 평가/배포 시
            action = agent.select_action(obs, deterministic=True)
        """
        with torch.no_grad():  # Gradient 계산 안 함 (추론만)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # (1, obs_dim)
            action, _ = self.policy.get_action(obs_tensor, deterministic)
        return action

    def update(
        self,
        obs: torch.Tensor,           # 현재 상태 배치
        actions: torch.Tensor,       # 선택한 액션 배치
        rewards: torch.Tensor,       # 받은 보상 배치
        next_obs: torch.Tensor,      # 다음 상태 배치
        dones: torch.Tensor,         # 종료 플래그 배치
        weights: Optional[torch.Tensor] = None  # PER 중요도 가중치
    ) -> dict:
        """
        ⭐⭐⭐ SAC-Discrete의 핵심 함수! ⭐⭐⭐

        리플레이 버퍼에서 샘플링한 경험으로 네트워크들을 학습시킴

        Args:
            obs: (batch_size, obs_dim) - 예: (256, 50)
            actions: (batch_size,) - 예: [3, 7, 2, 15, ...]
            rewards: (batch_size,) - 예: [-0.5, -1.2, 0.3, ...]
            next_obs: (batch_size, obs_dim)
            dones: (batch_size,) - 예: [0, 0, 1, 0, ...] (1=종료)
            weights: (batch_size,) - PER의 중요도 가중치 (없으면 모두 1)

        Returns:
            학습 통계를 담은 딕셔너리

        학습 순서:
            1. Q-네트워크 업데이트 (가치 추정 개선)
            2. Policy 업데이트 (정책 개선)
            3. Alpha 업데이트 (온도 조절)
            4. Target 네트워크 업데이트 (천천히)
        """
        if weights is None:
            weights = torch.ones_like(rewards)  # PER 안 쓰면 모두 동일 가중치

        # ============================================================
        # STEP 1: Q-네트워크 업데이트
        # ============================================================
        # 목표: Q(s,a) ≈ r + γ * V(s')
        # V(s') = E_a'~π [Q(s',a') - α*log π(a'|s')]

        with torch.no_grad():  # Target 계산 시 gradient 전파 안 함
            # 1-1. 다음 상태에서 정책의 확률 분포 계산
            next_logits = self.policy(next_obs)              # (batch, 16) logits
            next_probs = F.softmax(next_logits, dim=-1)      # (batch, 16) 확률

            # 1-2. 다음 상태의 Q-value (Target 네트워크 사용)
            next_q1 = self.q1_target(next_obs)               # (batch, 16)
            next_q2 = self.q2_target(next_obs)               # (batch, 16)
            next_q = torch.min(next_q1, next_q2)             # (batch, 16) - Twin Q의 최소값

            # 1-3. 다음 상태의 Value 계산 (기댓값)
            # V(s') = Σ π(a'|s') * [Q(s',a') - α*log π(a'|s')]
            # 이게 SAC의 핵심! Entropy 보정된 Value
            next_v = (next_probs * (next_q - self.alpha * torch.log(next_probs + 1e-8))).sum(dim=-1)
            # next_v.shape = (batch,)

            # 1-4. TD Target 계산
            # Target = r + γ * V(s') (단, 종료 상태면 V(s')=0)
            q_target = rewards + (1 - dones) * self.gamma * next_v
            # q_target.shape = (batch,)

        # 1-5. 현재 Q-value 계산 (실제 선택한 액션의 Q)
        q1_pred = self.q1(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_pred = self.q2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        # gather: 16개 Q-value 중에서 실제 선택한 액션의 Q만 추출
        # 예: Q-values = [1.2, 3.5, 0.8, ...], action=1 → Q=3.5

        # 1-6. Q-loss 계산 (MSE)
        # Loss = (Q_predicted - Q_target)^2
        # weights는 PER의 중요도 (중요한 경험일수록 큰 가중치)
        q1_loss = (weights * F.mse_loss(q1_pred, q_target, reduction='none')).mean()
        q2_loss = (weights * F.mse_loss(q2_pred, q_target, reduction='none')).mean()

        # 1-7. Q1 업데이트
        self.q1_optimizer.zero_grad()  # Gradient 초기화
        q1_loss.backward()             # Backpropagation
        self.q1_optimizer.step()       # 가중치 업데이트

        # 1-8. Q2 업데이트
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # ============================================================
        # STEP 2: Policy 업데이트
        # ============================================================
        # 목표: 기댓값 최대화 E_a~π [Q(s,a) - α*log π(a|s)]
        #      = 보상 많이 받으면서 + 엔트로피 높게 (탐험)

        # 2-1. 현재 정책의 확률 분포
        logits = self.policy(obs)                    # (batch, 16)
        probs = F.softmax(logits, dim=-1)           # (batch, 16) 확률
        log_probs = F.log_softmax(logits, dim=-1)   # (batch, 16) log 확률

        # 2-2. 현재 Q-values (gradient 전파 안 함!)
        with torch.no_grad():
            q1_curr = self.q1(obs)                  # (batch, 16)
            q2_curr = self.q2(obs)                  # (batch, 16)
            q_curr = torch.min(q1_curr, q2_curr)    # (batch, 16)

        # 2-3. Policy Loss 계산
        # Loss = E_a~π [α*log π(a|s) - Q(s,a)]
        # (주의: Maximize하려면 Loss에 -붙여야 하는데, 여기선 이미 부호 반영됨)
        policy_loss = (probs * (self.alpha * log_probs - q_curr)).sum(dim=-1).mean()
        # 해석:
        #   - Q-value 높은 액션의 확률을 높임 (보상 최대화)
        #   - 동시에 엔트로피도 유지 (탐험)

        # 2-4. Policy 업데이트
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ============================================================
        # STEP 3: Temperature (α) 업데이트
        # ============================================================
        # 목표: 현재 엔트로피를 목표 엔트로피에 맞춤
        #      엔트로피 > 목표 → α 증가 (탐험 줄임)
        #      엔트로피 < 목표 → α 감소 (탐험 늘림)

        alpha_loss = torch.tensor(0.0)
        if self.auto_entropy_tuning:
            # 3-1. 현재 엔트로피 계산
            # H = -Σ p(a) * log p(a)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            # 3-2. Alpha Loss
            # Loss = -α * (H - H_target)
            # H > H_target → Loss < 0 → α 감소 (탐험 줄임)
            # H < H_target → Loss > 0 → α 증가 (탐험 늘림)
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

            # 3-3. Alpha 업데이트
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # 3-4. α = exp(log α) (항상 양수 유지)
            self.alpha = self.log_alpha.exp()

        # ============================================================
        # STEP 4: Target 네트워크 업데이트 (Soft Update)
        # ============================================================
        # θ_target ← τ*θ + (1-τ)*θ_target
        # τ=0.005 → 매 스텝마다 0.5%씩 천천히 업데이트
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        self.training_step += 1

        # ============================================================
        # STEP 5: TD Error 계산 (PER용)
        # ============================================================
        # PER에서 우선순위를 업데이트하려면 TD error 필요
        with torch.no_grad():
            td_errors = torch.abs(q1_pred - q_target)

        # ============================================================
        # 학습 통계 반환
        # ============================================================
        return {
            'q1_loss': q1_loss.item(),       # Q1 손실
            'q2_loss': q2_loss.item(),       # Q2 손실
            'policy_loss': policy_loss.item(),  # Policy 손실
            'alpha_loss': alpha_loss.item(), # Alpha 손실
            'alpha': self.alpha.item(),      # 현재 온도
            'entropy': -(probs * log_probs).sum(dim=-1).mean().item(),  # 현재 엔트로피
            'q_mean': q_curr.mean().item(),  # 평균 Q-value
            'td_errors': td_errors.cpu().numpy()  # TD errors (PER용)
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """
        Target 네트워크 소프트 업데이트

        θ_target ← τ*θ_source + (1-τ)*θ_target

        왜 소프트 업데이트?
          - Target이 너무 빨리 변하면 학습 불안정
          - 천천히 업데이트해서 안정적인 목표 제공
          - τ=0.005 → 200 스텝 후에야 50% 변화

        DDPG/TD3와 동일한 기법
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path: str):
        """
        모델 체크포인트 저장

        저장 내용:
          - 모든 네트워크 가중치 (policy, q1, q2, targets)
          - 모든 optimizer 상태 (학습 재개용)
          - Alpha 파라미터 (자동 튜닝 시)
          - 학습 스텝 카운터
        """
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None,
            'training_step': self.training_step
        }, path)

    def load(self, path: str):
        """
        모델 체크포인트 로드

        사용 예:
            agent = SACDiscrete(obs_dim=50)
            agent.load('models/model_ep1000.pt')
            # 이제 agent로 추론 또는 학습 재개 가능
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha.data.copy_(checkpoint['log_alpha'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.training_step = checkpoint['training_step']


# ===================================================================================
# 사용 예시
# ===================================================================================
if __name__ == '__main__':
    """
    간단한 사용 예시
    """
    # 1. Agent 생성
    obs_dim = 50  # 관측 차원
    agent = SACDiscrete(
        obs_dim=obs_dim,
        num_actions=16,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        auto_entropy_tuning=True
    )

    # 2. 액션 선택
    obs = np.random.randn(obs_dim)
    action = agent.select_action(obs, deterministic=False)
    print(f"Selected action: {action}")

    # 3. 학습 (더미 데이터)
    batch_size = 256
    obs_batch = torch.randn(batch_size, obs_dim)
    actions_batch = torch.randint(0, 16, (batch_size,))
    rewards_batch = torch.randn(batch_size)
    next_obs_batch = torch.randn(batch_size, obs_dim)
    dones_batch = torch.zeros(batch_size)

    metrics = agent.update(obs_batch, actions_batch, rewards_batch,
                          next_obs_batch, dones_batch)
    print(f"Training metrics: {metrics}")
