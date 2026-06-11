# Pipe Inspection — Topological SLAM + CPP Coverage Planner

배관 탐사 로봇용 토폴로지컬 SLAM과 중국인 우편배달부(CPP) 검사 경로 플래너.
Python(ROS 2)과 C++(독립 빌드) 두 가지 구현, 그리고 Teensy 4.1 펌웨어(PlatformIO)를 포함합니다.

## 디렉터리 구조

```
pipe_inspection/
├── graph_map.py          # Python: 토폴로지 맵 (networkx)
├── cpp_planner.py        # Python: CPP 플래너
├── pipe_slam_node.py     # Python: ROS 2 노드 (PC)
├── simulator.py          # Python: 독립 시뮬레이터 (ROS 불필요)
├── cpp/                  # C++17 PC 빌드 (CMake)
│   ├── include/graph_map.hpp
│   ├── include/cpp_planner.hpp
│   └── src/              # 시뮬레이터 + 단위 테스트
├── teensy_pio/           # Teensy 4.1 PlatformIO 프로젝트
│   ├── platformio.ini
│   └── src/main.cpp
└── teensy_firmware/      # (참고용) 동일 펌웨어 단일 파일
```

## C++ PC 빌드 (Linux)

의존성 없음 (header-only, C++17).

```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/pipe_tests                 # 단위 테스트
./build/pipe_simulator medium      # simple | medium | complex
```

## Python 시뮬레이터

```bash
pip install networkx
python3 simulator.py --pipe complex --save map.json --path-save path.json
```

## Teensy 펌웨어 (PlatformIO)

```bash
cd teensy_pio
pio run                 # 빌드
pio run -t upload       # USB 플래시
pio device monitor      # 시리얼 모니터
```

PC 쪽 micro-ROS 에이전트:

```bash
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -b 115200
```

### 하드웨어 배선

| 부품 | 연결 |
|---|---|
| VL53L0X ×3 (전/좌/우) | I2C (SDA/SCL 공유), XSHUT → 핀 6/7/8 |
| BNO085 IMU | I2C |
| 휠 엔코더 (쿼드러처) | A → 핀 25, B → 핀 26 |

`platformio.ini`의 `lib_deps`가 micro_ros_arduino, Adafruit BNO08x,
Pololu VL53L0X를 자동으로 받아옵니다.
펌웨어 상단의 상수(`WHEEL_CIRC_M`, `COUNTS_PER_REV`, ToF 임계값)를
실제 하드웨어에 맞게 조정하세요.

## ROS 2 인터페이스

| 토픽/서비스 | 타입 | 방향 | 내용 |
|---|---|---|---|
| `/pipe/odom_segment` | Float32MultiArray | Teensy→PC | `[delta_dist_m, heading_rad]` |
| `/pipe/junction_event` | Int32MultiArray | Teensy→PC | `[type, n_open, f_mm, l_mm, r_mm]` |
| `/pipe/tof_raw` | Int32MultiArray | Teensy→PC | ToF 원시값 3채널 |
| `/pipe/graph_json` | String (latched) | PC | 전체 맵 JSON |
| `/pipe/path_json` | String (latched) | PC | CPP 경로 JSON |
| `/pipe/plan_inspection` | Trigger | 서비스 | CPP 실행 |
| `/pipe/save_map` | Trigger | 서비스 | `/tmp/pipe_map.json` 저장 |
| `/pipe/reset` | Trigger | 서비스 | 맵 초기화 |

## 알고리즘

1. **탐사**: 엔코더+IMU 데드레코닝으로 엣지(구간 길이/방위)를 측정,
   ToF 분기점 감지 이벤트마다 노드를 추가. 기존 노드 0.25 m 이내에
   도달하면 루프 클로저로 병합.
2. **CPP 경로 생성**: 홀수 차수 노드를 찾고, 다익스트라 최단경로 거리로
   최소 가중 완전 매칭(C++은 비트마스크 DP 정해, 홀수 노드 ≤24개)을
   풀어 deadhead 엣지를 복제 → Hierholzer로 오일러 회로 추출.
3. 결과: 모든 배관을 최소 재주행으로 한 번 이상 지나 입구로 복귀하는
   닫힌 경로.
