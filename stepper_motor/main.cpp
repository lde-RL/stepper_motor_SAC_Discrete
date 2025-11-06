#include <Arduino.h>
#include <NativeEthernet.h>
#include <NativeEthernetUdp.h>
#include <micro_ros_platformio.h>
#include <uxr/client/transport.h>
#include <rmw_microros/rmw_microros.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <rclc/timer.h>
#include <std_msgs/msg/string.h>
#include <std_msgs/msg/float32.h>
#include <std_msgs/msg/int32.h>

// ======================= 하드웨어 핀/상수 (⚠️ 보드에 맞게 수정) =======================
// Phase A full-bridge
#define PH_A_HS 2     // Phase-A high-side PWM 핀
#define PH_A_LS 3     // Phase-A low-side  PWM 핀
// Phase B full-bridge
#define PH_B_HS 4     // Phase-B high-side PWM 핀
#define PH_B_LS 5     // Phase-B low-side  PWM 핀

// ADC 핀 (전류센싱)
#define IA_ADC_PIN A0   // Phase-A 전류(증폭된 샨트 전압)
#define IB_ADC_PIN A1   // Phase-B 전류(증폭된 샨트 전압)

// 엔코더 핀
#define ENCODER_A 25
#define ENCODER_B 26

// PWM/ADC/전류 변환 파라미터 (예시값, ⚠️ 반드시 캘리브레이션)
static const float VBUS         = 12.0f;   // 드라이버 버스전압(V)
static const float SHUNT_OHMS   = 0.1f;    // 샨트저항(Ω)
static const float AMP_GAIN     = 20.0f;   // 전류증폭 게인(무차원)
static const float ADC_REF      = 3.3f;    // ADC 참조전압
static const int   ADC_MAX      = 4095;    // 12bit
// i[A] = (ADC/ADC_MAX * ADC_REF) / (SHUNT_OHMS * AMP_GAIN)

// PWM 설정
static const int   PWM_RES_BITS = 12;      // 0..4095
static const float PWM_FREQ_HZ  = 20000.0f; // 20 kHz 캐리어

// 스텝퍼/기계-전기각 변환
static const int   CPR            = 1024;  // 엔코더 PPR
static const int   QUAD_FACTOR    = 4;     // AB x4
static const int   STEPS_PER_REV  = 200;   // (참고) 스테퍼 기본 스텝수
static const int   POLE_PAIRS     = 50;    // ⚠️ 스테퍼의 전기적 pole pair (모델별 상이: 꼭 맞추세요)

// 제어 주기
static const float FOC_TS         = 1.0f / 20000.0f; // 20 kHz 전류루프
static const float SPD_TS         = 1.0f / 1000.0f;  // 1 kHz 속도루프 (타이머로 분리)

// ======================= micro-ROS 객체 =======================
rclc_executor_t executor;
rclc_support_t  support;
rcl_allocator_t allocator;
rcl_node_t      node;

// 퍼블리셔/구독자
rcl_publisher_t hello_pub;
rcl_publisher_t rpm_pub;     // /encoder_rpm (Float32)
rcl_publisher_t id_pub;      // /id_meas     (Float32)
rcl_publisher_t iq_pub;      // /iq_meas     (Float32)

rcl_subscription_t spd_sub;  // /speed_cmd   (Float32)

// 타이머 (ROS용)
rcl_timer_t hello_timer;     // 1 Hz
rcl_timer_t pub_timer;       // 100 Hz : 상태 퍼블리시(부담 줄임)

// 메시지 버퍼
std_msgs__msg__String  hello_msg;
std_msgs__msg__Float32 rpm_msg, id_msg, iq_msg;

// 오류 처리
#define RCCHECK(fn)      { rcl_ret_t rc = fn; if(rc != RCL_RET_OK) { error_loop(); } }
#define RCSOFTCHECK(fn)  { (void)fn; }
void error_loop(){
  while(1){ digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); delay(100); }
}

// ======================= 네트워크(UDP) 전송 설정 =======================
IPAddress agent_ip(192,168,1,113);
uint16_t  agent_port = 8888;
IPAddress local_ip(192,168,1,10);
IPAddress gateway (192,168,1,1);
IPAddress subnet  (255,255,255,0);
byte mac[6] = {0xDE,0xAD,0xBE,0xEF,0xFE,0xED};
EthernetUDP udp;

bool transport_open(struct uxrCustomTransport *){ Ethernet.begin(mac, local_ip, gateway, subnet); delay(500); return udp.begin(agent_port); }
bool transport_close(struct uxrCustomTransport *){ udp.stop(); return true; }
size_t transport_write(struct uxrCustomTransport*, const uint8_t* buf, size_t len, uint8_t*){ udp.beginPacket(agent_ip, agent_port); udp.write(buf, len); udp.endPacket(); return len; }
size_t transport_read(struct uxrCustomTransport*, uint8_t* buf, size_t len, int timeout, uint8_t*){
  unsigned long start = millis();
  while(millis() - start < (unsigned long)timeout){
    int pk = udp.parsePacket();
    if(pk>0){ int r = udp.read(buf, len); return (r>0)? (size_t)r : 0; }
  }
  return 0;
}
void set_microros_teensy_ethernet_transports(){
  rmw_uros_set_custom_transport(false, NULL, transport_open, transport_close, transport_write, transport_read);
}

// ======================= 엔코더(기계각/속도) =======================
volatile long encoder_count = 0;
void ISR_A(){
  bool A = digitalRead(ENCODER_A);
  bool B = digitalRead(ENCODER_B);
  if(A == B) encoder_count++; else encoder_count--;
}
void ISR_B(){
  bool A = digitalRead(ENCODER_A);
  bool B = digitalRead(ENCODER_B);
  if(A != B) encoder_count++; else encoder_count--;
}

long     last_count = 0;
uint32_t last_us_spd = 0;
float    mech_rpm = 0.0f;

// 1 kHz 속도계산 (loop에서 주기적으로 호출)
void update_speed_mech_rpm(){
  uint32_t now = micros();
  float dt = (now - last_us_spd) / 1e6f; if(dt <= 0) dt = SPD_TS;
  noInterrupts(); long cnt = encoder_count; interrupts();
  long dcnt = cnt - last_count;
  float rev = (float)dcnt / (float)(CPR * QUAD_FACTOR);
  mech_rpm = (rev / dt) * 60.0f;
  last_count = cnt; last_us_spd = now;
}

// 전기각 θₑ (rad)
inline float mech_count_to_elec_angle_rad(long cnt){
  float mech_rev = (float)(cnt % (CPR * QUAD_FACTOR)) / (float)(CPR * QUAD_FACTOR); // 0..1
  float mech_theta = mech_rev * 2.0f * PI;      // 기계각 [0..2π)
  return (float)POLE_PAIRS * mech_theta;        // θe = p * θm
}

// ======================= 전류/FOC 제어 =======================
IntervalTimer foc_timer; // 20 kHz ISR

// 제어상태
volatile float rpm_ref = 0.0f;     // /speed_cmd
volatile float iq_ref  = 0.0f;     // 속도루프 결과
volatile float id_ref  = 0.0f;     // 보통 0 (필요시 미세 음전류)

float id_meas = 0.0f, iq_meas = 0.0f;

// PI 게인(예시값 → 반드시 튜닝)
float Kp_id = 0.5f, Ki_id = 200.0f;
float Kp_iq = 0.5f, Ki_iq = 200.0f;
float int_id = 0.0f, int_iq = 0.0f;

// 속도루프 (1 kHz)
float Kp_spd = 0.02f, Ki_spd = 1.0f;
float int_spd = 0.0f;
float iq_ref_limit = 3.0f;   // [A] 제한 (드라이버/열에 맞춰 조정)

// 유틸
inline float clampf(float x, float lo, float hi){ return (x<lo)?lo:((x>hi)?hi:x); }

// ADC → 전류[A]
inline float adc_to_current(int adc){
  float v = ( (float)adc / (float)ADC_MAX ) * ADC_REF;
  return v / (SHUNT_OHMS * AMP_GAIN);
}

// dq → AB 전압지령 (정규화: -1..+1) 후 → 풀브리지 PWM 듀티
void apply_voltage_AB(float vA, float vB){
  // vA, vB는 -1..+1 범위로 정규화된 전압명령이라고 가정
  vA = clampf(vA, -1.0f, 1.0f);
  vB = clampf(vB, -1.0f, 1.0f);

  auto set_phase = [&](int hs_pin, int ls_pin, float vx){
    // 양수: HS PWM, LS OFF / 음수: LS PWM, HS OFF (간단 스킴, 실기는 데드타임 권장)
    int duty = (int)(fabsf(vx) * ((1<<PWM_RES_BITS)-1));
    if(vx >= 0){
      analogWrite(hs_pin, duty);
      analogWrite(ls_pin, 0);
    }else{
      analogWrite(hs_pin, 0);
      analogWrite(ls_pin, duty);
    }
  };

  set_phase(PH_A_HS, PH_A_LS, vA);
  set_phase(PH_B_HS, PH_B_LS, vB);
}

// FOC 20 kHz ISR: 전류 측정 → dq 변환 → PI → 역변환 → PWM
void foc_isr(){
  // 1) 전류 측정
  int adcA = analogRead(IA_ADC_PIN);
  int adcB = analogRead(IB_ADC_PIN);
  float iA = adc_to_current(adcA);
  float iB = adc_to_current(adcB);

  // 2) 전기각
  noInterrupts(); long cnt = encoder_count; interrupts();
  float theta_e = mech_count_to_elec_angle_rad(cnt);
  float s = sinf(theta_e), c = cosf(theta_e);

  // 3) Park 변환 (2상 → dq)
  float i_d =  c * iA + s * iB;
  float i_q = -s * iA + c * iB;
  id_meas = i_d; iq_meas = i_q;

  // 4) 전류 루프 PI
  float err_d = id_ref - i_d;
  float err_q = iq_ref - i_q;
  int_id += err_d * FOC_TS;
  int_iq += err_q * FOC_TS;

  // 간단 앤티윈드업
  int_id = clampf(int_id, -5.0f, 5.0f);
  int_iq = clampf(int_iq, -5.0f, 5.0f);

  float v_d = Kp_id*err_d + Ki_id*int_id;
  float v_q = Kp_iq*err_q + Ki_iq*int_iq;

  // 5) dq → AB (역Park)
  float vA =  c * v_d - s * v_q;
  float vB =  s * v_d + c * v_q;

  // 6) 전압 정규화 및 출력 (버스전압 기준 스케일 필요 → 여기선 단순 노멀라이즈 가정)
  // 실제로는 v_d, v_q 한계(전압 서클 클리핑/SVPWM 유사)와 VBUS 스케일링 적용 권장
  float norm = 1.0f / (VBUS * 0.5f); // 예: 임시 스케일
  apply_voltage_AB(vA * norm, vB * norm);
}

// 1 kHz: 속도루프 (mech rpm → iq_ref)
void speed_loop_tick(){
  update_speed_mech_rpm();

  float err = rpm_ref - mech_rpm;
  int_spd += err * SPD_TS;
  int_spd = clampf(int_spd, -50.0f, 50.0f);

  float iq_cmd = Kp_spd*err + Ki_spd*int_spd;
  iq_ref = clampf(iq_cmd, -iq_ref_limit, iq_ref_limit);
}

// ======================= ROS 콜백/타이머 =======================
void hello_timer_cb(rcl_timer_t*, int64_t){
  static char buf[64];
  hello_msg.data.data = buf;
  hello_msg.data.capacity = sizeof(buf);
  int n = snprintf(buf, sizeof(buf), "Hello(FOC) Teensy 4.1");
  hello_msg.data.size = (n>0)? (size_t)n : 0;
  RCSOFTCHECK(rcl_publish(&hello_pub, &hello_msg, NULL));
}

void pub_timer_cb(rcl_timer_t*, int64_t){
  rpm_msg.data = mech_rpm;
  id_msg.data  = id_meas;
  iq_msg.data  = iq_meas;
  RCSOFTCHECK(rcl_publish(&rpm_pub, &rpm_msg, NULL));
  RCSOFTCHECK(rcl_publish(&id_pub,  &id_msg,  NULL));
  RCSOFTCHECK(rcl_publish(&iq_pub,  &iq_msg,  NULL));
}

void speed_cmd_cb(const void * msgin){
  const std_msgs__msg__Float32 * in = (const std_msgs__msg__Float32 *)msgin;
  rpm_ref = in->data; // 목표 속도(RPM), 부호=방향
}

// ======================= setup/loop =======================
void setup(){
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);

  // PWM 설정
  analogWriteResolution(PWM_RES_BITS);
  analogWriteFrequency(PH_A_HS, PWM_FREQ_HZ);
  analogWriteFrequency(PH_A_LS, PWM_FREQ_HZ);
  analogWriteFrequency(PH_B_HS, PWM_FREQ_HZ);
  analogWriteFrequency(PH_B_LS, PWM_FREQ_HZ);

  pinMode(PH_A_HS, OUTPUT); pinMode(PH_A_LS, OUTPUT);
  pinMode(PH_B_HS, OUTPUT); pinMode(PH_B_LS, OUTPUT);
  analogWrite(PH_A_HS, 0); analogWrite(PH_A_LS, 0);
  analogWrite(PH_B_HS, 0); analogWrite(PH_B_LS, 0);

  // ADC
  analogReadResolution(12);

  // 엔코더
  pinMode(ENCODER_A, INPUT_PULLUP);
  pinMode(ENCODER_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_A), ISR_A, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER_B), ISR_B, CHANGE);
  last_us_spd = micros();

  // micro-ROS
  set_microros_teensy_ethernet_transports();
  allocator = rcl_get_default_allocator();
  RCCHECK( rclc_support_init(&support, 0, NULL, &allocator) );
  RCCHECK( rclc_node_init_default(&node, "teensy_foc_node", "", &support) );

  RCCHECK( rclc_publisher_init_default(
    &hello_pub, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String), "Hello_world") );

  RCCHECK( rclc_publisher_init_default(
    &rpm_pub,  &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "encoder_rpm") );
  RCCHECK( rclc_publisher_init_default(
    &id_pub,   &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "id_meas") );
  RCCHECK( rclc_publisher_init_default(
    &iq_pub,   &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "iq_meas") );

  RCCHECK( rclc_subscription_init_default(
    &spd_sub,  &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "speed_cmd") );

  RCCHECK( rclc_timer_init_default(&hello_timer, &support, RCL_MS_TO_NS(1000), hello_timer_cb) );
  RCCHECK( rclc_timer_init_default(&pub_timer,   &support, RCL_MS_TO_NS(10),   pub_timer_cb) ); // 100 Hz

  RCCHECK( rclc_executor_init(&executor, &support.context, 3, &allocator) );
  RCCHECK( rclc_executor_add_timer(&executor, &hello_timer) );
  RCCHECK( rclc_executor_add_timer(&executor, &pub_timer) );
  RCCHECK( rclc_executor_add_subscription(&executor, &spd_sub, (void*)&rpm_msg, speed_cmd_cb, ON_NEW_DATA) );

  // FOC ISR 시작 (20 kHz)
  foc_timer.begin(foc_isr, (unsigned long)(1e6f / PWM_FREQ_HZ));
}

void loop(){
  // ROS 실행
  rclc_executor_spin_some(&executor, RCL_MS_TO_NS(1));

  // 1 kHz 속도루프 (간단히 소프트 타임슬라이싱)
  static uint32_t last = 0;
  uint32_t now = micros();
  if(now - last >= (uint32_t)(1e6f * SPD_TS)){
    last = now;
    speed_loop_tick();
  }
}




// #include <Arduino.h>
// #include <NativeEthernet.h>
// #include <NativeEthernetUdp.h>
// #include <micro_ros_platformio.h>
// #include <uxr/client/transport.h>
// #include <rmw_microros/rmw_microros.h>
// #include <stdio.h>
// #include <rcl/rcl.h>
// #include <rcl/error_handling.h>
// #include <rclc/rclc.h>
// #include <rclc/executor.h>
// #include <rclc/timer.h>
// #include <std_msgs/msg/string.h>
// #include <std_msgs/msg/int32.h>


// //-------micro_ROS 설정-------//
// rclc_executor_t executor;
// '''
// rclc의 실행기. 콜백(타이머·구독)을 한곳에 등록해 돌립니다. 
// rclc_executor_init(...)로 초기화하고 rclc_executor_add_timer(...), 
// rclc_executor_add_subscription(...)으로 콜백을 붙인 뒤, rclc_executor_spin_some(...)로 실행합니다.
// '''
// rclc_support_t support;
// '''
// 컨텍스트/메모리/미들웨어 설정을 묶는 지원 객체. 
// 반드시 rclc_support_init(...)로 먼저 초기화한 뒤, 이걸 바탕으로 노드/타이머/퍼블리셔 등을 만듭니다
// '''
// rcl_allocator_t allocator;
// '''
// rcl 계층에서 쓰는 메모리 할당자. 보통 rcl_get_default_allocator()로 기본 할당자를 받아서 rclc_support_init(...) 등에 넘깁니다.
// '''
// rcl_node_t node;
// rcl_timer_t timerS;
// rcl_timer_t timerE;

// rcl_publisher_t start_publisher;
// std_msgs__msg__String msg;

// rcl_publisher_t encoder_publisher;
// std_msgs__msg__Int32 encoder_msg;

// rcl_subscription_t motor_subscriber;
// std_msgs__msg__Int32 motor_msg;

// // 모터 제어용 변수
// int target_speed = 0;

// #define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}
// #define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){}}

// void error_loop(){
//   while(1){
//     digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
//     delay(100);
//   }
// }

// void timer_callback(rcl_timer_t * timerS, int64_t last_call_timeS) {
//   RCLC_UNUSED(last_call_timeS);
//   if (timerS != NULL) {
//     sprintf(msg.data.data, "Hello, world! from Teensy 4.1");
//     msg.data.size = strlen(msg.data.data);
//     RCSOFTCHECK(rcl_publish(&start_publisher, &msg, NULL));
//   }
// }

// //------Encoder 설정------//
// #define ENCODER_A 25
// #define ENCODER_B 26
// volatile long encoder_count = 0;

// void readEncoder() {
//   if (digitalRead(ENCODER_A) == digitalRead(ENCODER_B)) {
//     encoder_count++;
//   } else {
//     encoder_count--;
//   }
// }

// void encoder_callback(rcl_timer_t *timerE, int64_t last_call_timeE) {
//   RCLC_UNUSED(last_call_timeE);
//   if (timerE != NULL) {
//     encoder_msg.data = encoder_count;
//     RCSOFTCHECK(rcl_publish(&encoder_publisher, &encoder_msg, NULL));
//   }
// }

// //------Step motor 설정------//
// const int ENA = 9;
// const int ENB = 10;
// const int IN1 = 27;
// const int IN2 = 28;
// const int IN3 = 29;
// const int IN4 = 30;

// const long stepInterval = 10;
// const long directionInterval = 3000; // 3초로 늘려서 방향 전환 확인 용이하게 함

// int currentStep = 0;
// bool isForward = true;

// unsigned long lastStepTime = 0;
// unsigned long lastDirectionToggleTime = 0;

// const int stepSequence[4][4] = {
//   {HIGH, LOW,  HIGH, LOW},
//   {LOW,  HIGH, HIGH, LOW},
//   {LOW,  HIGH, LOW,  HIGH},
//   {HIGH, LOW,  LOW,  HIGH}
// };

// void executeStep(int step) {
//   digitalWrite(IN1, stepSequence[step][0]);
//   digitalWrite(IN2, stepSequence[step][1]);
//   digitalWrite(IN3, stepSequence[step][2]);
//   digitalWrite(IN4, stepSequence[step][3]);
// }

// void Stepping() {
//   unsigned long currentTime = millis();

//   if (currentTime - lastDirectionToggleTime >= directionInterval) {
//     isForward = !isForward;
//     lastDirectionToggleTime = currentTime;
//   }

//   if (currentTime - lastStepTime >= stepInterval) {
//     lastStepTime = currentTime;
//     if (isForward) {
//       currentStep++;
//       if (currentStep > 3) currentStep = 0;
//     } else {
//       currentStep--;
//       if (currentStep < 0) currentStep = 3;
//     }
//     executeStep(currentStep);
//   }
// }

// void motor_subscription_callback(const void * msgin)
// {
//   const std_msgs__msg__Int32 * incoming = (const std_msgs__msg__Int32 *)msgin;
//   target_speed = incoming->data;   // PC에서 보낸 값 저장
//   Serial.print("Received motor_cmd: ");
//   Serial.println(target_speed);

//   // TODO: target_speed 기반으로 Stepping() 내부 속도 제어 변수 갱신
// }

// //------Ethernet 설정------//
// // Agent 설정
// IPAddress agent_ip(192, 168, 1, 113);
// uint16_t agent_port = 8888;

// // Local 설정
// IPAddress local_ip(192, 168, 1, 10);
// IPAddress gateway(192, 168, 1, 1);
// IPAddress subnet(255, 255, 255, 0);
// byte mac[6] = {0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED};

// // UDP 객체
// EthernetUDP udp;

// // ---- transport callbacks ----
// bool transport_open(struct uxrCustomTransport * transport)
// {
//   (void) transport;
//   Ethernet.begin(mac, local_ip, gateway, subnet);
//   delay(1000);
//   return udp.begin(agent_port);   // UDP 시작 (true/false 반환)
// }

// bool transport_close(struct uxrCustomTransport * transport)
// {
//   (void) transport;
//   udp.stop();
//   return true;
// }

// size_t transport_write(struct uxrCustomTransport* transport,
//                        const uint8_t* buf, size_t len, uint8_t* err)
// {
//   (void) transport;
//   (void) err;
//   udp.beginPacket(agent_ip, agent_port);
//   udp.write(buf, len);
//   udp.endPacket();
//   return len;
// }

// size_t transport_read(struct uxrCustomTransport* transport,
//                       uint8_t* buf, size_t len, int timeout, uint8_t* err)
// {
//   (void) transport;
//   (void) err;

//   unsigned long start = millis();
//   while ((millis() - start) < (unsigned long)timeout) {
//     int packetSize = udp.parsePacket();
//     if (packetSize > 0) {
//       int read_len = udp.read(buf, len);
//       return (read_len > 0) ? read_len : 0;
//     }
//   }
//   return 0;
// }

// // ---- transport 등록 ----
// void set_microros_teensy_ethernet_transports()
// {
//   rmw_uros_set_custom_transport(
//       false,                      // true=Serial, false=UDP/TCP
//       NULL,                       // transport args (없으면 NULL)
//       transport_open,
//       transport_close,
//       transport_write,
//       transport_read);
// }

// void setup() {
//   Serial.begin(115200);

//   set_microros_teensy_ethernet_transports();

  
//   pinMode(LED_BUILTIN, OUTPUT);
  
//   //-------Step motor Pin-------//
//   pinMode(ENA, OUTPUT);
//   pinMode(ENB, OUTPUT);
//   pinMode(IN1, OUTPUT);
//   pinMode(IN2, OUTPUT);
//   pinMode(IN3, OUTPUT);
//   pinMode(IN4, OUTPUT);
//   digitalWrite(ENA, HIGH);
//   digitalWrite(ENB, HIGH);

//   //-------Encoder Pin-------//
//   pinMode(ENCODER_A, INPUT_PULLUP);
//   pinMode(ENCODER_B, INPUT_PULLUP);
//   attachInterrupt(digitalPinToInterrupt(ENCODER_A), readEncoder, CHANGE);

//   //-------micro_ROS setup-------//
//   allocator = rcl_get_default_allocator();

//   RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
//   RCCHECK(rclc_node_init_default(&node, "micro_ros_teensy_node", "", &support));

//   // --- "Hello_world" 퍼블리셔 초기화 ---
//   RCCHECK(rclc_publisher_init_default(
//     &start_publisher,
//     &node,
//     ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String),
//     "Hello_world"));
  
//   // --- "encoder_data" 퍼블리셔 초기화 ---
//   RCCHECK(rclc_publisher_init_default(
//     &encoder_publisher,
//     &node,
//     ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
//     "encoder_data"));

//   // --- "motor_cmd" 구독자 초기화 ---
//   RCCHECK(rclc_subscription_init_default(
//     &motor_subscriber,
//     &node,
//     ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
//     "motor_cmd"));
  
//   // [위치 변경] 메시지 버퍼는 퍼블리셔 생성 직후에 할당하는 것이 안전합니다.
//   static char msg_buffer[50];
//   msg.data.data = msg_buffer;
//   msg.data.capacity = 50;  

//   // --- 타이머 초기화 ---
//   RCCHECK(rclc_timer_init_default(
//     &timerS,
//     &support,
//     RCL_MS_TO_NS(1000),
//     timer_callback));

//   RCCHECK(rclc_timer_init_default(
//     &timerE,
//     &support,
//     RCL_MS_TO_NS(10),
//     encoder_callback));  

//   // --- 실행기(Executor) 초기화 ---
//   RCCHECK(rclc_executor_init(&executor, &support.context, 3, &allocator)); // 타이머 2개 + 구독 1개

//   RCCHECK(rclc_executor_add_timer(&executor, &timerS));
//   RCCHECK(rclc_executor_add_timer(&executor, &timerE));
//   RCCHECK(rclc_executor_add_subscription(
//     &executor, &motor_subscriber, &motor_msg,
//     &motor_subscription_callback, ON_NEW_DATA));
// }

// void loop() {
//   rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100));
//   Stepping();
// }