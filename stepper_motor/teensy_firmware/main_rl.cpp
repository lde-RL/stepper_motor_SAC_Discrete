/*
 * Teensy 4.1 Stepper Motor RL Control Firmware
 * - 4-bit switch control (16 discrete actions)
 * - 1 kHz control loop
 * - Safety guards (short-through prevention, current/temp limits)
 * - On-board policy inference (optional)
 * - ROS 2 micro-ROS communication
 */

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
#include <std_msgs/msg/uint8.h>
#include <std_msgs/msg/float32.h>

// Custom messages (need to be generated from msg files)
// Placeholder: will be replaced with actual generated types
// #include <stepper_rl_msgs/msg/ref_target.h>
// #include <stepper_rl_msgs/msg/observation.h>
// #include <stepper_rl_msgs/msg/action.h>
// #include <stepper_rl_msgs/msg/encoder_data.h>
// #include <stepper_rl_msgs/msg/health.h>

// ======================= Hardware Configuration =======================
// 4-bit Switch Pins (adjust to your hardware)
#define SWITCH_0  2  // Bit 0
#define SWITCH_1  3  // Bit 1
#define SWITCH_2  4  // Bit 2
#define SWITCH_3  5  // Bit 3

// Encoder pins
#define ENCODER_A 25
#define ENCODER_B 26

// Current sensing ADC pins
#define CURRENT_ADC_PIN A0

// Encoder & motor constants
static const int CPR = 1024;              // Encoder counts per revolution
static const int QUAD_FACTOR = 4;         // Quadrature encoding x4
static const int POLE_PAIRS = 50;         // Motor pole pairs
static const float TWO_PI = 6.28318530718f;

// Control loop timing
static const float CONTROL_DT = 0.001f;   // 1 kHz = 1ms
static const uint32_t CONTROL_US = 1000;  // 1000 microseconds

// Safety limits
static const float MAX_CURRENT_A = 3.0f;
static const float MAX_TEMP_C = 80.0f;
static const float MIN_VOLTAGE_V = 10.0f;
static const float MAX_VOLTAGE_V = 14.0f;

// Preview & history sizes
static const int K_PREVIEW = 32;          // Future target preview length
static const int N_ACTION_HIST = 16;      // Action history length

// Fault flags (bit positions)
#define FAULT_OVERCURRENT   (1 << 0)
#define FAULT_OVERHEAT      (1 << 1)
#define FAULT_UNDERVOLTAGE  (1 << 2)
#define FAULT_OVERVOLTAGE   (1 << 3)
#define FAULT_COMM_TIMEOUT  (1 << 4)
#define FAULT_ILLEGAL_COMBO (1 << 5)

// ======================= State Variables =======================
// Encoder
volatile long encoder_count = 0;
long last_encoder_count = 0;
float mech_angle_rad = 0.0f;
float elec_angle_rad = 0.0f;
float mech_rpm = 0.0f;

// Reference target (from PC)
float theta_ref_now = 0.0f;
float dtheta_ref_now = 0.0f;
float theta_ref_preview[K_PREVIEW] = {0};
uint32_t last_ref_time_ms = 0;

// Action & control
uint8_t current_action_mask = 0;
uint8_t action_history[N_ACTION_HIST] = {0};
uint8_t action_hist_idx = 0;

// Episode tracking
uint32_t episode_step_count = 0;
uint32_t episode_start_ms = 0;

// Safety & health
uint8_t fault_flags = 0;
float supply_voltage = 12.0f;
float temperature_c = 25.0f;
float current_a = 0.0f;

// Policy (on-board inference - placeholder)
uint32_t policy_version = 0;
bool policy_loaded = false;

// ======================= Illegal Action Combinations =======================
// Define illegal 4-bit combinations (short-through prevention)
// Example: both bits in same phase should not be HIGH simultaneously
// Adjust based on your H-bridge configuration
bool is_illegal_action(uint8_t mask) {
  // Example: if bit 0 and bit 1 control same phase, both HIGH is illegal
  // Modify this logic based on your actual wiring

  // Simple example: disallow all bits HIGH (0xF) and all LOW (0x0)
  if (mask == 0x00 || mask == 0x0F) {
    return true;
  }

  // Add more sophisticated checks here based on your motor driver
  // For example:
  // if ((mask & 0x03) == 0x03) return true;  // Bits 0 and 1 both HIGH
  // if ((mask & 0x0C) == 0x0C) return true;  // Bits 2 and 3 both HIGH

  return false;
}

// Apply action with safety guard
void apply_action_safe(uint8_t mask) {
  // Check for illegal combinations
  if (is_illegal_action(mask)) {
    fault_flags |= FAULT_ILLEGAL_COMBO;
    mask = 0x00;  // Safe default: all OFF
  }

  // Apply to hardware
  digitalWrite(SWITCH_0, (mask & 0x01) ? HIGH : LOW);
  digitalWrite(SWITCH_1, (mask & 0x02) ? HIGH : LOW);
  digitalWrite(SWITCH_2, (mask & 0x04) ? HIGH : LOW);
  digitalWrite(SWITCH_3, (mask & 0x08) ? HIGH : LOW);

  current_action_mask = mask;

  // Update action history (circular buffer)
  action_history[action_hist_idx] = mask;
  action_hist_idx = (action_hist_idx + 1) % N_ACTION_HIST;
}

// ======================= Encoder ISR =======================
void ISR_encoder_A() {
  bool A = digitalRead(ENCODER_A);
  bool B = digitalRead(ENCODER_B);
  if (A == B) encoder_count++; else encoder_count--;
}

void ISR_encoder_B() {
  bool A = digitalRead(ENCODER_A);
  bool B = digitalRead(ENCODER_B);
  if (A != B) encoder_count++; else encoder_count--;
}

// ======================= Sensor Reading =======================
void update_sensors() {
  // Encoder angle calculation
  noInterrupts();
  long cnt = encoder_count;
  interrupts();

  long cnt_per_rev = CPR * QUAD_FACTOR;
  float mech_rev = (float)(cnt % cnt_per_rev) / (float)cnt_per_rev;
  mech_angle_rad = mech_rev * TWO_PI;
  elec_angle_rad = fmodf((float)POLE_PAIRS * mech_angle_rad, TWO_PI);

  // RPM calculation (simple differential)
  static uint32_t last_rpm_time = 0;
  uint32_t now = millis();
  if (now != last_rpm_time) {
    float dt = (now - last_rpm_time) / 1000.0f;
    long dcnt = cnt - last_encoder_count;
    mech_rpm = (dcnt / (float)cnt_per_rev) / dt * 60.0f;
    last_encoder_count = cnt;
    last_rpm_time = now;
  }

  // Current sensing (simplified - adjust for your hardware)
  int adc = analogRead(CURRENT_ADC_PIN);
  current_a = (adc / 4095.0f) * 3.3f / 0.1f;  // Example conversion

  // Voltage & temperature (placeholder - implement actual reading)
  supply_voltage = 12.0f;  // Read from voltage divider
  temperature_c = 25.0f;   // Read from temperature sensor
}

// ======================= Safety Check =======================
void check_safety() {
  fault_flags = 0;  // Clear previous faults

  if (current_a > MAX_CURRENT_A) {
    fault_flags |= FAULT_OVERCURRENT;
  }

  if (temperature_c > MAX_TEMP_C) {
    fault_flags |= FAULT_OVERHEAT;
  }

  if (supply_voltage < MIN_VOLTAGE_V) {
    fault_flags |= FAULT_UNDERVOLTAGE;
  }

  if (supply_voltage > MAX_VOLTAGE_V) {
    fault_flags |= FAULT_OVERVOLTAGE;
  }

  // Communication timeout check (no /ref message for >1 second)
  if (millis() - last_ref_time_ms > 1000) {
    fault_flags |= FAULT_COMM_TIMEOUT;
  }

  // If any critical fault, apply safe action
  if (fault_flags != 0) {
    apply_action_safe(0x00);  // All switches OFF
  }
}

// ======================= Policy Inference (Placeholder) =======================
// This will be replaced with actual neural network inference
// For now, returns a simple action based on error
uint8_t infer_policy_action() {
  if (!policy_loaded) {
    return 0x00;  // No policy loaded, safe default
  }

  // Placeholder: simple proportional control
  float error = theta_ref_now - mech_angle_rad;

  // Wrap error to [-PI, PI]
  while (error > PI) error -= TWO_PI;
  while (error < -PI) error += TWO_PI;

  // Map error to action (very naive - replace with NN inference)
  if (error > 0.1f) return 0x01;
  else if (error < -0.1f) return 0x02;
  else return 0x00;
}

// ======================= micro-ROS Setup =======================
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;

// Publishers
rcl_publisher_t obs_pub;      // Observation (for PC training)
rcl_publisher_t encoder_pub;  // Encoder data (training only)
rcl_publisher_t health_pub;   // Health status

// Subscribers
rcl_subscription_t ref_sub;   // Reference target
rcl_subscription_t act_sub;   // Action command (from PC policy)

// Timers
rcl_timer_t control_timer;    // 1 kHz control loop
rcl_timer_t health_timer;     // 10 Hz health broadcast

// Message buffers (using std_msgs as placeholder until custom msgs are built)
std_msgs__msg__UInt8 action_msg;
std_msgs__msg__Float32 ref_msg;

// Error handling
#define RCCHECK(fn) { rcl_ret_t rc = fn; if(rc != RCL_RET_OK) { error_loop(); } }
#define RCSOFTCHECK(fn) { (void)fn; }

void error_loop() {
  while(1) {
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    delay(100);
  }
}

// ======================= Network Transport (UDP) =======================
IPAddress agent_ip(192, 168, 1, 113);
uint16_t agent_port = 8888;
IPAddress local_ip(192, 168, 1, 10);
IPAddress gateway(192, 168, 1, 1);
IPAddress subnet(255, 255, 255, 0);
byte mac[6] = {0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED};
EthernetUDP udp;

bool transport_open(struct uxrCustomTransport *) {
  Ethernet.begin(mac, local_ip, gateway, subnet);
  delay(500);
  return udp.begin(agent_port);
}

bool transport_close(struct uxrCustomTransport *) {
  udp.stop();
  return true;
}

size_t transport_write(struct uxrCustomTransport*, const uint8_t* buf, size_t len, uint8_t*) {
  udp.beginPacket(agent_ip, agent_port);
  udp.write(buf, len);
  udp.endPacket();
  return len;
}

size_t transport_read(struct uxrCustomTransport*, uint8_t* buf, size_t len, int timeout, uint8_t*) {
  unsigned long start = millis();
  while(millis() - start < (unsigned long)timeout) {
    int pk = udp.parsePacket();
    if(pk > 0) {
      int r = udp.read(buf, len);
      return (r > 0) ? (size_t)r : 0;
    }
  }
  return 0;
}

void set_microros_teensy_ethernet_transports() {
  rmw_uros_set_custom_transport(false, NULL, transport_open, transport_close,
                                 transport_write, transport_read);
}

// ======================= ROS Callbacks =======================
void ref_callback(const void * msgin) {
  // Placeholder: will use custom RefTarget message
  const std_msgs__msg__Float32 * msg = (const std_msgs__msg__Float32 *)msgin;
  theta_ref_now = msg->data;
  last_ref_time_ms = millis();
}

void action_callback(const void * msgin) {
  // Receive action from PC policy (during training or remote inference)
  const std_msgs__msg__UInt8 * msg = (const std_msgs__msg__UInt8 *)msgin;
  uint8_t action_mask = msg->data & 0x0F;  // Only 4 bits
  apply_action_safe(action_mask);
}

void control_timer_callback(rcl_timer_t * timer, int64_t last_call_time) {
  RCLC_UNUSED(last_call_time);
  if (timer == NULL) return;

  // 1 kHz control loop
  update_sensors();
  check_safety();

  // If no fault, infer action (on-board policy or wait for PC command)
  if (fault_flags == 0 && policy_loaded) {
    uint8_t action = infer_policy_action();
    apply_action_safe(action);
  }

  // Publish observation (placeholder - will use custom Observation message)
  // For now, skip publishing in timer (too high frequency for ROS)

  episode_step_count++;
}

void health_timer_callback(rcl_timer_t * timer, int64_t last_call_time) {
  RCLC_UNUSED(last_call_time);
  if (timer == NULL) return;

  // Publish health status (placeholder)
  // Will use custom Health message
}

// ======================= Setup =======================
void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);

  // Setup switch pins
  pinMode(SWITCH_0, OUTPUT);
  pinMode(SWITCH_1, OUTPUT);
  pinMode(SWITCH_2, OUTPUT);
  pinMode(SWITCH_3, OUTPUT);

  // Initialize all switches OFF
  apply_action_safe(0x00);

  // Setup encoder
  pinMode(ENCODER_A, INPUT_PULLUP);
  pinMode(ENCODER_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_A), ISR_encoder_A, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER_B), ISR_encoder_B, CHANGE);

  // Setup ADC
  analogReadResolution(12);

  // micro-ROS setup
  set_microros_teensy_ethernet_transports();
  allocator = rcl_get_default_allocator();

  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  RCCHECK(rclc_node_init_default(&node, "teensy_rl_node", "", &support));

  // Initialize subscribers (using std_msgs as placeholder)
  RCCHECK(rclc_subscription_init_default(
    &ref_sub, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "ref_target"));

  RCCHECK(rclc_subscription_init_default(
    &act_sub, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, UInt8), "action_cmd"));

  // Initialize timers
  RCCHECK(rclc_timer_init_default(
    &control_timer, &support, RCL_MS_TO_NS(1), control_timer_callback));

  RCCHECK(rclc_timer_init_default(
    &health_timer, &support, RCL_MS_TO_NS(100), health_timer_callback));

  // Initialize executor
  RCCHECK(rclc_executor_init(&executor, &support.context, 4, &allocator));
  RCCHECK(rclc_executor_add_subscription(&executor, &ref_sub, &ref_msg, ref_callback, ON_NEW_DATA));
  RCCHECK(rclc_executor_add_subscription(&executor, &act_sub, &action_msg, action_callback, ON_NEW_DATA));
  RCCHECK(rclc_executor_add_timer(&executor, &control_timer));
  RCCHECK(rclc_executor_add_timer(&executor, &health_timer));

  episode_start_ms = millis();

  Serial.println("Teensy RL Control Node Ready");
}

// ======================= Main Loop =======================
void loop() {
  // Spin executor
  rclc_executor_spin_some(&executor, RCL_MS_TO_NS(1));

  // Additional monitoring or diagnostics can go here
  static uint32_t last_debug = 0;
  if (millis() - last_debug > 1000) {
    last_debug = millis();
    Serial.print("RPM: "); Serial.print(mech_rpm);
    Serial.print(" | Angle: "); Serial.print(mech_angle_rad);
    Serial.print(" | Faults: 0x"); Serial.println(fault_flags, HEX);
  }
}
