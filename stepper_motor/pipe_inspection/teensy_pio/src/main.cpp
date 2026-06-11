/*
 * Pipe Explorer Firmware — Teensy 4.1
 *
 * Hardware
 * --------
 *   • 3× VL53L0X ToF distance sensors (I2C, XSHUT-gated addressing)
 *       Front: XSHUT → pin 6,  assigned address 0x30
 *       Left : XSHUT → pin 7,  assigned address 0x31
 *       Right: XSHUT → pin 8,  assigned address 0x32
 *   • BNO085 IMU (I2C) — yaw only
 *   • Quadrature wheel encoders on pins 25/26
 *   • micro-ROS over USB (Serial) or Ethernet (swap transport below)
 *
 * Published topics
 * ----------------
 *   /pipe/odom_segment   std_msgs/Float32MultiArray  [delta_dist_m, heading_rad]
 *   /pipe/junction_event std_msgs/Int32MultiArray    [type, n_open, f_mm, l_mm, r_mm]
 *   /pipe/tof_raw        std_msgs/Int32MultiArray    [front_mm, left_mm, right_mm]
 *
 * Junction event types
 * --------------------
 *   0 = START      (published once at boot)
 *   1 = JUNCTION   (≥1 lateral opening detected)
 *   2 = DEAD_END   (front blocked, no side openings)
 *   3 = LOOP_CLOSURE_HINT (sharp heading + distance match — PC validates)
 *
 * Tuning constants are at the top of this file.
 */

#include <Arduino.h>
#include <Wire.h>

// ── micro-ROS ────────────────────────────────────────────────────────
#include <micro_ros_arduino.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/float32_multi_array.h>
#include <std_msgs/msg/int32_multi_array.h>

// ── BNO085 (Adafruit AHRS library) ──────────────────────────────────
#include <Adafruit_BNO08x.h>

// ── VL53L0X (Pololu library) ──────────────────────────────────────
#include <VL53L0X.h>

// ════════════════════════════════════════════════════════════════════
// Tuning constants
// ════════════════════════════════════════════════════════════════════

// ToF: XSHUT pin assignments
static const int TOF_XSHUT_FRONT = 6;
static const int TOF_XSHUT_LEFT  = 7;
static const int TOF_XSHUT_RIGHT = 8;

// ToF: I2C addresses after re-assignment
static const uint8_t TOF_ADDR_FRONT = 0x30;
static const uint8_t TOF_ADDR_LEFT  = 0x31;
static const uint8_t TOF_ADDR_RIGHT = 0x32;

// Wheel encoder
static const int ENC_A_PIN = 25;
static const int ENC_B_PIN = 26;
static const int COUNTS_PER_REV = 1024;      // encoder PPR × 4 (quadrature)
static const float WHEEL_CIRC_M = 0.2199f;   // π × wheel_diameter(m)

// Distance thresholds (mm)
static const int PIPE_OPEN_MM     = 200;  // > this → opening detected (not pipe wall)
static const int PIPE_BLOCKED_MM  = 120;  // < this → wall ahead / dead-end
static const int JUNCTION_HYST_MM =  30;  // hysteresis to avoid re-triggering

// Odometry publish interval (ms)
static const uint32_t ODOM_PERIOD_MS = 200;

// Loop-closure heading tolerance (radians)
static const float CLOSURE_HDG_TOL = 0.25f;

// ════════════════════════════════════════════════════════════════════
// Hardware objects
// ════════════════════════════════════════════════════════════════════

VL53L0X tof_front, tof_left, tof_right;
Adafruit_BNO08x  bno;

// Encoder state
volatile long enc_count = 0;
long enc_last = 0;
float heading_rad = 0.0f;

// ════════════════════════════════════════════════════════════════════
// micro-ROS objects
// ════════════════════════════════════════════════════════════════════

rcl_node_t           node;
rcl_publisher_t      pub_odom;
rcl_publisher_t      pub_junction;
rcl_publisher_t      pub_tof;
rclc_support_t       support;
rcl_allocator_t      allocator;
rclc_executor_t      executor;

std_msgs__msg__Float32MultiArray odom_msg;
std_msgs__msg__Int32MultiArray   junc_msg;
std_msgs__msg__Int32MultiArray   tof_msg;

float odom_data[2];   // [delta_dist_m, heading_rad]
int32_t junc_data[5]; // [type, n_open, f_mm, l_mm, r_mm]
int32_t tof_data[3];  // [front_mm, left_mm, right_mm]

// ════════════════════════════════════════════════════════════════════
// Encoder ISR
// ════════════════════════════════════════════════════════════════════

void enc_a_isr() {
    if (digitalRead(ENC_B_PIN)) enc_count--;
    else                         enc_count++;
}

// ════════════════════════════════════════════════════════════════════
// ToF initialisation with XSHUT gating
// ════════════════════════════════════════════════════════════════════

bool init_tof_sensors() {
    // All sensors off
    digitalWrite(TOF_XSHUT_FRONT, LOW);
    digitalWrite(TOF_XSHUT_LEFT,  LOW);
    digitalWrite(TOF_XSHUT_RIGHT, LOW);
    delay(10);

    // Bring up one at a time, assign unique address
    auto bring_up = [](int xshut_pin, VL53L0X &sensor, uint8_t addr) -> bool {
        digitalWrite(xshut_pin, HIGH);
        delay(10);
        if (!sensor.init()) return false;
        sensor.setAddress(addr);
        sensor.setTimeout(500);
        sensor.startContinuous(50);   // 50 ms between measurements
        return true;
    };

    if (!bring_up(TOF_XSHUT_FRONT, tof_front, TOF_ADDR_FRONT)) return false;
    if (!bring_up(TOF_XSHUT_LEFT,  tof_left,  TOF_ADDR_LEFT )) return false;
    if (!bring_up(TOF_XSHUT_RIGHT, tof_right, TOF_ADDR_RIGHT)) return false;

    return true;
}

// ════════════════════════════════════════════════════════════════════
// BNO085 — yaw-only setup
// ════════════════════════════════════════════════════════════════════

bool init_imu() {
    if (!bno.begin_I2C()) return false;
    bno.enableReport(SH2_ARVR_STABILIZED_RV, 5000);  // 200 Hz game rotation
    return true;
}

float read_yaw_rad() {
    sh2_SensorValue_t val;
    if (!bno.getSensorEvent(&val)) return heading_rad;
    if (val.sensorId != SH2_ARVR_STABILIZED_RV) return heading_rad;

    // Convert quaternion → yaw
    float qr = val.un.arvrStabilizedRV.real;
    float qi = val.un.arvrStabilizedRV.i;
    float qj = val.un.arvrStabilizedRV.j;
    float qk = val.un.arvrStabilizedRV.k;

    float siny = 2.0f * (qr * qk + qi * qj);
    float cosy = 1.0f - 2.0f * (qj * qj + qk * qk);
    return atan2f(siny, cosy);
}

// ════════════════════════════════════════════════════════════════════
// Junction detection state machine
// ════════════════════════════════════════════════════════════════════

enum class DetState { NOMINAL, JUNCTION_PENDING, DEAD_END_PENDING };
DetState det_state = DetState::NOMINAL;
uint32_t det_entry_ms = 0;
static const uint32_t DET_CONFIRM_MS = 300;  // ms to hold before confirming

// Previous readings for hysteresis
int prev_left_mm  = 0;
int prev_right_mm = 0;
int prev_front_mm = 9999;

void publish_junction_event(int etype, int n_open,
                            int f_mm, int l_mm, int r_mm) {
    junc_data[0] = etype;
    junc_data[1] = n_open;
    junc_data[2] = f_mm;
    junc_data[3] = l_mm;
    junc_data[4] = r_mm;

    junc_msg.data.data     = junc_data;
    junc_msg.data.size     = 5;
    junc_msg.data.capacity = 5;
    rcl_publish(&pub_junction, &junc_msg, NULL);
}

void process_junction_detection(int f_mm, int l_mm, int r_mm) {
    bool left_open  = (l_mm > PIPE_OPEN_MM);
    bool right_open = (r_mm > PIPE_OPEN_MM);
    bool front_open = (f_mm > PIPE_BLOCKED_MM);

    int n_open = (int)left_open + (int)right_open + (int)front_open;

    switch (det_state) {
    case DetState::NOMINAL:
        if (!front_open && !left_open && !right_open) {
            // Dead-end candidate
            det_state = DetState::DEAD_END_PENDING;
            det_entry_ms = millis();
        } else if (left_open || right_open) {
            // Junction candidate
            det_state = DetState::JUNCTION_PENDING;
            det_entry_ms = millis();
        }
        break;

    case DetState::JUNCTION_PENDING:
        if (millis() - det_entry_ms > DET_CONFIRM_MS) {
            if (left_open || right_open) {
                // Confirmed junction
                publish_junction_event(1, n_open, f_mm, l_mm, r_mm);
            }
            det_state = DetState::NOMINAL;
        }
        // Cancel if readings changed significantly
        if (!left_open && !right_open) det_state = DetState::NOMINAL;
        break;

    case DetState::DEAD_END_PENDING:
        if (millis() - det_entry_ms > DET_CONFIRM_MS) {
            if (!front_open && !left_open && !right_open) {
                publish_junction_event(2, 0, f_mm, l_mm, r_mm);
            }
            det_state = DetState::NOMINAL;
        }
        if (front_open) det_state = DetState::NOMINAL;
        break;
    }

    prev_left_mm  = l_mm;
    prev_right_mm = r_mm;
    prev_front_mm = f_mm;
}

// ════════════════════════════════════════════════════════════════════
// micro-ROS helpers
// ════════════════════════════════════════════════════════════════════

#define RCCHECK(fn) { rcl_ret_t rc = fn; if (rc != RCL_RET_OK) { return; } }

void setup_microros() {
    set_microros_serial_transports(Serial);
    delay(2000);

    allocator = rcl_get_default_allocator();
    rclc_support_init(&support, 0, NULL, &allocator);

    rclc_node_init_default(&node, "pipe_explorer", "", &support);

    // Publishers
    rclc_publisher_init_default(
        &pub_odom, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
        "/pipe/odom_segment");

    rclc_publisher_init_default(
        &pub_junction, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32MultiArray),
        "/pipe/junction_event");

    rclc_publisher_init_default(
        &pub_tof, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32MultiArray),
        "/pipe/tof_raw");

    rclc_executor_init(&executor, &support.context, 1, &allocator);
}

// ════════════════════════════════════════════════════════════════════
// Arduino setup / loop
// ════════════════════════════════════════════════════════════════════

void setup() {
    Serial.begin(115200);

    // XSHUT pins
    pinMode(TOF_XSHUT_FRONT, OUTPUT);
    pinMode(TOF_XSHUT_LEFT,  OUTPUT);
    pinMode(TOF_XSHUT_RIGHT, OUTPUT);

    // Encoder
    pinMode(ENC_A_PIN, INPUT_PULLUP);
    pinMode(ENC_B_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(ENC_A_PIN), enc_a_isr, RISING);

    Wire.begin();
    Wire.setClock(400000);

    bool tof_ok = init_tof_sensors();
    bool imu_ok = init_imu();

    setup_microros();

    // Publish START event once sensors are ready
    if (tof_ok && imu_ok) {
        publish_junction_event(0, 0, 9999, 9999, 9999);
    }
}

uint32_t last_odom_ms = 0;

void loop() {
    rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10));

    // ── Read sensors ─────────────────────────────────────────────
    int f_mm = (int)tof_front.readRangeContinuousMillimeters();
    int l_mm = (int)tof_left.readRangeContinuousMillimeters();
    int r_mm = (int)tof_right.readRangeContinuousMillimeters();

    // Clamp timeout readings
    if (tof_front.timeoutOccurred() || f_mm > 8190) f_mm = 9999;
    if (tof_left.timeoutOccurred()  || l_mm > 8190) l_mm = 9999;
    if (tof_right.timeoutOccurred() || r_mm > 8190) r_mm = 9999;

    heading_rad = read_yaw_rad();

    // ── Publish raw ToF at ~20 Hz ─────────────────────────────────
    tof_data[0] = f_mm;
    tof_data[1] = l_mm;
    tof_data[2] = r_mm;
    tof_msg.data.data     = tof_data;
    tof_msg.data.size     = 3;
    tof_msg.data.capacity = 3;
    rcl_publish(&pub_tof, &tof_msg, NULL);

    // ── Junction detection ────────────────────────────────────────
    process_junction_detection(f_mm, l_mm, r_mm);

    // ── Odometry at ODOM_PERIOD_MS ────────────────────────────────
    uint32_t now_ms = millis();
    if (now_ms - last_odom_ms >= ODOM_PERIOD_MS) {
        long cnt = enc_count;
        long delta_cnt = cnt - enc_last;
        enc_last = cnt;

        float delta_dist_m = (float)delta_cnt / COUNTS_PER_REV * WHEEL_CIRC_M;

        odom_data[0] = delta_dist_m;
        odom_data[1] = heading_rad;
        odom_msg.data.data     = odom_data;
        odom_msg.data.size     = 2;
        odom_msg.data.capacity = 2;
        rcl_publish(&pub_odom, &odom_msg, NULL);

        last_odom_ms = now_ms;
    }

    delayMicroseconds(500);  // ~2 kHz sensor polling
}
