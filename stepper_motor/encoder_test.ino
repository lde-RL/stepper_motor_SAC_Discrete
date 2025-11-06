#define ENCODER_A 6
#define ENCODER_B 7

volatile long encoderCount = 0;
const int CPR = 1024;          // 엔코더 해상도
const int QUAD_FACTOR = 4;     // Quadrature Encoding
const float DEG_PER_COUNT = (360.0) / (CPR * QUAD_FACTOR);  // 디그리 단위로 변경

unsigned long prevTime = 0;
float prevDeg = 0.0;

void setup() {
  Serial.begin(115200);

  pinMode(ENCODER_A, INPUT_PULLUP);
  pinMode(ENCODER_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENCODER_A), ISR_A, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER_B), ISR_B, CHANGE);

  prevTime = micros();
}

void loop() {
  unsigned long currentTime = micros();
  float deltaTime = (currentTime - prevTime) / 1.0e6; // 초 단위

  noInterrupts(); // 인터럽트 임시 비활성화
  long count = encoderCount;
  interrupts(); // 인터럽트 재활성화

  float currentDeg = count * DEG_PER_COUNT;
  float degPerSec = (currentDeg - prevDeg) / deltaTime;

  // 현재 값 출력
  Serial.print("Angle (deg): ");
  Serial.print(currentDeg, 2);
  Serial.print("\t Speed (deg/s): ");
  Serial.println(degPerSec, 2);

  prevDeg = currentDeg;
  prevTime = currentTime;

  delay(50); // 20Hz 주기
}

void ISR_A() {
  bool A = digitalRead(ENCODER_A);
  bool B = digitalRead(ENCODER_B);
  
  if (A == B) encoderCount++;
  else encoderCount--;
}

void ISR_B() {
  bool A = digitalRead(ENCODER_A);
  bool B = digitalRead(ENCODER_B);

  if (A != B) encoderCount++;
  else encoderCount--;
}
