import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from std_msgs.msg import String, Int32

ENCODER_RESOLUTION = 1024  # 엔코더 PPR(펄스/회전), 실제 값으로 맞춰주세요


class PC_Controller(Node):

    def __init__(self):
        super().__init__('pc_controller')

        # ---- 구독자 ----
        # 디버그용 Hello_world
        self.hello_sub = self.create_subscription(
            String,
            'Hello_world',
            self.hello_callback,
            10)

        # 엔코더 데이터
        self.encoder_sub = self.create_subscription(
            Int32,
            'encoder_data',
            self.encoder_callback,
            10)

        # ---- 퍼블리셔 ----
        self.motor_pub = self.create_publisher(
            Int32,
            'motor_cmd',
            10)

        # 내부 상태
        self.last_count = None
        self.last_time = self.get_clock().now()

    # 디버깅용 콜백
    def hello_callback(self, msg: String):
        self.get_logger().info(f'[Hello_world] "{msg.data}"')

    # 엔코더 콜백
    def encoder_callback(self, msg: Int32):
        current_count = msg.data
        current_time = self.get_clock().now()

        if self.last_count is not None:
            dt = (current_time - self.last_time).nanoseconds / 1e9
            dcount = current_count - self.last_count

            # 위치(도 단위)
            position_deg = (current_count % ENCODER_RESOLUTION) * (360.0 / ENCODER_RESOLUTION)

            # 속도(RPM)
            speed_rpm = (dcount / ENCODER_RESOLUTION) / dt * 60.0

            self.get_logger().info(
                f'[Encoder] count={current_count}, pos={position_deg:.2f} deg, speed={speed_rpm:.2f} RPM'
            )

            # 모터 제어 명령 생성 (예시: 속도를 그대로 전달)
            cmd = Int32()
            cmd.data = int(speed_rpm)  # 필요에 맞게 수정 가능
            self.motor_pub.publish(cmd)

        # 상태 업데이트
        self.last_count = current_count
        self.last_time = current_time


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PC_Controller()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

