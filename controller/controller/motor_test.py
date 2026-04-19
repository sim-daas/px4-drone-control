import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import ActuatorMotors

class MotorTest(Node):
    def __init__(self):
        super().__init__('motor_test')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.motor_publisher = self.create_publisher(
            ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)

        self.timer = self.create_timer(0.02, self.timer_callback) # 50Hz
        self.throttle = 0.0
        self.increasing = True

    def timer_callback(self):
        msg = ActuatorMotors()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        # Simple ramp up and down
        if self.increasing:
            self.throttle += 0.005
            if self.throttle >= 0.3:
                self.increasing = False
        else:
            self.throttle -= 0.005
            if self.throttle <= 0.1:
                self.increasing = True

        # Publish to first 4 motors (quadrotor)
        msg.control = [0.0] * 12
        for i in range(4):
            msg.control[i] = self.throttle
        
        self.motor_publisher.publish(msg)
        # self.get_logger().info(f"Publishing throttle: {self.throttle:.3f}")

def main(args=None):
    rclpy.init(args=args)
    motor_test = MotorTest()
    rclpy.spin(motor_test)
    motor_test.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
