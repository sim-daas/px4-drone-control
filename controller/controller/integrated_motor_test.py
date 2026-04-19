import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, ActuatorMotors
import math

class IntegratedMotorTest(Node):
    def __init__(self):
        super().__init__('integrated_motor_test')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.motor_publisher = self.create_publisher(
            ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)

        # Timer running at 50Hz
        self.timer = self.create_timer(0.02, self.timer_callback)
        self.setpoint_counter = 0
        self.throttle = 0.1
        self.increasing = True

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Arm command sent")

    def engage_offboard_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.get_logger().info("Offboard mode command sent")

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = True
        self.offboard_control_mode_publisher.publish(msg)

    def publish_actuator_motors(self):
        msg = ActuatorMotors()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.timestamp_sample = msg.timestamp
        
        # Simple ramp up and down between 0.1 and 0.3
        if self.increasing:
            self.throttle += 0.002
            if self.throttle >= 0.8:
                self.increasing = False
        else:
            self.throttle -= 0.002
            if self.throttle <= 0.79:
                self.increasing = True

        msg.control = [float('nan')] * 12
        for i in range(4):
            msg.control[i] = self.throttle
        
        msg.reversible_flags = 0
        
        self.motor_publisher.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_publisher.publish(msg)

    def timer_callback(self):
        # 1. ALWAYS publish the offboard control mode and the setpoint simultaneously
        self.publish_offboard_control_mode()
        self.publish_actuator_motors()

        # 2. Engage after a short delay (e.g. 50 ticks = 1 second)
        if self.setpoint_counter == 50:
            self.engage_offboard_mode()
            self.arm()
        
        self.setpoint_counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = IntegratedMotorTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
