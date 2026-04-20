import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, ActuatorMotors, VehicleOdometry
import numpy as np
import math


class PID:
    def __init__(self, kp, ki, kd, max_integral=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.max_integral = max_integral
    
    def update(self, error, dt):
        if dt <= 0.0:
            return 0.0
        self.integral += error * dt
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class RateController(Node):
    def __init__(self):
        super().__init__('rate_controller')

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
        
        self.odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.odometry_callback, qos_profile)

        # 250Hz timer
        self.dt = 1.0 / 250.0
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        self.setpoint_counter = 0
        self.current_rates = np.zeros(3)
        self.origin_set = False

        # Mixing Matrix inverse (from thrust/torques to motor commands)
        # Vector order: [tau_x, tau_y, tau_z, Thrust]
        self.mixer = np.array([
            [-0.43773276,  0.70710677,  0.909091, 1.0],
            [ 0.43773273, -0.70710677,  1.0,      1.0],
            [ 0.43773276,  0.70710677, -0.909091, 1.0],
            [-0.43773273, -0.70710677, -1.0,      1.0]
        ])

        # PID controllers
        self.pid_p = PID(0.1, 0.0001, 0.0001)
        self.pid_q = PID(0.1, 0.0001, 0.0001)
        self.pid_r = PID(0.1, 0.0001, 0.0001)

        # Desired rates and thrust
        self.desired_rates = np.zeros(3)
        self.hover_thrust = 0.735 # Nominal hover thrust

    def odometry_callback(self, msg):
        # Rates are in FRD (Forward-Right-Down) body frame
        self.current_rates[0] = msg.angular_velocity[0] # roll rate (p)
        self.current_rates[1] = msg.angular_velocity[1] # pitch rate (q)
        self.current_rates[2] = msg.angular_velocity[2] # yaw rate (r)
        
        if not self.origin_set and not np.isnan(msg.position[0]):
            self.origin_set = True
            self.get_logger().info("Odometry received. Controller active.")

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
        if not self.origin_set:
            return

        # Hardcoded step input sequence
        time_sec = self.setpoint_counter * self.dt
        if 5.0 < time_sec < 7.0:
            self.desired_rates = np.array([0.0, 0.0, 0.0]) # Roll step placeholder
        else:
            self.desired_rates = np.array([0.0, 0.0, 0.0]) # Stabilize

        # Calculate rate errors (directly in body frame)
        error_p = self.desired_rates[0] - self.current_rates[0]
        error_q = self.desired_rates[1] - self.current_rates[1]
        error_r = self.desired_rates[2] - self.current_rates[2]

        # Calculate Torques
        tau_x = self.pid_p.update(error_p, self.dt)
        tau_y = self.pid_q.update(error_q, self.dt)
        tau_z = self.pid_r.update(error_r, self.dt)

        wrench = np.array([tau_x, tau_y, tau_z, self.hover_thrust])

        # Control Allocation
        u = self.mixer @ wrench
        
        # Constrain motor commands [0, 1]
        u = np.clip(u, 0.0, 1.0)
        
        # Testing: override with fixed throttle
        # u = np.array([0.75, 0.75, 0.75, 0.75])

        # Publish
        self.publish_offboard_control_mode()
        
        msg = ActuatorMotors()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.timestamp_sample = msg.timestamp
        msg.reversible_flags = 0
        
        msg.control = [float('nan')] * 12
        msg.control[0] = u[0]
        msg.control[1] = u[1]
        msg.control[2] = u[2]
        msg.control[3] = u[3]
        
        self.motor_publisher.publish(msg)
        
        # Debug logging every 0.5s
        if self.setpoint_counter % 62.5 == 0:
            self.get_logger().info(
                f"t={time_sec:.1f}s | u={u[0]:.4f},{u[1]:.4f},{u[2]:.4f},{u[3]:.4f} | "
                f"err={error_p:.4f},{error_q:.4f},{error_r:.4f}")

        if self.setpoint_counter == int(1.0 / self.dt):
            self.engage_offboard_mode()
            self.arm()
        
        self.setpoint_counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = RateController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
