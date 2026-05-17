#!/usr/bin/env python3
# pyrefly: ignore [missing-import]
import rclpy
# pyrefly: ignore [missing-import]
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, TrajectorySetpoint, VehicleOdometry
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Bool
import numpy as np

class LandingController(Node):
    def __init__(self):
        super().__init__('landing_controller')
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # State Variables
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.platform_pos = None
        self.platform_vel = np.zeros(3)
        self.last_marker_seen_time = None
        
        self.state = "IDLE" # IDLE, DESCEND, LAND
        self.target_alt_relative = -5.5 # Start descent from 6m relative
        
        # Publishers
        self.offboard_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        # Subscribers
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos)
        self.platform_pose_sub = self.create_subscription(PoseStamped, '/precision_landing/platform_pose', self.platform_pose_callback, 10)
        # We no longer use visually derived ArUco velocity, we will lock the matched velocity at handover
        self.handover_sub = self.create_subscription(Bool, '/precision_landing/handover', self.handover_callback, 10)

        # Timers
        self.timer = self.create_timer(0.05, self.timer_callback) # 20Hz
        
        self.offboard_setpoint_counter = 0
        self.armed = False
        
        self.get_logger().info("Landing Controller initialized")

    def odom_callback(self, msg):
        self.drone_pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
        self.drone_vel = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]])

    def platform_pose_callback(self, msg):
        self.platform_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.last_marker_seen_time = self.get_clock().now()

    def handover_callback(self, msg):
        if msg.data and self.state == "IDLE":
            self.get_logger().info("Handover received! Waiting for ArUco lock to begin descent...")
            self.state = "WAIT_ARUCO"
            # Lock the perfectly matched drone velocity to use as feedforward!
            self.platform_vel = self.drone_vel.copy()
            self.get_logger().info(f"Locked feedforward velocity: {self.platform_vel[:2]}")

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Arm command sent")

    def disarm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        self.get_logger().info("Disarm command sent")

    def engage_offboard_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.get_logger().info("Offboard mode command sent")

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)

    def timer_callback(self):
        if self.state in ["IDLE", "DONE"]:
            return # Do not publish offboard heartbeats until handover, or after landing

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()
            self.armed = True

        # Publish OffboardControlMode heartbeat
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position = True
        offboard_msg.velocity = True
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        self.offboard_mode_pub.publish(offboard_msg)

        if self.armed:
            self.update_state_machine()

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

    def update_state_machine(self):
        now = self.get_clock().now()
        marker_timeout = (self.last_marker_seen_time is not None) and ((now - self.last_marker_seen_time).nanoseconds / 1e9 > 2.0)
        
        if marker_timeout and self.state not in ["IDLE", "WAIT_ARUCO", "LAND", "DONE"]:
            self.get_logger().warn("ArUco Marker lost during descent! Ascending to re-acquire...")
            self.state = "WAIT_ARUCO"
            self.target_alt_relative = -6.0

        if self.state == "WAIT_ARUCO":
            if self.platform_pos is not None and not marker_timeout:
                self.state = "DESCEND"
                self.get_logger().info("ArUco Locked! Beginning final precision descent...")
            else:
                # Keep holding last known drone position while waiting for ArUco
                self.publish_trajectory_setpoint(self.drone_pos[0], self.drone_pos[1], self.drone_pos[2])

        elif self.state == "DESCEND":
            # Gradually decrease relative altitude
            if self.target_alt_relative < -0.2:
                self.target_alt_relative += 0.05 # Descend at ~1m/s (0.05 * 20Hz)
            
            self.publish_trajectory_setpoint(self.platform_pos[0], self.platform_pos[1], self.platform_pos[2] + self.target_alt_relative,
                                            vx=self.platform_vel[0], vy=self.platform_vel[1], vz=0.5)

            # If very close to platform, land
            z_error = abs(self.drone_pos[2] - self.platform_pos[2])
            if z_error < 0.1:
                self.state = "LAND"
                self.get_logger().info("Touchdown! Disarming...")

        elif self.state == "LAND":
            # Hand over to PX4's internal landing flight mode to handle touchdown and auto-disarm gracefully
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.get_logger().info("PX4 Land mode engaged! Mission complete.")
            self.state = "DONE"

    def publish_trajectory_setpoint(self, x, y, z, vx=np.nan, vy=np.nan, vz=np.nan):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [float(vx), float(vy), float(vz)]
        msg.yaw = np.nan # Maintain north heading
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LandingController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
