#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, TrajectorySetpoint, VehicleOdometry
from std_msgs.msg import String, Bool, Float64
import numpy as np
import json

class MissionOrchestrator(Node):
    def __init__(self):
        super().__init__('mission_orchestrator')
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # State Variables
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.drone_yaw = 0.0
        
        self.state = "TAKEOFF" # TAKEOFF, MATCH_VELOCITY, SERVO_DESCEND, HANDOVER
        self.search_altitude = -10.0 # Fly at 15m altitude initially
        self.tracking_altitude = -5.5 # Descend to 6m when tracking
        self.current_target_altitude = self.search_altitude
        
        self.target_pos = np.zeros(3)
        self.target_yaw = 0.0
        self.start_pos = None

        self.car_pos = np.zeros(3)
        self.car_vel = np.zeros(3)
        self.car_nx = 0.0
        self.car_ny = 0.0
        self.car_move = False
        self.last_det_time = None
        
        self.global_car_vel = 2.0 # Set this to test different velocities
        
        self.sync_start_time = None
        
        # Publishers
        self.offboard_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)
        self.handover_pub = self.create_publisher(Bool, '/precision_landing/handover', 10)
        self.enable_aruco_pub = self.create_publisher(Bool, '/precision_landing/enable_aruco', 10)
        self.platform_cmd_vel_pub = self.create_publisher(Float64, '/model/pickup_aruco/joint/rail_joint/cmd_vel', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos)
        self.det_sub = self.create_subscription(String, '/det', self.det_callback, 10)
        self.global_vel_sub = self.create_subscription(Float64, '/precision_landing/global_car_vel', self.global_vel_callback, 10)

        # Timers
        self.timer = self.create_timer(0.05, self.timer_callback) # 20Hz
        
        self.offboard_setpoint_counter = 0
        self.armed = False
        
        self.get_logger().info("Mission Orchestrator initialized")

    def global_vel_callback(self, msg):
        self.get_logger().info(f"Updated global car velocity to {msg.data} m/s")
        self.global_car_vel = msg.data

    def odom_callback(self, msg):
        self.drone_pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
        self.drone_vel = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]])
        
        w, x, y, z = msg.q
        self.drone_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        if self.start_pos is None and not np.isnan(msg.position[0]):
            self.start_pos = self.drone_pos.copy()
            self.target_pos = self.start_pos.copy()
            self.target_pos[2] = self.search_altitude
            self.target_yaw = self.drone_yaw

    def det_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if data.get("detected", False) and len(data.get("boxes", [])) > 0:
                # Find most confident box
                best_box = max(data["boxes"], key=lambda b: b["confidence"])
                self.car_pos = np.array([best_box["world_x"], best_box["world_y"], best_box["world_z"]])
                # self.car_vel = np.array([best_box["vel_x"], best_box["vel_y"], best_box["vel_z"]])
                self.car_vel = np.array([self.global_car_vel, 0.0, 0.0]) # Telemetry Option A
                self.car_nx = best_box.get("nx", 0.0)
                self.car_ny = best_box.get("ny", 0.0)
                self.last_det_time = self.get_clock().now()
        except Exception as e:
            self.get_logger().error(f"Error parsing /det JSON: {e}")

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Arm command sent")

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
        if self.start_pos is None:
            return

        if ((self.drone_pos[2] - self.search_altitude) < 1) or self.car_move:
            self.car_move = True
            vel_msg = Float64()
            vel_msg.data = self.global_car_vel
            self.platform_cmd_vel_pub.publish(vel_msg)

        if self.state == "HANDOVER":
            # Enable ArUco and fire handover
            enable_msg = Bool()
            enable_msg.data = True
            self.enable_aruco_pub.publish(enable_msg)
            
            msg = Bool()
            msg.data = True
            self.handover_pub.publish(msg)
            return

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()
            self.armed = True

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
        
        if self.state == "TAKEOFF":
            self.publish_trajectory_setpoint(self.target_pos[0], self.target_pos[1], self.search_altitude, self.target_yaw)
            
            if abs(self.drone_pos[2] - self.search_altitude) < 1:
                self.state = "MATCH_VELOCITY"
                self.get_logger().info("Search altitude reached. Looking for cars to match velocity...")

        elif self.state == "MATCH_VELOCITY":
            if self.last_det_time and (now - self.last_det_time).nanoseconds / 1e9 < 1.0:
                vel_error = np.linalg.norm(self.drone_vel[:2] - self.car_vel[:2])
                pos_error = np.linalg.norm(self.drone_pos[:2] - self.car_pos[:2])

                # Command drone to MATCH VELOCITY ONLY at search altitude (no position correction)
                self.publish_trajectory_setpoint(np.nan, np.nan, self.search_altitude, self.target_yaw,
                                                vx=self.car_vel[0], vy=self.car_vel[1])

                # Check if velocity is matched and roughly above car
                if vel_error < 0.5:
                    self.get_logger().info("Velocity matched! Beginning visual servo descent...")
                    self.state = "SERVO_DESCEND"
            else:
                self.publish_trajectory_setpoint(self.drone_pos[0], self.drone_pos[1], self.search_altitude, self.target_yaw)

        elif self.state == "SERVO_DESCEND":
            if self.last_det_time and (now - self.last_det_time).nanoseconds / 1e9 < 1.0:
                # Descend gradually
                if self.current_target_altitude < self.tracking_altitude:
                    self.current_target_altitude += 0.05
                
                # Visual Servoing using 3D projected position (tilt-compensated)
                kp = 1.0 # Position gain
                
                # P-controller on NED position error
                err_x = self.car_pos[0] - self.drone_pos[0]
                err_y = self.car_pos[1] - self.drone_pos[1]
                
                # Deadband to prevent wobbling: don't correct if error is small (< 0.3m)
                if abs(err_x) < 0.15: err_x = 0.0
                if abs(err_y) < 0.15: err_y = 0.0
                
                v_n = kp * err_x
                v_e = kp * err_y
                
                # Limit the correction velocity (the part that causes tilt)
                # This prevents huge initial acceleration if the car is far away.
                max_correction = 2.0 # m/s
                corr_speed = np.linalg.norm([v_n, v_e])
                if corr_speed > max_correction:
                    v_n = (v_n / corr_speed) * max_correction
                    v_e = (v_e / corr_speed) * max_correction

                # Total commanded velocity (match car vel + limited position correction)
                cmd_vx = self.car_vel[0] + v_n
                cmd_vy = self.car_vel[1] + v_e
                
                # Limit total velocity if needed, but allow it to match car speed
                max_total_vel = self.global_car_vel + 0.5 # Give it slightly more to catch up
                speed = np.linalg.norm([cmd_vx, cmd_vy])
                if speed > max_total_vel:
                    cmd_vx = (cmd_vx / speed) * max_total_vel
                    cmd_vy = (cmd_vy / speed) * max_total_vel

                # Send velocity with altitude hold/descent
                print("cmd_vx", cmd_vx)
                print("cmd_vy", cmd_vy)
                self.publish_trajectory_setpoint(np.nan, np.nan, self.current_target_altitude, self.target_yaw,
                                                vx=cmd_vx, vy=cmd_vy)

                # Check if we are at tracking altitude and well-centered (using 3D distance, immune to tilt)
                xy_error = np.linalg.norm(self.drone_pos[:2] - self.car_pos[:2])
                is_centered = (xy_error < 1)
                at_altitude = abs(self.drone_pos[2] - self.tracking_altitude) < 1
                
                if is_centered and at_altitude:
                    self.get_logger().info("Visual servoing complete. Handing over to ArUco Lander!")
                    self.state = "HANDOVER"
            else:
                self.get_logger().warn("Car lost during servo descent! Ascending to match velocity.")
                self.state = "MATCH_VELOCITY"

    def publish_trajectory_setpoint(self, x, y, z, yaw, vx=np.nan, vy=np.nan, vz=np.nan):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [float(vx), float(vy), float(vz)]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MissionOrchestrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
