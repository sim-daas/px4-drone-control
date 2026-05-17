"""
drone_controller.py — Full 4-loop cascaded controller for PX4 SITL (direct_actuator mode)

Execution order per odometry message (event-driven):
  Position (NED) → Velocity (NED) → Attitude (Euler) → Rate (FRD) → Mixer → Publish

Frame conventions:
  World: NED  — North (+X), East (+Y), Down (+Z)
  Body:  FRD  — Forward (+X), Right (+Y), Down (+Z)
  Quaternion: PX4 convention [w, x, y, z]
  Gravity:    +9.81 m/s² in NED +Z (downward)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, ActuatorMotors, VehicleOdometry
from geometry_msgs.msg import Point
import numpy as np


# ============================================================================
# Utility functions
# ============================================================================

def quat_to_rot(q):
    """Convert PX4 quaternion [w, x, y, z] → Body-to-NED rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def rot_to_euler(R):
    """
    Extract Euler angles from a Body-to-NED rotation matrix.
    Returns (roll φ, pitch θ, yaw ψ) in radians.
    """
    pitch = np.arcsin(-R[2, 0])
    roll  = np.arctan2(R[2, 1], R[2, 2])
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


# ============================================================================
# PID Controller
# ============================================================================

class PID:
    def __init__(self, kp, ki, kd, max_integral=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral  = 0.0
        self.prev_error = 0.0
        self.max_integral = max_integral

    def update(self, error, dt):
        if dt <= 0.0:
            return 0.0
        self.integral += error * dt
        self.integral  = np.clip(self.integral, -self.max_integral, self.max_integral)
        derivative     = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0


# ============================================================================
# DroneController — 4-loop cascaded architecture
# ============================================================================

class DroneController(Node):
    def __init__(self):
        super().__init__('drone_controller')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.command_pub  = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos)
        self.motor_pub    = self.create_publisher(
            ActuatorMotors, '/fmu/in/actuator_motors', qos)

        # Subscriber — event-driven control loop
        self.odom_sub = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self.odometry_callback, qos)

        # Subscriber — external position setpoints from operator input script
        # Topic: /drone/target_position  |  Type: geometry_msgs/Point
        # Field mapping (NED): x = North [m], y = East [m], z = Down [m]
        self.setpoint_sub = self.create_subscription(
            Point, '/drone/target_position',
            self.setpoint_callback, 10)

        # 250 Hz heartbeat (keepalive only — no control math here)
        self.heartbeat_timer = self.create_timer(1.0 / 250.0, self.heartbeat_callback)

        # ------------------------------------------------------------------
        # Physical constants
        # ------------------------------------------------------------------
        self.g = 9.81   # m/s², gravity magnitude (positive in NED +Z = Down)

        # ------------------------------------------------------------------
        # State (all updated fresh each odometry message)
        # ------------------------------------------------------------------
        self.current_pos_ned  = np.zeros(3)   # [N, E, D]  metres
        self.current_vel_ned  = np.zeros(3)   # [vN, vE, vD]  m/s
        self.current_euler    = np.zeros(3)   # [roll φ, pitch θ, yaw ψ]  rad
        self.current_rates    = np.zeros(3)   # [p, q, r]  rad/s  (FRD body)

        # ------------------------------------------------------------------
        # Setpoints
        # ------------------------------------------------------------------
        # Safe hover altitude above spawn point [m, NED Down axis].
        # The actual hover position is latched in the first odometry callback
        # once the real spawn coordinates are known.
        self.safe_hover_alt_offset = -5.0   # 5 m above spawn (NED: negative = up)

        # Initial value — overwritten in first odometry callback to
        # [spawn_N, spawn_E, spawn_D + safe_hover_alt_offset].
        self.target_pos_ned  = np.array([0.0, 0.0, -5.0])
        self.origin_pos_ned  = np.zeros(3)   # latched spawn position

        # Yaw target [rad]. Latched to initial heading at startup.
        self.target_yaw = 0.0

        # ------------------------------------------------------------------
        # Limits
        # ------------------------------------------------------------------
        # Maximum velocity the position controller is allowed to command [m/s]
        self.max_vel_horiz = 1.5   # North / East
        self.max_vel_down  = 1.5   # Down  (ascent: magnitude)
        self.max_vel_up    = 1.5   # Up    (descent: magnitude — NED -Z)

        # Maximum horizontal acceleration fed into arcsin [m/s²]
        # 5 m/s² → arcsin(5/9.81) ≈ 30.7° max tilt
        self.max_horiz_accel = 5.0

        # Maximum body-rate commands from attitude loop [rad/s]
        self.max_body_rate = np.radians(120)

        # Hover thrust (normalised [0,1]).  Tuned value carried from velocity phase.
        self.hover_thrust = 0.71

        # ------------------------------------------------------------------
        # Timing
        # ------------------------------------------------------------------
        self.last_odom_ns = None
        self.start_time_s = None
        self.odom_count   = 0
        self.origin_set   = False
        self.armed        = False

        # ------------------------------------------------------------------
        # Mixing matrix — FRD body frame (corrected signs, tuned phase)
        # Column order: [tau_roll, tau_pitch, tau_yaw, Thrust]
        # ------------------------------------------------------------------
        self.mixer = np.array([
            [-0.43773276,  0.70710677,  0.909091, 1.0],   # Motor 1 Front-Right
            [ 0.43773273, -0.70710677,  1.0,      1.0],   # Motor 2 Rear-Left
            [ 0.43773276,  0.70710677, -0.909091, 1.0],   # Motor 3 Front-Left
            [-0.43773273, -0.70710677, -1.0,      1.0]    # Motor 4 Rear-Right
        ])

        # ==================================================================
        # LOOP 0 — Position  (NED pos error → NED velocity setpoint)
        # Simple proportional — integration of vel error is handled by Loop 1
        # ==================================================================
        self.pid_pN = PID(kp=2, ki=0.0, kd=0.07)
        self.pid_pE = PID(kp=2, ki=0.0, kd=0.07)
        self.pid_pD = PID(kp=2.8, ki=0.07, kd=0.0)

        # ==================================================================
        # LOOP 1 — Velocity  (NED vel error → NED acceleration command)
        # Tuned in velocity phase.
        # ==================================================================
        self.pid_vN = PID(kp=2.5, ki=0.3, kd=0.03, max_integral=2.0)
        self.pid_vE = PID(kp=2.5, ki=0.3, kd=0.03, max_integral=2.0)
        self.pid_vD = PID(kp=3.4, ki=0.0, kd=0.7,  max_integral=2.0)

        # ==================================================================
        # LOOP 2 — Attitude  (Euler angle error → body rate setpoint)
        # Tuned in attitude phase.
        # ==================================================================
        self.pid_roll_att  = PID(kp=5.0, ki=0.0, kd=0.0)
        self.pid_pitch_att = PID(kp=5.0, ki=0.0, kd=0.0)
        self.pid_yaw_att   = PID(kp=2.7, ki=0.0, kd=0.0)

        # ==================================================================
        # LOOP 3 — Rate  (body rate error → motor torque command)
        # Tuned in rate phase.
        # ==================================================================
        self.pid_p = PID(kp=0.23, ki=0.5, kd=0.001)
        self.pid_q = PID(kp=0.23, ki=0.5, kd=0.001)
        self.pid_r = PID(kp=0.95, ki=0.1, kd=0.00)

        self.get_logger().info(
            f"DroneController initialised. "
            f"Target waypoint: N={self.target_pos_ned[0]:.1f} "
            f"E={self.target_pos_ned[1]:.1f} "
            f"D={self.target_pos_ned[2]:.1f} m (NED)"
        )

    # =========================================================================
    # Main control callback — event-driven from VehicleOdometry
    # =========================================================================
    def odometry_callback(self, msg):
        now_ns = self.get_clock().now().nanoseconds

        # ---- First message: latch origin, set safe hover target ----------
        if not self.origin_set:
            if np.isnan(msg.position[0]):
                return
            R_init = quat_to_rot(msg.q)
            _, _, initial_yaw = rot_to_euler(R_init)
            self.target_yaw      = initial_yaw
            self.last_odom_ns    = now_ns
            self.start_time_s    = now_ns * 1e-9
            self.origin_set      = True

            # Latch spawn position and set initial hover target directly above it
            self.origin_pos_ned  = np.array([msg.position[0],
                                             msg.position[1],
                                             msg.position[2]])
            self.target_pos_ned  = np.array([self.origin_pos_ned[0],
                                             self.origin_pos_ned[1],
                                             self.origin_pos_ned[2] + self.safe_hover_alt_offset])
            self.get_logger().info(
                f"Odometry locked. "
                f"Spawn NED=[{self.origin_pos_ned[0]:.2f},{self.origin_pos_ned[1]:.2f},{self.origin_pos_ned[2]:.2f}] m | "
                f"Safe hover target NED=[{self.target_pos_ned[0]:.2f},{self.target_pos_ned[1]:.2f},{self.target_pos_ned[2]:.2f}] m | "
                f"Initial yaw={np.degrees(initial_yaw):.1f} deg"
            )
            return  # no valid dt on the first message

        # ---- Compute true dt from ROS clock ------------------------------
        dt = (now_ns - self.last_odom_ns) * 1e-9
        self.last_odom_ns = now_ns
        if dt <= 0.0 or dt > 0.5:
            return   # degenerate dt — skip

        elapsed = now_ns * 1e-9 - self.start_time_s

        # ==================================================================
        # READ STATE
        # ==================================================================
        # Position in NED world frame (pose_frame=1 → LOCAL_FRAME_NED)
        self.current_pos_ned[0] = msg.position[0]   # North [m]
        self.current_pos_ned[1] = msg.position[1]   # East  [m]
        self.current_pos_ned[2] = msg.position[2]   # Down  [m]

        # Velocity in NED world frame (velocity_frame=1 → LOCAL_FRAME_NED)
        self.current_vel_ned[0] = msg.velocity[0]   # vN [m/s]
        self.current_vel_ned[1] = msg.velocity[1]   # vE [m/s]
        self.current_vel_ned[2] = msg.velocity[2]   # vD [m/s]

        # Attitude
        R = quat_to_rot(msg.q)
        self.current_euler = np.array(rot_to_euler(R))
        current_yaw = self.current_euler[2]

        # Body rates [p, q, r] — FRD body frame
        self.current_rates[0] = msg.angular_velocity[0]   # p (roll)
        self.current_rates[1] = msg.angular_velocity[1]   # q (pitch)
        self.current_rates[2] = msg.angular_velocity[2]   # r (yaw)

        # ==================================================================
        # LOOP 0: POSITION → VELOCITY SETPOINT (NED world frame)
        #
        # A proportional position loop generates a desired velocity in each
        # NED axis. The output is clamped to safe speed limits so the
        # velocity loop underneath is never overwhelmed.
        #
        # Sign convention:
        #   pos_err_N > 0  →  drone is south of target  →  command +vN (fly North)
        #   pos_err_D > 0  →  drone is below target      →  command +vD... wait:
        #
        # NED Down: target D = -5 m (5 m up), current D = 0 m (ground).
        # pos_err_D = target_D - current_D = -5 - 0 = -5  (negative)
        # P controller: vel_cmd_D = Kp * (-5) = negative → vD < 0 → ascend ✓
        # ==================================================================
        pos_err_N = self.target_pos_ned[0] - self.current_pos_ned[0]
        pos_err_E = self.target_pos_ned[1] - self.current_pos_ned[1]
        pos_err_D = self.target_pos_ned[2] - self.current_pos_ned[2]

        vel_cmd_N = self.pid_pN.update(pos_err_N, dt)
        vel_cmd_E = self.pid_pE.update(pos_err_E, dt)
        vel_cmd_D = self.pid_pD.update(pos_err_D, dt)

        # Clamp velocity commands to safe limits
        vel_cmd_N = np.clip(vel_cmd_N, -self.max_vel_horiz, self.max_vel_horiz)
        vel_cmd_E = np.clip(vel_cmd_E, -self.max_vel_horiz, self.max_vel_horiz)
        # vD: negative = ascent, positive = descent
        vel_cmd_D = np.clip(vel_cmd_D, -self.max_vel_up, self.max_vel_down)

        # ==================================================================
        # LOOP 1: VELOCITY → ACCELERATION (NED world frame)
        # ==================================================================
        vel_err_N = vel_cmd_N - self.current_vel_ned[0]
        vel_err_E = vel_cmd_E - self.current_vel_ned[1]
        vel_err_D = vel_cmd_D - self.current_vel_ned[2]

        a_N = self.pid_vN.update(vel_err_N, dt)
        a_E = self.pid_vE.update(vel_err_E, dt)
        a_D = self.pid_vD.update(vel_err_D, dt)

        # ==================================================================
        # STEP: THRUST from vertical acceleration + gravity compensation
        #
        # F_Z = a_D - g
        # T_norm = hover_thrust * (-F_Z / g)
        #
        # Derivation check at hover (a_D = 0):
        #   F_Z = 0 - 9.81 = -9.81
        #   T_norm = hover_thrust * (9.81 / 9.81) = hover_thrust ✓
        # Ascending (a_D < 0, e.g. -2 m/s²):
        #   F_Z = -2 - 9.81 = -11.81  →  T_norm > hover_thrust ✓
        # Descending (a_D > 0, e.g. +2 m/s²):
        #   F_Z = 2 - 9.81 = -7.81   →  T_norm < hover_thrust ✓
        # ==================================================================
        F_Z    = a_D - self.g
        thrust = self.hover_thrust * (-F_Z / self.g)
        thrust = np.clip(thrust, 0.0, 1.0)

        # ==================================================================
        # STEP: WORLD→HEADING ROTATION (NED → heading-relative body frame)
        #
        # Translates [a_N, a_E] into [a_forward, a_right] using the drone's
        # current yaw ψ so that velocity commands are decoupled from heading.
        #
        # R_yaw = [[ cos(ψ), sin(ψ)],
        #           [-sin(ψ), cos(ψ)]]
        #
        # Derivation: if ψ = 0 (facing North), a_fwd = a_N, a_rgt = a_E ✓
        #             if ψ = 90° (facing East),  a_fwd = a_E, a_rgt = -a_N ✓
        # ==================================================================
        cos_yaw   = np.cos(current_yaw)
        sin_yaw   = np.sin(current_yaw)
        a_forward = a_N * cos_yaw + a_E * sin_yaw
        a_right   = -a_N * sin_yaw + a_E * cos_yaw

        # ==================================================================
        # STEP: ACCELERATION → ATTITUDE SETPOINT (arcsin mapping)
        #
        # Pitch convention (NED/FRD):
        #   To accelerate forward (+a_forward), pitch nose down (θ < 0 in FRD).
        #   θ_d = arcsin(-a_forward / g)
        #
        # Roll convention (FRD):
        #   To accelerate right (+a_right), roll right (φ > 0 in FRD).
        #   φ_d = arcsin(a_right / g)
        #
        # Clip before arcsin: input must be ∈ (-g, g) to avoid domain error.
        # ==================================================================
        a_fwd_c = np.clip(a_forward, -self.max_horiz_accel, self.max_horiz_accel)
        a_rgt_c = np.clip(a_right,   -self.max_horiz_accel, self.max_horiz_accel)

        target_roll  = np.arcsin( a_rgt_c / self.g)   # φ_d
        target_pitch = np.arcsin(-a_fwd_c / self.g)   # θ_d

        # Yaw is held at the latched startup heading — not driven by velocity
        target_euler = np.array([target_roll, target_pitch, self.target_yaw])

        # ==================================================================
        # LOOP 2: ATTITUDE → RATE
        # ==================================================================
        error_att    = target_euler - self.current_euler
        # Yaw: wrap to shortest path in (−π, π]
        error_att[2] = (error_att[2] + np.pi) % (2 * np.pi) - np.pi

        desired_p = np.clip(self.pid_roll_att.update( error_att[0], dt),
                            -self.max_body_rate, self.max_body_rate)
        desired_q = np.clip(self.pid_pitch_att.update(error_att[1], dt),
                            -self.max_body_rate, self.max_body_rate)
        desired_r = np.clip(self.pid_yaw_att.update(  error_att[2], dt),
                            -self.max_body_rate, self.max_body_rate)

        # ==================================================================
        # LOOP 3: RATE → TORQUE
        # ==================================================================
        tau_x = self.pid_p.update(desired_p - self.current_rates[0], dt)
        tau_y = self.pid_q.update(desired_q - self.current_rates[1], dt)
        tau_z = self.pid_r.update(desired_r - self.current_rates[2], dt)

        # ==================================================================
        # CONTROL ALLOCATION — mixer maps [τ_roll, τ_pitch, τ_yaw, T] → u
        # ==================================================================
        wrench = np.array([tau_x, tau_y, tau_z, thrust])
        u = np.clip(self.mixer @ wrench, 0.0, 1.0)

        # ==================================================================
        # PUBLISH
        # ==================================================================
        m = ActuatorMotors()
        m.timestamp        = int(now_ns / 1000)
        m.timestamp_sample = m.timestamp
        m.control          = [float('nan')] * 12
        m.control[0]       = float(u[0])
        m.control[1]       = float(u[1])
        m.control[2]       = float(u[2])
        m.control[3]       = float(u[3])
        self.motor_pub.publish(m)

        # Arm after 1 s of heartbeat streaming
        if not self.armed and elapsed >= 1.0:
            self.engage_offboard_mode()
            self.arm()
            self.armed = True

        # ==================================================================
        # DEBUG LOGGING (~every 0.25 s @ 50 Hz odom)
        # ==================================================================
        self.odom_count += 1
        if self.odom_count % 12 == 0:
            pos_err_mag = np.linalg.norm([pos_err_N, pos_err_E, pos_err_D])
            print(
                f"t={elapsed:.2f}s | "
                f"pos=[{self.current_pos_ned[0]:.2f},{self.current_pos_ned[1]:.2f},{self.current_pos_ned[2]:.2f}] m | "
                # f"tgt=[{self.target_pos_ned[0]:.2f},{self.target_pos_ned[1]:.2f},{self.target_pos_ned[2]:.2f}] m | "
                # f"err_mag={pos_err_mag:.2f} m | "
                f"vel_cmd=[{vel_cmd_N:.2f},{vel_cmd_E:.2f},{vel_cmd_D:.2f}] m/s | "
                f"vel=[{self.current_vel_ned[0]:.2f},{self.current_vel_ned[1]:.2f},{self.current_vel_ned[2]:.2f}] m/s | "
                # f"att_tgt=[{np.degrees(target_roll):.1f},{np.degrees(target_pitch):.1f}] deg | "
                f"thrust={thrust:.3f}"
            )

    # =========================================================================
    # Setpoint callback — receives operator-commanded NED position
    # =========================================================================
    def setpoint_callback(self, msg: Point):
        """
        Receives a new NED position target from the operator input script.
        msg.x = North [m], msg.y = East [m], msg.z = Down [m]
        """
        new_target = np.array([msg.x, msg.y, msg.z])
        self.target_pos_ned = new_target
        self.get_logger().info(
            f"New setpoint received: N={msg.x:.2f} E={msg.y:.2f} D={msg.z:.2f} m (NED)"
        )

    # =========================================================================
    # Helpers
    # =========================================================================
    def heartbeat_callback(self):
        """250 Hz offboard keepalive — publishes OffboardControlMode only."""
        msg = OffboardControlMode()
        msg.timestamp      = int(self.get_clock().now().nanoseconds / 1000)
        msg.direct_actuator = True
        self.offboard_pub.publish(msg)

    def arm(self):
        self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Arm command sent")

    def engage_offboard_mode(self):
        self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.get_logger().info("Offboard mode command sent")

    def _vehicle_cmd(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp        = int(self.get_clock().now().nanoseconds / 1000)
        msg.param1           = param1
        msg.param2           = param2
        msg.command          = command
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        self.command_pub.publish(msg)


# ============================================================================
# Entry point
# ============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = DroneController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
