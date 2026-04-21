import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, ActuatorMotors, VehicleOdometry
import numpy as np


def quat_to_rot(q):
    """Converts quaternion [w, x, y, z] to Body-to-NED rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def rot_to_euler(R):
    """Extracts Euler angles (roll, pitch, yaw) from Body-to-NED rotation matrix."""
    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


class PID:
    def __init__(self, kp, ki, kd, max_integral=1.0):
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


class VelocityController(Node):
    def __init__(self):
        super().__init__('velocity_controller')

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

        # Heartbeat timer - offboard keepalive only, no control math
        self.heartbeat_timer = self.create_timer(1.0 / 250.0, self.heartbeat_callback)

        # ------------------------------------------------------------------
        # Physical Constants
        # ------------------------------------------------------------------
        self.g = 9.81  # m/s^2, gravity in NED (+Z = Down)

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.current_rates = np.zeros(3)     # body rates [p, q, r] rad/s
        self.current_euler = np.zeros(3)     # [roll, pitch, yaw] rad
        self.current_vel_ned = np.zeros(3)   # [vN, vE, vD] m/s (NED world frame)
        self.origin_set = False
        self.armed = False

        # ------------------------------------------------------------------
        # Setpoints
        # ------------------------------------------------------------------
        # Target NED velocity [vN, vE, vD] in m/s.
        # Defaults to zero (hover). Override externally for flight commands.
        self.desired_vel_ned = np.zeros(3)

        # Yaw target (radians, NED). Set once at startup to current heading.
        self.target_yaw = 0.0

        # Thrust is computed dynamically by the velocity loop; it replaces
        # the static hover_thrust from the previous phase.
        self.hover_thrust = 0.71  # used as a feedforward reference for thrust mapping

        # Tilt limit: maximum horizontal acceleration command before arcsin (m/s²)
        # 5.0 m/s² ≈ 30.7° max tilt
        self.max_horiz_accel = 5.0

        # ------------------------------------------------------------------
        # Timing
        # ------------------------------------------------------------------
        self.last_odom_time_ns = None
        self.start_time_s = None
        self.odom_count = 0

        # ------------------------------------------------------------------
        # Mixing Matrix (FRD body frame, corrected signs)
        # Vector order: [tau_x (roll), tau_y (pitch), tau_z (yaw), Thrust]
        # ------------------------------------------------------------------
        self.mixer = np.array([
            [-0.43773276,  0.70710677,  0.909091, 1.0],   # Motor 1 Front-Right
            [ 0.43773273, -0.70710677,  1.0,      1.0],   # Motor 2 Rear-Left
            [ 0.43773276,  0.70710677, -0.909091, 1.0],   # Motor 3 Front-Left
            [-0.43773273, -0.70710677, -1.0,      1.0]    # Motor 4 Rear-Right
        ])

        # ------------------------------------------------------------------
        # LOOP 1: Velocity Controllers (NED world frame)
        # Input:  velocity error [m/s]  →  Output: desired acceleration [m/s²]
        # ------------------------------------------------------------------
        self.pid_vN = PID(kp=2.5, ki=0.3, kd=0.03, max_integral=2.0)
        self.pid_vE = PID(kp=2.5, ki=0.3, kd=0.03, max_integral=2.0)
        self.pid_vD = PID(kp=3.4, ki=0.0, kd=0.7, max_integral=2.0)

        # ------------------------------------------------------------------
        # LOOP 2: Attitude Controllers (target angle → target body rate)
        # Tuned values from previous phase.
        # ------------------------------------------------------------------
        self.pid_roll_att  = PID(kp=5.0, ki=0.0, kd=0.0)
        self.pid_pitch_att = PID(kp=5.0, ki=0.0, kd=0.0)
        self.pid_yaw_att   = PID(kp=2.7, ki=0.0, kd=0.0)

        # ------------------------------------------------------------------
        # LOOP 3: Rate Controllers (target body rate → motor torques)
        # Tuned values from previous phase.
        # ------------------------------------------------------------------
        self.pid_p = PID(kp=0.23, ki=0.5, kd=0.001)
        self.pid_q = PID(kp=0.23, ki=0.5, kd=0.001)
        self.pid_r = PID(kp=0.95, ki=0.1, kd=0.00)

    # ------------------------------------------------------------------
    # Main control callback — event-driven from odometry
    # Execution order:  Velocity → Attitude → Rate → Mixer → Publish
    # ------------------------------------------------------------------
    def odometry_callback(self, msg):
        now_ns = self.get_clock().now().nanoseconds

        # First message: initialise state and lock initial yaw
        if not self.origin_set:
            if np.isnan(msg.position[0]):
                return
            R_init = quat_to_rot(msg.q)
            _, _, initial_yaw = rot_to_euler(R_init)
            self.target_yaw = initial_yaw
            self.origin_set = True
            self.last_odom_time_ns = now_ns
            self.start_time_s = now_ns * 1e-9
            self.get_logger().info(
                f"Velocity controller active. Initial yaw: {np.degrees(initial_yaw):.1f} deg")
            return  # skip first loop — no valid dt yet

        # Compute true dt
        dt = (now_ns - self.last_odom_time_ns) * 1e-9
        self.last_odom_time_ns = now_ns
        if dt <= 0.0 or dt > 0.5:
            return

        elapsed = now_ns * 1e-9 - self.start_time_s

        # ---- READ STATE -------------------------------------------------
        # Body rates [p, q, r] — already in FRD body frame
        self.current_rates[0] = msg.angular_velocity[0]  # roll rate  p
        self.current_rates[1] = msg.angular_velocity[1]  # pitch rate q
        self.current_rates[2] = msg.angular_velocity[2]  # yaw rate   r

        # Attitude
        R = quat_to_rot(msg.q)
        self.current_euler = np.array(rot_to_euler(R))
        current_yaw = self.current_euler[2]

        # Velocity in NED world frame (velocity_frame=1 → LOCAL_FRAME_NED)
        self.current_vel_ned[0] = msg.velocity[0]  # vN (m/s, North positive)
        self.current_vel_ned[1] = msg.velocity[1]  # vE (m/s, East positive)
        self.current_vel_ned[2] = msg.velocity[2]  # vD (m/s, Down positive)

        # ================================================================
        # LOOP 1: VELOCITY → ACCELERATION (NED world frame)
        # ================================================================
        vel_err_N = self.desired_vel_ned[0] - self.current_vel_ned[0]
        vel_err_E = self.desired_vel_ned[1] - self.current_vel_ned[1]
        vel_err_D = self.desired_vel_ned[2] - self.current_vel_ned[2]

        # PID outputs desired NED accelerations [m/s²]
        a_N = self.pid_vN.update(vel_err_N, dt)  # North acceleration
        a_E = self.pid_vE.update(vel_err_E, dt)  # East acceleration
        a_D = self.pid_vD.update(vel_err_D, dt)  # Down acceleration

        # ================================================================
        # STEP 2: THRUST from vertical acceleration + gravity compensation
        #
        # In NED, gravity acts in +Z (down). To hover (a_D = 0), motors
        # must cancel gravity: F_Z = a_D - g (= -g at hover).
        # T_norm = hover_thrust * (-F_Z / g)
        #   → at hover (a_D=0):  T_norm = hover_thrust * (g/g) = hover_thrust ✓
        #   → descend (a_D>0):   F_Z less negative → T_norm decreases ✓
        #   → ascend  (a_D<0):   F_Z more negative → T_norm increases ✓
        # ================================================================
        F_Z = a_D - self.g
        thrust = self.hover_thrust * (-F_Z / self.g)
        thrust = np.clip(thrust, 0.0, 1.0)

        # ================================================================
        # STEP 3: WORLD→HEADING ROTATION (NED → heading-relative body frame)
        #
        # Rotates [a_N, a_E] into [a_Forward, a_Right] using the yaw
        # rotation matrix. This decouples the velocity commands from the
        # drone's current heading.
        #
        # R_yaw = [[cos(ψ), sin(ψ)],
        #          [-sin(ψ), cos(ψ)]]
        #
        # a_Forward = a_N*cos(ψ) + a_E*sin(ψ)
        # a_Right   = -a_N*sin(ψ) + a_E*cos(ψ)
        # ================================================================
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)

        a_forward = a_N * cos_yaw + a_E * sin_yaw
        a_right   = -a_N * sin_yaw + a_E * cos_yaw

        # ================================================================
        # STEP 4: ACCELERATION → ATTITUDE SETPOINT
        #
        # Clip before arcsin — input must be in (-1, 1)*g to be physical.
        # max_horiz_accel = 5.0 m/s² → max tilt ≈ arcsin(5/9.81) ≈ 30.7°
        #
        # Pitch convention (NED/FRD):
        #   Pitch forward (nose down) = negative pitch angle.
        #   To accelerate forward (+a_forward), pitch nose down → theta_d < 0.
        #   → theta_d = arcsin(-a_forward / g)
        #
        # Roll convention (FRD):
        #   Roll right (right wing down) = positive roll angle.
        #   To accelerate right (+a_right), roll right → phi_d > 0.
        #   → phi_d = arcsin(a_right / g)
        # ================================================================
        a_forward_clamped = np.clip(a_forward, -self.max_horiz_accel, self.max_horiz_accel)
        a_right_clamped   = np.clip(a_right,   -self.max_horiz_accel, self.max_horiz_accel)

        target_pitch = np.arcsin(-a_forward_clamped / self.g)  # θ_d
        target_roll  = np.arcsin( a_right_clamped   / self.g)  # φ_d

        # Yaw target is independent of velocity — held from startup lock
        target_euler = np.array([target_roll, target_pitch, self.target_yaw])

        # ================================================================
        # LOOP 2: ATTITUDE → RATE
        # ================================================================
        error_att = target_euler - self.current_euler
        # Yaw wrap-around to shortest path (−π, π]
        error_att[2] = (error_att[2] + np.pi) % (2 * np.pi) - np.pi

        desired_p = self.pid_roll_att.update(error_att[0], dt)
        desired_q = self.pid_pitch_att.update(error_att[1], dt)
        desired_r = self.pid_yaw_att.update(error_att[2], dt)

        # Clamp desired rates to safe limits
        rate_limit = np.radians(120)  # 120 deg/s
        desired_p = np.clip(desired_p, -rate_limit, rate_limit)
        desired_q = np.clip(desired_q, -rate_limit, rate_limit)
        desired_r = np.clip(desired_r, -rate_limit, rate_limit)

        # ================================================================
        # LOOP 3: RATE → TORQUE
        # ================================================================
        error_p = desired_p - self.current_rates[0]
        error_q = desired_q - self.current_rates[1]
        error_r = desired_r - self.current_rates[2]

        tau_x = self.pid_p.update(error_p, dt)
        tau_y = self.pid_q.update(error_q, dt)
        tau_z = self.pid_r.update(error_r, dt)

        # ================================================================
        # CONTROL ALLOCATION
        # ================================================================
        wrench = np.array([tau_x, tau_y, tau_z, thrust])
        u = self.mixer @ wrench
        u = np.clip(u, 0.0, 1.0)

        # ================================================================
        # PUBLISH
        # ================================================================
        motor_msg = ActuatorMotors()
        motor_msg.timestamp = int(now_ns / 1000)
        motor_msg.timestamp_sample = motor_msg.timestamp
        motor_msg.control = [float('nan')] * 12
        motor_msg.control[0] = float(u[0])
        motor_msg.control[1] = float(u[1])
        motor_msg.control[2] = float(u[2])
        motor_msg.control[3] = float(u[3])
        self.motor_publisher.publish(motor_msg)

        # Arm after 1 second of streaming
        if not self.armed and elapsed >= 1.0:
            self.engage_offboard_mode()
            self.arm()
            self.armed = True

        if 0 < elapsed < 5:
            self.desired_vel_ned = [0.0, 0.0, -2.0]
        elif 6 < elapsed < 12:
            self.desired_vel_ned = [1.5, 0.0, 0.0]
        elif 12 < elapsed < 20:
            self.desired_vel_ned = [1.5, 1, -1.0]
        else:
            self.desired_vel_ned = [0.0, 0.0, 0.0]
        # ================================================================
        # DEBUG LOGGING (~every 0.25s at ~50Hz odom = every 12 messages)
        # ================================================================
        self.odom_count += 1
        if self.odom_count % 12 == 0:
            print(
                f"t={elapsed:.2f}s | "
                f"vel_ned=[{self.current_vel_ned[0]:.2f},{self.current_vel_ned[1]:.2f},{self.current_vel_ned[2]:.2f}] m/s | "
                # f"des_vel=[{self.desired_vel_ned[0]:.2f},{self.desired_vel_ned[1]:.2f},{self.desired_vel_ned[2]:.2f}] m/s | "
                # f"a_fwd={a_forward:.2f} a_rgt={a_right:.2f} m/s² | "
                f"att_tgt=[{np.degrees(target_roll):.1f},{np.degrees(target_pitch):.1f},{np.degrees(self.target_yaw):.1f}] deg | "
                f"att_cur=[{np.degrees(self.current_euler[0]):.1f},{np.degrees(self.current_euler[1]):.1f},{np.degrees(current_yaw):.1f}] deg | "
                f"thrust={thrust:.3f}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def heartbeat_callback(self):
        """250Hz offboard keepalive — no control math."""
        self.publish_offboard_control_mode()

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Arm command sent")

    def engage_offboard_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.get_logger().info("Offboard mode command sent")

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
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


def main(args=None):
    rclpy.init(args=args)
    node = VelocityController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
