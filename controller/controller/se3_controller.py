import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, ActuatorMotors, VehicleOdometry
import numpy as np
import math

def vee(S):
    """Extracts vector from skew-symmetric matrix."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

def quat_to_rot(q):
    """Converts quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

class SE3Controller(Node):
    def __init__(self):
        super().__init__('se3_controller')

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

        # High frequency loop (250Hz)
        self.dt = 1.0 / 250.0
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        self.setpoint_counter = 0

        # Physical Constants
        self.m = 2.064
        self.g = 9.81
        self.J = np.diag([0.08612, 0.08962, 0.16088])
        
        self.arm_length = 0.25
        self.k_f = 8.54858e-06
        self.k_m = 0.016
        self.omega_min = 150.0
        self.omega_max = 1000.0

        # Gains
        # k_v_critical = 2*sqrt(k_x*m) for critical damping.
        # k_x=2.0, m=2.064 → k_v_critical≈4.1  (use 4.5 for slight overdamping)
        self.k_x = np.diag([2.0, 2.0, 2.0])
        self.k_v = np.diag([4.5, 4.5, 3.5])
        self.k_R = np.diag([3.5, 3.5, 0.5])
        self.k_Omega = np.diag([0.5, 0.5, 0.2])

        # Control Allocation Matrix
        kS = math.sin(math.radians(45))
        self.A = np.array([
            [-kS,  kS,  kS, -kS],
            [-kS,  kS, -kS,  kS],
            [ 1.0,  1.0, -1.0, -1.0],
            [ 1.0,  1.0, 1.0, 1.0]
        ])
        k_diag = np.diag([
            self.k_f * self.arm_length,
            self.k_f * self.arm_length,
            self.k_m * self.k_f,
            self.k_f
        ])
        self.A = k_diag @ self.A
        self.A_inv = np.linalg.pinv(self.A)

        # State
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.R = np.eye(3)
        self.omega = np.zeros(3)
        self.R_init = np.eye(3)  # Initial attitude, used to zero out startup bias
        
        # References
        self.origin_set = False
        self.origin = np.zeros(3)
        self.pos_d = np.zeros(3)
        self.vel_d = np.zeros(3)
        self.acc_d = np.zeros(3)
        self.yaw_d = 0.0
        self.target_alt = -2.0  # Target altitude in NED (2m up)
        self.ramp_duration = 4.0  # Seconds to ramp position setpoint from ground to target

    def odometry_callback(self, msg):
        self.pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
        self.vel = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]])
        
        # msg.q is [w, x, y, z] in PX4 VehicleOdometry (usually)
        self.R = quat_to_rot(msg.q)
        
        self.omega = np.array([msg.angular_velocity[0], msg.angular_velocity[1], msg.angular_velocity[2]])

        if not self.origin_set and not np.isnan(self.pos[0]):
            self.origin = self.pos.copy()
            self.pos_d = self.origin + np.array([0.0, 0.0, -2.0])
            # Store the initial rotation matrix to zero out any startup bias
            # (handles quaternion convention quirks, non-level spawns, etc.)
            self.R_init = self.R.copy()
            # Initialize yaw_d to the drone's actual current yaw
            self.yaw_d = math.atan2(self.R[1, 0], self.R[0, 0])
            self.get_logger().info(
                f"Origin set. yaw_d={math.degrees(self.yaw_d):.1f}deg | "
                f"q={msg.q[0]:.3f},{msg.q[1]:.3f},{msg.q[2]:.3f},{msg.q[3]:.3f} | "
                f"R_init diag=[{self.R[0,0]:.3f},{self.R[1,1]:.3f},{self.R[2,2]:.3f}]"
            )
            self.origin_set = True

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
            return # Wait for odometry

        time_sec = self.setpoint_counter * self.dt
        armed_time = time_sec - 1.0  # Time since arm (arm happens at t=1s)

        # Ramp position setpoint from ground to 2m over ramp_duration seconds
        # to prevent high initial velocity that destabilises the attitude control
        if armed_time <= 0.0:
            ramp_z = self.origin[2]  # Stay at ground before arming
        elif armed_time < self.ramp_duration:
            alpha = armed_time / self.ramp_duration
            ramp_z = self.origin[2] + alpha * self.target_alt  # Smooth linear ramp
        else:
            ramp_z = self.origin[2] + self.target_alt

        # Step setpoint at 10s after arming
        if armed_time > 10.0:
            self.pos_d = self.origin + np.array([1.0, 0.0, self.target_alt])
        else:
            self.pos_d = np.array([self.origin[0], self.origin[1], ramp_z])

        # 1. Translational Control Law
        e_x = self.pos - self.pos_d
        e_v = self.vel - self.vel_d
        
        g_vec = np.array([0.0, 0.0, self.g]) # Gravity acts in +Z in NED
        
        F_d = self.k_x @ e_x + self.k_v @ e_v + self.m * g_vec + self.m * self.acc_d

        # 2. Thrust Extraction
        e3 = np.array([0.0, 0.0, 1.0])
        f = np.dot(F_d, self.R @ e3)
        # Clamp thrust: floor = 20% hover weight to keep motors spinning,
        # ceiling = 130% weight to prevent rocket overshoot / negative-F_d scenarios
        f_hover = self.m * self.g
        f = np.clip(f, 0.2 * f_hover, 1.3 * f_hover)

        norm_F_d = np.linalg.norm(F_d)
        if norm_F_d > 1e-4:
            b3c = F_d / norm_F_d
        else:
            b3c = e3

        b1d = np.array([math.cos(self.yaw_d), math.sin(self.yaw_d), 0.0])
        
        b3c_cross_b1d = np.cross(b3c, b1d)
        norm_cross = np.linalg.norm(b3c_cross_b1d)
        if norm_cross > 1e-4:
            b2c = b3c_cross_b1d / norm_cross
        else:
            b2c = np.array([0.0, 1.0, 0.0])
            
        b1c = np.cross(b2c, b3c)
        
        R_d = np.column_stack((b1c, b2c, b3c))

        # 3. Rotational Error Computation (relative to initial frame to zero startup bias)
        R_rel   = self.R_init.T @ self.R
        R_d_rel = self.R_init.T @ R_d
        e_R = 0.5 * vee(R_d_rel.T @ R_rel - R_rel.T @ R_d_rel)

        # Clamp attitude error to keep torques in a reasonable range.
        # Large e_R (>~0.3 rad) saturates motors, starving thrust.
        e_R = np.clip(e_R, -0.35, 0.35)

        # Assuming omega_d = 0 for setpoint tracking
        omega_d = np.zeros(3)
        e_Omega = self.omega - R_rel.T @ R_d_rel @ omega_d
        
        # 4. Rotational Control Law
        # Gyroscopic compensation: omega x (J * omega)
        gyro = np.cross(self.omega, self.J @ self.omega)
        M = -self.k_R @ e_R - self.k_Omega @ e_Omega + gyro

        # 5. Control Allocation
        # Wrench format matches mixing matrix: [tau_x, tau_y, tau_z, f]
        wrench = np.array([M[0], M[1], M[2], f])
        
        omega_squared = self.A_inv @ wrench
        
        # Ensure we don't sqrt negative numbers
        omega_squared = np.maximum(omega_squared, 0.0)
        motor_omegas = np.sqrt(omega_squared)
        
        # Map to normalized throttle [0, 1], capped at 0.90 to leave
        # headroom for attitude correction without saturating the allocator
        u = (motor_omegas - self.omega_min) / (self.omega_max - self.omega_min)
        u = np.clip(u, 0.0, 0.90)

        # Always publish heartbeat & motors
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
        if self.setpoint_counter % 125 == 0:
            self.get_logger().info(
                f"t={time_sec:.1f}s | f={f:.2f}N | u={u[0]:.2f},{u[1]:.2f},{u[2]:.2f},{u[3]:.2f} | "
                f"e_R={e_R[0]:.3f},{e_R[1]:.3f},{e_R[2]:.3f} | M={M[0]:.3f},{M[1]:.3f},{M[2]:.3f}"
            )

        # Engage and arm after 1 second of streaming
        if self.setpoint_counter == int(1.0 / self.dt):
            self.engage_offboard_mode()
            self.arm()
            
        self.setpoint_counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = SE3Controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
