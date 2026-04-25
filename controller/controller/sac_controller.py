import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, ActuatorMotors, VehicleOdometry
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def quat_to_rot(q):
    """Converts PX4 quaternion [w, x, y, z] to Body-to-NED rotation matrix."""
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
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


# ------------------------------------------------------------------
# Dummy Environment for VecNormalize
# ------------------------------------------------------------------
class DummyGazeboEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64)

    def step(self, action):
        return np.zeros(16), 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        return np.zeros(16), {}


# ------------------------------------------------------------------
# Main ROS 2 Node
# ------------------------------------------------------------------
class SACController(Node):
    def __init__(self):
        super().__init__('sac_controller')

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

        self.heartbeat_timer = self.create_timer(1.0 / 250.0, self.heartbeat_callback)

        # ------------------------------------------------------------------
        # Load SAC Model
        # ------------------------------------------------------------------
        pkg_share = get_package_share_directory('controller')
        self.model_path = os.path.join(pkg_share, 'models', 'final_hover_sac_model.zip')
        self.stats_path = os.path.join(pkg_share, 'models', 'final_vec_normalize_stats.pkl')
        
        dummy_env = DummyVecEnv([lambda: DummyGazeboEnv()])
        self.vec_env = VecNormalize.load(self.stats_path, dummy_env)
        self.vec_env.training = False
        self.vec_env.norm_reward = False

        self.model = SAC.load(self.model_path, env=self.vec_env, device="cpu")
        self.get_logger().info("SAC Model loaded successfully.")

        # ------------------------------------------------------------------
        # Configuration & Goals
        # ------------------------------------------------------------------
        self.goal_position_nwu = np.array([0.0, 0.0, 5.0], dtype=np.float64)
        self.state_machine = 0 
        
        self.max_roll_pitch_rate = 50.0 * np.pi / 180.0
        self.max_yaw_rate = 15.0 * np.pi / 180.0
        
        # ------------------------------------------------------------------
        # Thrust Correction Logic (PyBullet vs Gazebo)
        # PyBullet (2kg drone, 40N max thrust) -> Hover = (2*9.81)/40 = 0.49
        # Gazebo (X500 drone) -> Hover = 0.71 (from drone_controller.py)
        # Correction Factor = 0.71 / 0.49 = 1.45
        # ------------------------------------------------------------------
        self.hover_rl = 0.49
        self.hover_gazebo = 0.71
        self.thrust_scale = self.hover_gazebo / self.hover_rl
        self.get_logger().info(f"Thrust scaling factor: {self.thrust_scale:.2f}")

        # Integral error buffer
        self.error_buffer = deque(maxlen=50)

        self.g = 9.81
        self.mixer = np.array([
            [-0.43773276,  0.70710677,  0.909091, 1.0],   # Motor 1 Front-Right
            [ 0.43773273, -0.70710677,  1.0,      1.0],   # Motor 2 Rear-Left
            [ 0.43773276,  0.70710677, -0.909091, 1.0],   # Motor 3 Front-Left
            [-0.43773273, -0.70710677, -1.0,      1.0]    # Motor 4 Rear-Right
        ])
        
        # PIDs for Takeoff
        self.pid_pN = PID(kp=2, ki=0.0, kd=0.07)
        self.pid_pE = PID(kp=2, ki=0.0, kd=0.07)
        self.pid_pD = PID(kp=2.8, ki=0.07, kd=0.0)
        self.pid_vN = PID(kp=2.5, ki=0.3, kd=0.03, max_integral=2.0)
        self.pid_vE = PID(kp=2.5, ki=0.3, kd=0.03, max_integral=2.0)
        self.pid_vD = PID(kp=3.4, ki=0.0, kd=0.7,  max_integral=2.0)
        self.pid_roll_att  = PID(kp=5.0, ki=0.0, kd=0.0)
        self.pid_pitch_att = PID(kp=5.0, ki=0.0, kd=0.0)
        self.pid_yaw_att   = PID(kp=2.7, ki=0.0, kd=0.0)

        # PIDs for Rate Control
        self.pid_p = PID(kp=0.23, ki=0.5, kd=0.001)
        self.pid_q = PID(kp=0.23, ki=0.5, kd=0.001)
        self.pid_r = PID(kp=0.95, ki=0.1, kd=0.00)

        # State tracking
        self.odom_received = False
        self.armed = False
        self.start_time_s = None
        self.last_time_s = None
        self.origin_pos_ned = None
        
        self.current_pos_ned = np.zeros(3)
        self.current_vel_ned = np.zeros(3)
        self.current_pos_nwu = np.zeros(3)
        self.current_lin_vel_body_nwu = np.zeros(3)
        self.current_ang_vel_body_nwu = np.zeros(3)
        self.current_quat_nwu = np.array([0.0, 0.0, 0.0, 1.0])
        self.current_euler_frd = np.zeros(3)
        self.current_rates_frd = np.zeros(3)
        self.target_yaw_frd = 0.0

        self.control_timer = self.create_timer(1.0 / 30.0, self.control_loop)

    def odometry_callback(self, msg):
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if not self.odom_received:
            if np.isnan(msg.position[0]):
                return
            self.origin_pos_ned = np.array([msg.position[0], msg.position[1], msg.position[2]])
            R_init = quat_to_rot(msg.q)
            _, _, self.target_yaw_frd = rot_to_euler(R_init)
            self.start_time_s = now_s
            self.last_time_s = now_s
            self.odom_received = True

        self.current_pos_ned = np.array([msg.position[0], msg.position[1], msg.position[2]])
        self.current_vel_ned = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]])
        self.current_rates_frd = np.array([msg.angular_velocity[0], msg.angular_velocity[1], msg.angular_velocity[2]])
        
        R_ned = quat_to_rot(msg.q)
        self.current_euler_frd = np.array(rot_to_euler(R_ned))
        self.current_pos_nwu = np.array([msg.position[0], -msg.position[1], -msg.position[2]])
        
        # Frame conversion (FRD Body -> NWU Body)
        rot_nwu = Rotation.from_euler('xyz', [self.current_euler_frd[0], -self.current_euler_frd[1], -self.current_euler_frd[2]])
        self.current_quat_nwu = rot_nwu.as_quat() # [x, y, z, w]
        
        vel_world_nwu = np.array([msg.velocity[0], -msg.velocity[1], -msg.velocity[2]])
        self.current_lin_vel_body_nwu = rot_nwu.as_matrix().T @ vel_world_nwu
        
        self.current_ang_vel_body_nwu[0] = self.current_rates_frd[0]
        self.current_ang_vel_body_nwu[1] = -self.current_rates_frd[1]
        self.current_ang_vel_body_nwu[2] = -self.current_rates_frd[2]

    def control_loop(self):
        if not self.odom_received:
            return

        now_s = self.get_clock().now().nanoseconds * 1e-9
        dt = now_s - self.last_time_s
        self.last_time_s = now_s

        if dt <= 0.0:
            return

        elapsed = now_s - self.start_time_s
        
        if self.state_machine == 0:
            # TAKEOFF
            target_pos_ned = np.array([self.origin_pos_ned[0], self.origin_pos_ned[1], self.origin_pos_ned[2] - 5.0])
            pos_err = target_pos_ned - self.current_pos_ned
            vel_cmd = np.array([self.pid_pN.update(pos_err[0], dt), self.pid_pE.update(pos_err[1], dt), self.pid_pD.update(pos_err[2], dt)])
            vel_cmd = np.clip(vel_cmd, -1.5, 1.5)
            
            accel_cmd = np.array([self.pid_vN.update(vel_cmd[0] - self.current_vel_ned[0], dt), self.pid_vE.update(vel_cmd[1] - self.current_vel_ned[1], dt), self.pid_vD.update(vel_cmd[2] - self.current_vel_ned[2], dt)])
            
            thrust = np.clip(self.hover_gazebo * (-(accel_cmd[2] - self.g) / self.g), 0.0, 1.0)
            
            cos_y, sin_y = np.cos(self.current_euler_frd[2]), np.sin(self.current_euler_frd[2])
            a_fwd, a_rgt = accel_cmd[0] * cos_y + accel_cmd[1] * sin_y, -accel_cmd[0] * sin_y + accel_cmd[1] * sin_y
            
            desired_p = self.pid_roll_att.update(np.arcsin(np.clip(a_rgt, -5.0, 5.0)/self.g) - self.current_euler_frd[0], dt)
            desired_q = self.pid_pitch_att.update(np.arcsin(-np.clip(a_fwd, -5.0, 5.0)/self.g) - self.current_euler_frd[1], dt)
            desired_r = self.pid_yaw_att.update(((self.target_yaw_frd - self.current_euler_frd[2] + np.pi) % (2*np.pi) - np.pi), dt)
            
            if np.linalg.norm(pos_err) < 0.2 and np.linalg.norm(self.current_vel_ned) < 0.2:
                self.state_machine = 1
                self.error_buffer.clear()
                self.get_logger().info("SWITCHING TO RL...")

        else:
            # RL CONTROL
            pos_error_nwu = self.goal_position_nwu - self.current_pos_nwu
            self.error_buffer.append(pos_error_nwu)
            integral_error = np.sum(self.error_buffer, axis=0)

            raw_obs = np.concatenate([pos_error_nwu, self.current_quat_nwu, self.current_lin_vel_body_nwu, self.current_ang_vel_body_nwu, integral_error], axis=-1)
            norm_obs = self.vec_env.normalize_obs(raw_obs.reshape(1, -1))
            action, _ = self.model.predict(norm_obs, deterministic=True)
            action = action[0]

            desired_p = action[0] * self.max_roll_pitch_rate
            desired_q = -action[1] * self.max_roll_pitch_rate
            desired_r = -action[2] * self.max_yaw_rate
            
            # Apply Thrust Scaling (Correction for PyBullet vs Gazebo units)
            rl_thrust_01 = (action[3] + 1.0) / 2.0
            thrust = np.clip(rl_thrust_01 * self.thrust_scale, 0.0, 1.0)

            if np.random.rand() < 0.05:
                self.get_logger().info(f"RL: thrust_raw={rl_thrust_01:.3f}, thrust_scaled={thrust:.3f}, z_err={pos_error_nwu[2]:.2f}")

        # Common Rate Loop
        tau_x = self.pid_p.update(desired_p - self.current_rates_frd[0], dt)
        tau_y = self.pid_q.update(desired_q - self.current_rates_frd[1], dt)
        tau_z = self.pid_r.update(desired_r - self.current_rates_frd[2], dt)

        u = np.clip(self.mixer @ np.array([tau_x, tau_y, tau_z, thrust]), 0.0, 1.0)

        motor_msg = ActuatorMotors()
        motor_msg.timestamp = int(now_s * 1e6)
        motor_msg.control = [float('nan')] * 12
        for i in range(4): motor_msg.control[i] = float(u[i])
        self.motor_publisher.publish(motor_msg)

        if not self.armed and elapsed >= 1.0:
            self.engage_offboard_mode()
            self.arm()
            self.armed = True

    def heartbeat_callback(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.direct_actuator = True
        self.offboard_control_mode_publisher.publish(msg)

    def arm(self):
        self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

    def engage_offboard_mode(self):
        self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

    def _vehicle_cmd(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.param1, msg.param2, msg.command = param1, param2, command
        msg.target_system, msg.target_component = 1, 1
        msg.source_system, msg.source_component = 1, 1
        msg.from_external = True
        self.vehicle_command_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(SACController())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
