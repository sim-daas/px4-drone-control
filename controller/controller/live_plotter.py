import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry
from geometry_msgs.msg import Point
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import time
import numpy as np

class LivePlotter(Node):
    def __init__(self):
        super().__init__('live_plotter')

        # Data buffers
        self.window_size_sec = 30.0
        self.times = deque()
        self.pos_n = deque()
        self.pos_e = deque()
        self.pos_d = deque()
        self.target_n = deque()
        self.target_e = deque()
        self.target_d = deque()

        self.current_target = [0.0, 0.0, 0.0]
        self.start_time = time.time()
        self.lock = threading.Lock()

        # QoS for PX4
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriptions
        self.odom_sub = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos)
        
        self.target_sub = self.create_subscription(
            Point, '/drone/target_position', self.target_callback, 10)

        self.get_logger().info("Live Plotter Node Started")

    def odom_callback(self, msg: VehicleOdometry):
        now = time.time() - self.start_time
        
        with self.lock:
            self.times.append(now)
            # msg.position is [N, E, D]
            self.pos_n.append(msg.position[0])
            self.pos_e.append(msg.position[1])
            self.pos_d.append(msg.position[2])
            
            self.target_n.append(self.current_target[0])
            self.target_e.append(self.current_target[1])
            self.target_d.append(self.current_target[2])

            # Maintain 30s window
            while self.times and (now - self.times[0] > self.window_size_sec):
                self.times.popleft()
                self.pos_n.popleft()
                self.pos_e.popleft()
                self.pos_d.popleft()
                self.target_n.popleft()
                self.target_e.popleft()
                self.target_d.popleft()

    def target_callback(self, msg: Point):
        with self.lock:
            self.current_target = [msg.x, msg.y, msg.z]

def main(args=None):
    rclpy.init(args=args)
    plotter = LivePlotter()

    # Spin ROS in a background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(plotter,), daemon=True)
    spin_thread.start()

    # Matplotlib setup
    fig, (ax_n, ax_e, ax_d) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    fig.suptitle('Drone Position Tracking (NED)', fontsize=16)

    ln_n_act, = ax_n.plot([], [], 'b-', label='Actual North')
    ln_n_tgt, = ax_n.plot([], [], 'r--', label='Target North')
    ax_n.set_ylabel('North (m)')
    ax_n.legend(loc='upper right')
    ax_n.grid(True)

    ln_e_act, = ax_e.plot([], [], 'b-', label='Actual East')
    ln_e_tgt, = ax_e.plot([], [], 'r--', label='Target East')
    ax_e.set_ylabel('East (m)')
    ax_e.legend(loc='upper right')
    ax_e.grid(True)

    ln_d_act, = ax_d.plot([], [], 'b-', label='Actual Down')
    ln_d_tgt, = ax_d.plot([], [], 'r--', label='Target Down')
    ax_d.set_ylabel('Down (m)')
    ax_d.set_xlabel('Time (s)')
    ax_d.legend(loc='upper right')
    ax_d.grid(True)

    def init():
        ax_n.set_xlim(0, 30)
        ax_e.set_xlim(0, 30)
        ax_d.set_xlim(0, 30)
        return ln_n_act, ln_n_tgt, ln_e_act, ln_e_tgt, ln_d_act, ln_d_tgt

    def update(frame):
        with plotter.lock:
            if not plotter.times:
                return ln_n_act, ln_n_tgt, ln_e_act, ln_e_tgt, ln_d_act, ln_d_tgt
            
            t = list(plotter.times)
            pn = list(plotter.pos_n)
            pe = list(plotter.pos_e)
            pd = list(plotter.pos_d)
            tn = list(plotter.target_n)
            te = list(plotter.target_e)
            td = list(plotter.target_d)

        ln_n_act.set_data(t, pn)
        ln_n_tgt.set_data(t, tn)
        ln_e_act.set_data(t, pe)
        ln_e_tgt.set_data(t, te)
        ln_d_act.set_data(t, pd)
        ln_d_tgt.set_data(t, td)

        # Update x-axis limits to slide
        if t:
            current_time = t[-1]
            start_time = max(0, current_time - 30)
            ax_n.set_xlim(start_time, max(30, current_time))
            
            # Auto-scale y-axis with some margin
            for ax, data_act, data_tgt in zip([ax_n, ax_e, ax_d], [pn, pe, pd], [tn, te, td]):
                all_data = data_act + data_tgt
                if all_data:
                    ymin, ymax = min(all_data), max(all_data)
                    margin = max(1.0, (ymax - ymin) * 0.1)
                    ax.set_ylim(ymin - margin, ymax + margin)

        return ln_n_act, ln_n_tgt, ln_e_act, ln_e_tgt, ln_d_act, ln_d_tgt

    ani = FuncAnimation(fig, update, init_func=init, blit=False, interval=100)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Cleanup
    plotter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
