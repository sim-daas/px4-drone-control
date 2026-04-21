"""
position_input.py — Operator terminal input for drone position setpoints.

Publishes NED position targets to /drone/target_position (geometry_msgs/Point).
The drone_controller node subscribes to this topic and updates its position loop.

Field mapping:
  msg.x = North [m]   (positive = North)
  msg.y = East  [m]   (positive = East)
  msg.z = Down  [m]   (positive = Down, NEGATIVE = UP)

Usage:
  ros2 run controller position_input
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import sys
import threading


BANNER = """
╔══════════════════════════════════════════════════════════╗
║           Drone Position Setpoint Publisher              ║
║                                                          ║
║  Topic : /drone/target_position                          ║
║  Frame : NED  (North, East, Down)                        ║
║                                                          ║
║  Enter target position when prompted.                    ║
║  Altitude: NEGATIVE = UP  (e.g. -5 = 5 m above ground)  ║
║  Type  'q'  or Ctrl-C to quit.                           ║
╚══════════════════════════════════════════════════════════╝
"""


class PositionInput(Node):
    def __init__(self):
        super().__init__('position_input')
        self.pub = self.create_publisher(Point, '/drone/target_position', 10)
        self.get_logger().info("Position input node started.")

    def send(self, north: float, east: float, down: float):
        msg = Point()
        msg.x = float(north)
        msg.y = float(east)
        msg.z = float(down)
        self.pub.publish(msg)
        self.get_logger().info(
            f"Published → N={north:.2f}  E={east:.2f}  D={down:.2f} m (NED)"
        )


def input_loop(node: PositionInput):
    print(BANNER)

    while rclpy.ok():
        print("─" * 56)
        try:
            raw = input("  Enter  N  E  D  (space-separated, metres, NED): ").strip()
        except EOFError:
            break

        if not raw or raw.lower() == 'q':
            print("  Exiting position input. Drone will hold last setpoint.")
            break

        parts = raw.split()
        if len(parts) != 3:
            print(f"  ✗ Expected 3 values, got {len(parts)}. Try again.")
            continue

        try:
            north, east, down = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            print("  ✗ Non-numeric input. Try again.")
            continue

        # Safety check — warn if altitude is positive (below ground in NED)
        if down > 0:
            confirm = input(
                f"  ⚠  Down={down:.2f} m is BELOW ground level in NED. "
                f"Are you sure? (y/N): "
            ).strip().lower()
            if confirm != 'y':
                print("  Cancelled.")
                continue

        node.send(north, east, down)
        print(f"  ✓ Setpoint sent: N={north:.2f}  E={east:.2f}  D={down:.2f}")

    # Signal ROS to shut down so rclpy.spin() returns
    rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = PositionInput()

    # Run input loop in a background thread so rclpy.spin() stays on the main thread
    thread = threading.Thread(target=input_loop, args=(node,), daemon=True)
    thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n  Interrupted. Shutting down.")
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
