#!/usr/bin/env python3
"""Move the pickup truck in a straight line at constant velocity."""

import math
import subprocess
import time

# Simulation parameters
MODEL_NAME = "pickup_aruco"
WORLD_NAME = "default"
UPDATE_HZ = 30

# Path parameters
START_X = 528.11
START_Y = -419.37
END_X = 448.24
END_Y = -169.41
Z = -0.12  # Height above ground
SPEED = 2.0  # m/s (constant velocity)

# Calculate trajectory
DX = END_X - START_X
DY = END_Y - START_Y
DISTANCE = math.sqrt(DX**2 + DY**2)
YAW = math.atan2(DY, DX)
UX = DX / DISTANCE
UY = DY / DISTANCE

def set_pose(x, y, z, yaw):
    """Set model pose via gz service."""
    qz = math.sin(yaw / 2)
    qw = math.cos(yaw / 2)

    req = (
        f'name: "{MODEL_NAME}", '
        f'position: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}}, '
        f'orientation: {{x: 0, y: 0, z: {qz:.6f}, w: {qw:.6f}}}'
    )

    subprocess.run(
        ['gz', 'service', '-s', f'/world/{WORLD_NAME}/set_pose',
         '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
         '--timeout', '100', '--req', req],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def main():
    print(f"Moving '{MODEL_NAME}' in a straight line")
    print(f"  Speed  : {SPEED} m/s")
    print("Press Ctrl+C to stop.\n")

    dt = 1.0 / UPDATE_HZ
    t0 = time.time()

    while True:
        t = time.time() - t0
        dist_moved = SPEED * t
        
        if dist_moved >= DISTANCE:
            # Snap to end point and stop
            set_pose(END_X, END_Y, Z, YAW)
            print(f"Reached destination: ({END_X}, {END_Y})")
            break
            
        x = START_X + UX * dist_moved
        y = START_Y + UY * dist_moved
        
        set_pose(x, y, Z, YAW)
        time.sleep(dt)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
