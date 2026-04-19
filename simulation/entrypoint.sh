#!/bin/bash
set -e

echo "=== Simulation Container Starting ==="

# Source ROS2
source /opt/ros/humble/setup.bash 2>/dev/null || true
source /root/ros2_ws/install/setup.bash 2>/dev/null || true
source /root/.bashrc 2>/dev/null || true

# Copy tmuxinator config
mkdir -p ~/.config/tmuxinator
cp /app/sim.yml ~/.config/tmuxinator/sim.yml

# Build ROS2 workspace if needed
echo "Building ROS2 Workspace..."
cd /root/ros2_ws
# colcon build

# # Start tmuxinator
# echo "Starting simulation with tmuxinator..."
# tmuxinator start sim

# # Keep container alive
exec tail -f /dev/null
