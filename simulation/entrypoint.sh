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

# Deploy PX4 SITL parameter overrides (land detector + commander fixes)
# This file is sourced by rcS at SITL startup via `. px4-rc.params`
PX4_RC_PARAMS_DEST=/root/PX4-Autopilot/build/px4_sitl_default/etc/init.d-posix/px4-rc.params
if [ -f "$PX4_RC_PARAMS_DEST" ]; then
    cp /app/px4-rc.params "$PX4_RC_PARAMS_DEST"
    echo "PX4 parameter overrides deployed to $PX4_RC_PARAMS_DEST"
else
    echo "WARNING: PX4 SITL build dir not found, skipping param override deploy."
    echo "  Run manually: cp /app/px4-rc.params $PX4_RC_PARAMS_DEST"
fi

# Build ROS2 workspace if needed
echo "Building ROS2 Workspace..."
cd /root/ros2_ws
# colcon build

# # Start tmuxinator
# echo "Starting simulation with tmuxinator..."
# tmuxinator start sim

# # Keep container alive
exec tail -f /dev/null
