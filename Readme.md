# Project Report: Actuator-Level Drone Controller for PX4 SITL

## Executive Summary
This report details the implementation of an actuator-level control pipeline for a quadrotor (x500) in PX4 SITL using ROS 2. By bypassing PX4's internal high-level position and attitude controllers, we established direct control over the `ActuatorMotors` topic. The project evolved from an open-loop motor test to a robust, fully cascaded control architecture (Position → Velocity → Attitude → Rate → Mixer) and also explored an advanced SE(3) geometric tracking controller.

## Demo Video
[![Final Demo](https://img.youtube.com/vi/5ZKq_dR_DrE/0.jpg)](https://youtu.be/5ZKq_dR_DrE)

## 1. Architectural Foundation & PX4 Integration

### Offboard Mode & Synchronization
A critical initial hurdle was understanding PX4's strict offboard failsafes. PX4 requires continuous, synchronous publishing (>2Hz) of both `OffboardControlMode` (the heartbeat) and the target setpoint (`ActuatorMotors`). 
*   **NaN Padding**: Sending `0.0` to unused motor indices (4 through 11) is interpreted as a command and causes unpredictable mixer behavior. We explicitly padded unused motors with `NaN`.
*   **Event-Driven Execution**: Our initial design used a fixed 250Hz timer for PID updates while odometry arrived at 50Hz. This "Discretization Trap" caused the derivative term to intermittently hit zero or spike, leading to sluggishness and crashes. The architecture was rewritten to be **event-driven**, running the entire control math inside the odometry callback with a dynamic true `dt` calculation based on message timestamps.

### Mid-Air Disarming (Land Detector)
During low-velocity maneuvers and stable hover, PX4's internal land detector would falsely trigger, disarming the drone mid-air. We solved this by injecting custom SITL parameters via a persistent `px4-rc.params` script deployed on container startup:
*   `COM_DISARM_LAND = -1` (Disables auto-disarm)
*   `LNDMC_Z_VEL_MAX = 10.0` (Raises vertical velocity threshold)

## 2. Control Allocation & Frame Mathematics

### The Mixer Matrix Bug
When adapting the C++ reference mixer, we initially calculated the pseudo-inverse of the mixing matrix. However, the provided matrix *already* mapped torques and thrust to individual motor throttles. Taking the pseudo-inverse mathematically destroyed this mapping, resulting in near-zero throttles and preventing takeoff. We corrected this by directly multiplying the torque/thrust vector by the mixer matrix.

### Sign Inversions & Positive Feedback
Initial tests revealed rapid pitch and yaw divergence (e.g., pitch error ballooning from 0.06 to -5.70 within seconds). This was caused by inverted signs in the pitch and yaw columns of the mixer matrix. A commanded pitch-up increased front motor speeds instead of decreasing them. Inverting the signs fixed the positive feedback loop.

### Euler Extraction & Initialization Bias
We discovered a frame mismatch in our state estimation: `quat_to_rot` produced a Body-to-NED matrix, but our `rot_to_euler` function used indices meant for a NED-to-Body matrix, causing inverted attitude readings. 
Furthermore, the drone spawned with an arbitrary yaw orientation. To prevent massive initial correction torques, we implemented **origin latching**: storing the initial rotation matrix (`R_init`) at startup and computing all subsequent attitude errors relative to it.

## 3. PID Tuning Learnings & Insights

Through extensive flight testing and log analysis, we gathered several critical insights into drone dynamics and PID tuning:

### 1. No Damping for Yaw
While Roll and Pitch loops required a derivative term (`Kd = 0.005`) to dampen oscillations, the **Yaw rate loop was tuned with `Kd = 0.0`**. 
*   **Why?** Yaw control via differential torque is inherently less responsive than roll/pitch (which rely on direct thrust changes). Furthermore, yaw dynamics are heavily damped by aerodynamic drag on the propellers. Adding a derivative gain to yaw tends to amplify sensor noise without providing any meaningful stabilization benefit. 

### 2. The Danger of Over-Gaining Attitude
In an attempt to aggressively stabilize the drone, we briefly increased attitude proportional gains ($k_R$ from 3.5 to 10.0). This was catastrophic. The large gains generated enormous correction torques at the slightest measurement noise, saturating the motor allocator (clipping motors to 1.0 or 0.0). Because of the saturation, the average thrust dropped below the required hover point, pinning the drone to the ground. **Lesson:** Attitude gains must leave headroom for base hover thrust.

### 3. Velocity Damping & The "Rocket Launch" Effect
During position control testing, the drone would shoot upward like a rocket, overshoot the target, and crash. 
*   **Cause:** The controller was underdamped. For a critically damped system, the velocity gain must relate to the position gain and mass ($k_v = 2\sqrt{k_x m}$). With $k_x = 7.0$ and $m = 2.064$, our $k_v = 6.0$ was too low. The initial 2m error commanded 97% throttle. As it overshot, the error flipped, the thrust command went negative, and since motors can't spin backward, thrust clipped to 0, causing a freefall.
*   **Solution:** We lowered $k_x$, increased $k_v$ for slight overdamping, added a thrust floor/ceiling clamp, and introduced a **setpoint ramp** (interpolating the target altitude over 3 seconds) to prevent massive initial velocity errors.

## 4. Final Control Architecture

The final cascaded implementation seamlessly links four control loops:
1.  **Position (NED)**: Computes position error and outputs a clamped desired velocity vector.
2.  **Velocity (NED)**: Computes accelerations ($a_N, a_E, a_D$). Thrust is extracted via gravity compensation. A Yaw rotation matrix maps the NED accelerations into the heading-relative body frame, which are then passed through an `arcsin` mapping to generate target roll and pitch angles.
3.  **Attitude**: Compares target Euler angles to current state (using the origin-latched orientation) to output desired body rates ($p, q, r$).
4.  **Rate**: High-frequency loop comparing desired vs actual body rates, outputting desired torques ($\tau_x, \tau_y, \tau_z$) to the motor mixer.

This project successfully demonstrated full-stack autonomous drone control, from raw odometry ingestion to direct PWM actuator allocation, overcoming significant hurdles in PX4 middleware integration and nonlinear control dynamics.
