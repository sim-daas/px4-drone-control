# Sim-to-Sim Transfer Analysis: PyBullet (PyFlyt) to Gazebo (PX4 SITL)

## Executive Summary
This report analyzes the challenges and failure modes encountered when transferring a Stable-Baselines3 Soft Actor-Critic (SAC) reinforcement learning policy—which achieved perfect hover in PyBullet—into the PX4 Gazebo SITL environment. Despite careful coordinate alignment, observation state matching, and post-hoc actuator scaling (thrust compensation), the policy exhibited high-frequency instability. This highlights critical discrepancies in the physics fidelity, timing, and control-loop paradigms between the two simulation engines.

---

## 1. The Transfer Pipeline
The goal was to feed the trained model's low-level decision-making into the physical hardware simulator.
- **Observations:** A 16D vector tracking Position Error, Quaternion, Linear Velocity, Angular Velocity, and Integral error.
- **Actions:** 4D output specifying desired Body Rates (Roll, Pitch, Yaw) and Normalized Thrust.
- **Middleware:** A ROS 2 wrapper mapping PX4's NED state into PyBullet's NWU expected format.

---

## 2. Core Failure Modes and Mitigations

### A. The Coordinate & Reference Frame Trap
- **Issue:** PyBullet standardizes local body velocities, while raw PX4 telemetry yields world-frame velocities.
- **Fix:** Implemented a World-to-Body rotation matrix on the ROS 2 boundary.

### B. The Actuator/Thrust Scaling Discrepancy
- **Issue:** 
  - PyBullet URDF simulated a 2.0kg drone demanding roughly ~49% thrust to stay stationary. 
  - Gazebo simulated a realistic X500 frame requiring ~71% motor effort to counteract gravity.
- **Fix:** Dynamically mapped motor outputs using an empirical $1.45\times$ scalar multiplier. However, the non-linear relationship between motor RPM and aerodynamic lift meant scaling was mathematically imperfect.

---

## 3. Why the Transfer Failed: The "Reality Gap" (Sim-to-Sim)

Even with mathematical scaling, the control loop destabilized due to deep architectural assumptions.

### I. Simplified vs. High-Fidelity Physics Models
* **PyBullet (PyFlyt):** Typically uses generalized rigid body physics. Thrust is applied as a direct vector force acting cleanly on the center of mass.
* **PX4 SITL/Gazebo:** Simulates the complex aerodynamics of spinning blades. It incorporates blade-element theory, ground effect (aerodynamic cushion near surfaces), and motor torque reaction forces. The RL agent's "clean" PyBullet logic breaks when applied to these chaotic forces.

### II. Control Latency and DDS Transport Jitter
* **PyBullet:** Operates in deterministic, synchronous lock-step. A control action instantly updates the physics on the next tick.
* **PX4 over ROS 2:** Operates asynchronously. Data crosses the MicroXRCE-DDS bridge, introducing inherent communication latency (milliseconds) and jitter. For an RL agent operating at 30Hz without a history of past actions, a slight delay in control execution leads to aggressive, over-corrective oscillations.

### III. Sensitivity of Rate Control
Because the agent was commanding *rates*, any lag between commanding a target rate and achieving it via the mixer creates an accumulation of attitude error. 

---

## 4. Architectural Recommendations for Future Runs

To successfully bridge the gap, the training methodology must change:

1. **Domain Randomization:** Train the SAC agent in PyBullet with randomized mass, inertia vectors, drag coefficients, and artificial control latency (5–30ms jitter).
2. **Shift to Attitude Control:** Rather than commanding high-frequency body rates, the RL agent should output target Roll, Pitch, and Yaw angles. Let the heavily-optimized PX4 onboard PID loops process the rates.
3. **Continuous Actions & Filtered Outputs:** Encourage smoother policy actions during training using rewards penalizing massive action steps.
