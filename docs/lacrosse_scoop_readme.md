# Lacrosse Scoop — RL Policy → Open-Loop Real-Robot Replay

This document explains how the scooping policy for the 3-DOF `lacrosse_scoop` robot
(chassis slide `x` + shoulder `q1` + elbow `q2`) was trained, exactly what reward
signal it learned from, and how a single successful simulated trajectory was
extracted and played back **open-loop** on the physical robot.

This is the robot and reward that were actually deployed to hardware — not the
separate x/z/pitch gantry project described in `scoop_project.md` at the repo root
(that is a different, earlier robot model).

Source of truth: `source/isaaclab_tasks/isaaclab_tasks/direct/lacrosse_scoop/lacrosse_scoop_env.py`.

---

## 1. Task & Robot

**Robot**: 3-DOF mobile manipulator — a lacrosse-stick end-effector on a 2-link arm
mounted on a sliding chassis (`source/isaaclab_assets/data/robots/lacrosse_manipulator/robot_v14.usda`).

| Joint | DOF | Control | Limits |
|---|---|---|---|
| `ChassisSlideJoint` (x) | prismatic | velocity target | ±3.0 m/s, 500 N |
| `ShoulderJoint` (q1) | revolute | velocity target | ±4 rad/s, 200 N·m |
| `ElbowJoint` (q2) | revolute | velocity target | ±4 rad/s, 200 N·m |

Ball: 32 mm-radius sphere, 0.145 kg, μs=0.8/μd=0.7, restitution 0.3. Constrained to
the x–z plane during training (vy, roll, yaw zeroed every step).

**Task**: starting from the home pose (`x=q1=q2=0`), drive the pocket under a ball
that spawns on the ground `0.8–1.0 m` in front of the chassis, scoop it up, and hold
it (near-zero ball speed, in contact) until a success hold counter fires.

Physics runs at 1200 Hz; the policy runs at 60 Hz (20 substeps/step, `decimation=20`).
Episodes are capped at 10 s (600 policy steps).

Pocket position is not measured directly — it's computed analytically from the two
arm joint angles via forward kinematics:

```
theta = q1 + q2
pocket_x = x + L_arm·sin(q1) − 0.323·cos(theta) − 0.07·sin(theta)
pocket_z =     Z0 + L_arm·cos(q1) + 0.323·sin(theta) − 0.07·cos(theta)
```
with `L_arm = 0.30 m`, `Z0 = 0.15 m` (shoulder height).

---

## 2. RL Formulation

**Observation (11-dim, no history/LSTM):**

| # | Field |
|---|---|
| 1–3 | joint positions: x, q1, q2 |
| 4–6 | joint velocities: ẋ, q̇1, q̇2 |
| 7 | ball x − pocket x |
| 8 | ball z − pocket z |
| 9 | ball orientation (quaternion y-component, proxy for spin) |
| 10–11 | ball velocity: vx, vz |

**Action (3-dim, ∈ [−1, 1]):**

| # | Effect |
|---|---|
| a[0] | chassis **acceleration** — added to an internal velocity target each step, clamped to ±2 m/s |
| a[1] | shoulder velocity target = a[1] × 2.0 rad/s |
| a[2] | elbow velocity target = a[2] × 2.0 rad/s |

The chassis uses acceleration (not direct velocity) so the policy has to actively
decelerate before contact rather than snapping to a velocity target — this is
controlled by a curriculum: the per-step acceleration limit (`vx_accel`) ramps from
0.3 m/s/step down to 1/60 m/s/step between policy steps 30k–130k, forcing
progressively finer chassis control.

**Termination:**

| Condition | Trigger |
|---|---|
| `success` | contact + ball lifted above 2.5×radius + ball speed < 0.5 m/s, held for 5 consecutive steps |
| `head_ground` | pocket z < 0.05 m (stick would hit the floor) |
| `ball_oob` | ball x outside ±2.5 m of env origin |
| `ball_high` | ball z > 0.8 m (flung) |
| `timeout` | 600 steps (10 s) reached |

---

## 3. Reward — every term

All terms are summed into a single scalar per policy step (60 Hz). This is the
**final, currently-checked-in** reward (the training script calls this run
`dense_v16`); it replaced an earlier one-shot "milestone bonus" design that
collapsed during training (see [§5](#5-version-history-why-the-reward-looks-like-this)).

**Approach shaping**

- **Chassis-toward-ball** (weight 0.8) — only active while the pocket is more than
  0.35 m from the ball in x. Disabled up close to prevent the policy from just
  rushing through the ball and knocking it out of bounds.
  ```
  r_chassis = 0.8 · clamp(ẋ · −sign(dx), 0) · [|dx| > 0.35]
  ```
- **X-approach** (weight 2.0) — always on, exponential reward for closing the
  horizontal gap.
  ```
  r_x = 2.0 · exp(−3·|dx|)
  ```
- **Z-approach** (weight 1.0) — only once the pocket is already close in x, so the
  policy doesn't get an early gradient to just drop the arm from far away.
  ```
  r_z = 1.0 · exp(−3·|dz|) · [|dx| < 0.35]
  ```

**Contact & manipulation**

- **Contact** (weight 5.0) — scales with measured end-effector/ball contact force.
  ```
  r_contact = 5.0 · clamp(F_contact / 10 N, 0, 1)
  ```
- **Lift** (weight 5.0) — any ball height above resting height.
  ```
  r_lift = 5.0 · clamp(ball_z − r_ball, 0) / r_ball
  ```
- **Scoop** (weight 8.0) — rewards the elbow actively rising while in contact and
  before the ball is lifted — this directly shapes the sweeping scoop motion.
  ```
  r_scoop = 8.0 · [contact] · [ball_z < 2.5·r_ball] · clamp(q̇2 / 2.0, 0, 1)
  ```
- **Holding** (weight 10.0) — fires every step the ball is above the success
  height and in contact.
  ```
  r_holding = 10.0 · [contact] · [ball_z > 2.5·r_ball]
  ```
- **Combo streak** (weight 0.2) — grows over 5 consecutive steps of continuous
  ball-off-ground contact; small so it never outweighs the terms above.
- **Velocity-inverse** (weight 2.0, ε=0.05) — dense gradient to keep decelerating
  once holding: worth ~40/step at rest, ~2/step at 1 m/s.
  ```
  r_vel_inv = 2.0 · [holding] / (‖v_ball‖ + 0.05)
  ```
- **Deceleration** (weight 30.0) — same idea, but active as soon as the ball is
  lifted (doesn't require contact — also rewards a ball resting in the pocket
  without the arm pushing on it).
  ```
  r_decel = 30.0 · exp(−‖v_ball‖ / 0.5) · [ball_z > 2.5·r_ball]
  ```

**Safety**

- **Soft safety** (weight 8.0/m) — continuous penalty as the pocket nears the
  ground-contact height, so the policy feels the wall coming before it crashes.
- **Head-ground** (weight −50) — one-off penalty exactly when the hard
  `head_ground` termination fires.
- **Alive bonus** (+0.005/step) — small constant survival reward.

There is **no single dominant sparse success reward** in this version — success is
purely a termination condition, and the dense terms above (contact → lift → scoop →
hold → decelerate) are shaped so that chaining them naturally produces the scoop
behavior.

---

## 4. Training Setup

| | |
|---|---|
| Algorithm | PPO (skrl), shared actor-critic backbone (`separate=False`) |
| Network | `Linear(11→256)→ELU→Linear(256→128)→ELU` → 3-dim policy head + 1-dim value head, ~36.5k params |
| Parallel envs | 256 |
| Rollout length | 32 steps, 8 learning epochs, 8 minibatches |
| Discount γ / GAE λ | 0.99 / 0.95 |
| Learning rate | 3e-4, KL-adaptive (threshold 0.008) |
| Entropy scale | 0.01 |
| Clip ratio / value clip | 0.2 / 0.2 |
| Obs & value preprocessing | RunningStandardScaler (online) |
| Total steps | 400,000 policy steps (256 envs), trained from scratch |

Command used: `./isaaclab.sh -p train_lacrosse_scoop.py --headless` (see
`train_lacrosse_scoop.py`).

---

## 5. Version History — why the reward looks like this

This reward is the product of several failed designs, documented in
`docs/v14_experiment.md`:

- **Early dense versions** rewarded approach + contact + lift with hand-tuned
  weights; the policy learned to "rush through" the ball (knocking it out of
  bounds) to farm the approach reward — fixed by disabling the chassis-velocity
  and z-approach terms once close to the ball (still present above).
- **Milestone-bonus design (an earlier "v14")** replaced dense shaping with
  one-time bonuses (contact → lift → hold → near-stationary hold: 100/200/400/800)
  and *no success termination*. It reached the "hold" milestone in ~54/2000
  episodes at 50k steps, then **collapsed back to zero** by 126k steps — because
  episodes always ran the full 600 steps regardless of success, so the value
  function got no clean signal for the long post-success window, destabilizing
  training.
- **This version** fixes both problems: it restores dense per-step shaping for
  every phase of the task *and* adds an explicit `success` termination once the
  ball is held near-stationary for 5 consecutive steps, so successful episodes end
  immediately instead of running to timeout.

---

## 6. What "success" looks like in sim

No rigorous held-out success-rate evaluation was run for this checkpoint (this is
a real gap — see Limitations). Two things we do know from this run:

- **Training telemetry** (`lacrosse_runs/dense_v16/dense_v16/events.out.tfevents.*`):
  mean episode length converges to ~580/600 steps by the end of training, i.e. most
  episodes still run close to the 10 s timeout rather than terminating early on
  success — the policy is far from a "fast, reliable success" regime.
- **At least one clean success trajectory exists and was captured.** The
  environment itself saves the *first* successful episode's full joint + ball
  trajectory to `/tmp/ball_traj_success.npz` the moment `success` fires
  (see `_get_dones` in `lacrosse_scoop_env.py`), and it can be replayed in Isaac
  Sim with `replay_lacrosse.py`. `ball_trajectory_v7.png` shows an example from an
  earlier version of this same reward family: the ball lifts off the ground at
  t≈1.05 s and crosses the success-height threshold at t≈1.07 s.

So the honest framing: this is a **proof-of-concept scoop policy that reliably
produces at least one clean, physically-consistent scoop-and-hold trajectory**, not
a policy with a measured high sim success rate. That single trajectory is what got
carried over to hardware.

---

## 7. Sim-to-Real: extracting and replaying the trajectory

What was actually done, in order:

1. Ran the trained `dense_v16` policy in sim with the ball spawned ~1 m in front of
   the robot (within the trained spawn range `x ∈ [−1.0, −0.8] m`).
2. Captured the **joint-space** sequence `(x, q1, q2)` at 60 Hz for every step from
   episode start to the moment of successful scoop (this is exactly the recording
   pipeline already built into the env: `_traj_buf` / `/tmp/ball_traj_success.npz`,
   also exposed via `record_lacrosse_v5*.py` for visualization).
3. Took that fixed `(x(t), q1(t), q2(t))` sequence — **no ball position feedback,
   no policy inference, no contact sensing** — and replayed it **open-loop** as a
   joint-position (or velocity) reference directly on the physical robot, with a
   real lacrosse ball placed ~1 m in front of it at the start.
4. Result: **the open-loop replay scoops the ball successfully on roughly half of
   the attempts.**

This is deliberately the simplest possible sim-to-real path — it skips perception,
closed-loop control, and policy inference entirely on the real robot. It is a
timed-motion playback, not a deployed policy.

---

## 8. Why only ~50%, and not more

The policy was trained with **perfect, noise-free simulated state** (ball position
is ground truth, robot dynamics are the implicit PD/velocity actuator model in
PhysX). Replaying the resulting motion open-loop removes every source of
correction the policy would otherwise have:

- **No feedback on ball position.** In sim the trajectory was generated for one
  specific ball spawn point; on hardware, ball placement, rolling, and initial
  bounce vary attempt-to-attempt, so the same blind motion doesn't always align
  the pocket with the ball.
- **No perception loop at all on hardware** — the entire premise of this
  experiment is a fixed motion, so any real-world deviation (surface friction,
  ball spin, an off-by-a-few-cm placement) cannot be corrected mid-motion.
- **Actuator/timing mismatch.** Sim uses velocity-controlled implicit actuators
  with idealized 60 Hz control and no communication latency; real motor
  controllers, encoder latency, and mechanical backlash change the realized
  joint trajectory even when commanded identically.
- **No domain randomization was used for this policy** (unlike the `v14_dr`
  checkpoint in the other gantry project) — no friction/mass/observation-noise
  randomization was applied during training, so the motion itself is tuned to one
  narrow simulated physical setting.

Given all of that, a ~50% blind open-loop transfer is a reasonable result — it is
evidence the *shape* of the learned motion is physically valid on real hardware,
not that the policy has closed the sim-to-real gap.

---

## 9. Limitations & Natural Next Steps

- **No sim held-out success rate** for `dense_v16` — only training telemetry and
  one saved success trajectory. Worth running a frozen-checkpoint eval (e.g. 200+
  episodes) before drawing further conclusions.
- **Real-world trial count is small and informal** ("about half") — worth logging
  N attempts with a fixed ball placement protocol to get an actual success rate
  with a confidence interval.
- **Open-loop replay is a stepping stone, not the end state.** The natural next
  step is closing the loop on hardware: read back real ball position (camera or
  marker tracking) and re-run the trained policy's inference on the real encoder
  feedback each control step, instead of replaying a fixed motion.
- **No domain randomization / observation noise** was used in training this
  policy — adding it (as was done for the other project's `v14_dr`, see
  `scoop_project.md`) would likely improve robustness to the real ball-placement
  and friction variance that's the main suspected failure mode above.

---

## File Map

| File | Purpose |
|---|---|
| `source/isaaclab_tasks/isaaclab_tasks/direct/lacrosse_scoop/lacrosse_scoop_env.py` | Environment + reward (current) |
| `train_lacrosse_scoop.py` | Training entry point (`dense_v16` run) |
| `lacrosse_runs/dense_v16/dense_v16/checkpoints/` | Trained checkpoints |
| `play_lacrosse_scoop.py` | GUI inference/visualization of a checkpoint |
| `record_lacrosse_v5*.py`, `make_replay_usda.py`, `save_replay_usda.py` | Export a recorded trajectory as a Blender-viewable USD animation |
| `replay_lacrosse.py` | Replay `/tmp/ball_traj_success.npz` inside Isaac Sim |
| `record_ball_success.py` / `record_ball_simple.py` | Run to first success, plot ball x/z/vx/vz vs time |
| `ball_trajectory_v7.png` | Example plotted success trajectory (earlier version of this reward family) |
| `docs/v14_experiment.md` | Postmortem of the milestone-reward design that preceded this one |
