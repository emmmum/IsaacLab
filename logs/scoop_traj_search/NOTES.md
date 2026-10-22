# RoboMaster lacrosse scoop -- status notes (2026-10-16)

Task: `Isaac-RoboMaster-Lacrosse-Direct-v0`
(`source/isaaclab_tasks/isaaclab_tasks/direct/robomaster_lacrosse/`, asset
`source/isaaclab_assets/isaaclab_assets/robots/robomaster_lacrosse.py`). All of these files are untracked/uncommitted.

## Machine issue
NVIDIA driver auto-upgraded 2026-10-16 06:40 (user-space 595.99, kernel module 595.91.07). Headless CUDA physics works,
RTX rendering fails ("No device could be created"). User is rebooting. After reboot, rendered video:
`./isaaclab.sh -p logs/scoop_traj_search/record_best.py --enable_cameras --render --out_dir logs/scoop_traj_search/best_run`

## Env fixes / features made in this session (all in the env + cfg)
- `gpu_collision_stack_size=2**29` (default 2**26 overflowed at 8192 envs -> dropped contacts; first policy exploited it).
- Robot spawn z 0.005 -> 0.182 (was buried, flipped on every reset).
- Ball ~1 m ahead (x 0.8-1.2, y +-0.15), robot and ball at rest (user's choice).
- Ball<->stick ContactSensor; `_ball_in_net` = stick contact & ball off ground (>3 mm) & within 0.15 m of pocket.
- `reward_stage` 1 (ball off ground near net) / 2 (in net by contact); reach reward = net lip -> ball bottom distance;
  -20 early-termination penalty; lift-quality factor exp(-forward/0.3) * exp(-(robot_speed/1.0)^2).
- Arm joint-target range shoulder -34..+23 deg, elbow -52..+23 deg (USD joints are unlimited).
- `arm_locked`, `arm_monotonic` (arm only raises, 0.05 rad/step max), `base_forward_only` (vx in [0, 1] m/s).

## RL results
- Many runs collapsed into local optima (arm folded to action-range corner, covering the ball). Fixed by lip-based reach,
  arm range limits, then curriculum.
- Stage 1 (`scoop_stage1_fwd`, arm locked, forward-only, model_150): ball popped onto net 99.4%, max clearance 5.2 cm,
  ball forward 3 cm, robot 1.23 m/s; 32% ball carried out of 3 m arena.
- Stage 2 resume failed: untrained arm outputs + monotonic arm -> arm rose before impact. Stopped. User then switched to
  hand-designed trajectory + (later) residual RL idea.

## Open-loop trajectory search (no uncertainty: slip off, lag 0.1 s fixed, ball 1.0 m dead ahead)
Old arm PD (Kp 200, Kd 20, 50 N m, implicit): best = v 0.8 m/s, lift at 0.8 s at 1 rad/s (shoulder +0.3, elbow +0.99 rad),
stop driving 0.5 s after lift, elbow initial -0.632 -> **96.0% / 150 runs** (`best_trajectory.json`). Lip must press into the
ground (raising it 5 mm -> ~0%). Residual failures: ball kicked forward at impact (contact chaos), even identical runs differ.
Plots/animation: `best_run/` (`trajectory_animation.mp4`, `trajectory_timeseries.png`).

Arm PD then changed (asset file) to Kp 500, Kd 5, 12 N m: grid success dropped 22% -> 5.6% (underdamped; only 0.5 rad/s lifts
work). Results `grid_kp500_el*.json`.

## Real-arm numbers from user
- link1 (upper arm) 358 g, 3D printed, 0.30 m; two motors 321 g each; stick 457 g, CoM 0.5 m from tip.
- Stick length from model: 1.146 m (lowerarm frame x -0.504..0.642; elbow at x=0) -> CoM 0.142 m in front of elbow.
- Motors: MIT control (q_des, qd_des, tau_ff, Kp, Kd). Kd = 3, torque limit 4 N m, vel limit 5 rad/s.
- Motor mounting CONFIRMED by user: shoulder motor on the chassis, elbow motor at the elbow (on link 1).
- Computed (uniform-rod stick; no gear-reflected inertia):
  I_elbow 0.059; I_shoulder 0.118 (initial) .. 0.155 (upper limit). With Kd 3, zeta 1.2: **Kp_sh ~10, Kp_el ~25**
  (zeta=1: 14.5 / 38). Torque budget with gravity FF: shoulder ~3.3 N m (gravity 1.1 + ball 0.65 + 10 rad/s^2 accel 1.55),
  elbow ~1.8 -> 4 N m OK if lift accel <= ~10 rad/s^2. Without gravity FF shoulder would sag ~6 deg at Kp 10.
- Current sim arm masses are wrong: upperarm 0.500 kg, lowerarm (incl. stick) 0.170 kg with CoM 0.34 m from elbow.

## Proposed next step (waiting for user OK + confirm motor mounting)
1. Arm actuator -> `IdealPDActuatorCfg` (explicit MIT-style: Kp e + Kd edot + tau_ff, total clipped to 4 N m),
   Kp_sh 10, Kp_el 25, Kd 3.
2. Commands: q_des (position target), qd_des = trajectory rate (velocity target), tau_ff = gravity comp
   (`robot.data.gravity_compensation_forces`) + lip press-down torque (new search param); later randomize FF +-20%.
3. Override arm body masses/CoM/inertia at startup to real values (upperarm 0.679 kg CoM ~0.22 m from shoulder; lowerarm
   0.457 kg CoM 0.142 m ahead of elbow, rod 1.146 m).
4. Re-run trajectory grid + 150-run validation; then add uncertainty / residual RL.

## 2026-10-16 evening: MIT arm implemented
- Asset arm actuator -> IdealPDActuatorCfg Kp {sh 10, el 25}, Kd 3, effort_limit 4, velocity_limit 5.
- Env: real masses written at startup (`arm_real_inertia`), velocity target from the monotonic ramp, feedforward =
  gravity comp (`arm_gravity_comp`) + `env._arm_ff_extra`. Verified: mass-matrix I_sh 0.1536 / I_el 0.0592 at lifted pose
  (matches hand calc); hold error with gravity comp sh -0.4 deg / el ~0 (without: 3.1 / 1.4 deg sag).
- Gravity comp cancels stick weight, so lip no longer presses the ground -> new search param: elbow press-down FF before lift.
- Grid `traj_grid_mit.py` -> `grid_mit.json`. Next: validate top combos (150 runs), then uncertainty (joint friction, slip,
  lip/ball-ground friction, restitution, gravity-comp error, latency). Wheel-axle friction has no effect while the base is
  velocity-driven (wheels cosmetic) -- needs user decision if wanted.

## 2026-10-16 late: uncertainty + base dynamics
- Uncertainty knobs added to the env cfg (all default off): arm joint friction (explicit Coulomb+viscous torque),
  gravity-comp scale, action delay, ball friction/restitution, robot friction scale (materials, once per env).
- Robustness of best MIT trajectory (old base model, 500 runs each): nominal 96.0, joint friction 94.8, gravcomp +-20%
  96.8, latency 0-33 ms 95.4, ball contact 94.8, robot friction 95.0, OLD slip (x0.7-1.2, drift) 43.8, all 41.0.
- Hover-then-set-down (old base model, 100 runs): lip must be on the ground at impact (hover 13 mm never set down: 3-7%);
  hovering while driving and pressing down with -1.5 N m at 0.3-0.5 s keeps 94-98%.
- Shoulder torque hitting 4 N m before impact was NOT lip drag (it also happens while hovering) -- it is the unrealistic
  base acceleration of the old lag-only model (4-16 m/s^2 from rest).
- User: real base accel ~1.7 m/s^2, top speed ~2 m/s, straight driving accurate, speed error +-10%.
  New cfg defaults: slip scale 0.9-1.1, no bias, vx noise 0.02, base_max_lin_accel 1.5, base_max_ang_accel 4.0 (assumed).
- Running: `traj_grid_accel.py` -> grid_accel.json (lift times shifted to 0.7-1.2 s). Next: validate top, check shoulder
  torque, re-check hover/set-down, robustness with the new realistic slip + other uncertainties.

## 2026-10-16 night: robust trajectory + alignment tolerance
- Grid under full realistic uncertainty (`traj_grid_accel.py --slip` + knobs): best `best_trajectory_robust.json`
  (v 0.9, lift 0.9 s @1 rad/s, sh +0.3, el +0.7, brake 1.0 s, press -1.0) = 97.6% / 500 randomized runs.
  Video/plots: `best_run_robust/`. Hover ~18 mm + set-down at 0.5-0.7 s also 95-98%; never set down 9-13%.
- Shoulder torque spikes (~3.4 N m) remain even while hovering and with accel limited; measured base accel jitters
  (p50 2.4, max 9.4 m/s^2) -> likely sim artifact of velocity-overwrite base + passive wheel contacts (unverified).
- Alignment tolerance (`traj_alignment*.py`, realistic uncertainty): lateral +-2 cm OK (95-98%), +-3 cm ~40-50%,
  +-4 cm 0%; heading +-1 deg OK, +-1.5 deg ~82%, +-2 deg ~0; distance: ball 15 cm closer OK (95%), 5 cm farther 92.5%,
  10 cm farther 35%. Hypothesis (unverified): curved scoop lip only touches the ground at its center at 48 deg.

## 2026-10-17: wheel-driven base
- New USD `robomaster_lacrosse_long_stick_wheeled.usd` (script `data/robomaster_lacrosse/make_wheeled_usd.py`):
  articulation root on chassis `base_link/base`, wheel SDF mesh + bolt colliders disabled, cylinder colliders
  (r 0.0512, w 0.045, friction 1.0/0.9), angular drives on wheel joints. Asset `ROBOMASTER_LACROSSE_WHEELED_CFG`
  (wheels: velocity mode, damping 2, 1.5 N m, 50 rad/s; root spawn z 0.005). Env `base_drive="wheels"` (default):
  (vx, wz) after slip/accel model -> wheel speed targets (vx - wz*y)/r; vy ignored. Legacy `"root_velocity"` kept.
- Straight 0.9 m/s: reaches 0.85 m/s in 0.62 s, no drift; base accel jitter p50 0.12 / max 1.55 m/s^2 (old 2.4/9.4);
  shoulder torque during approach max 0.76-1.6 N m (old 4.0 saturating) -> old jitter was a base-model artifact.
- Cannot turn in place (skid-steer scrub; +0.5 rad/s for 1 s -> 0.7 deg). Lip lifts ~12 mm during acceleration
  without press torque; with press -1.0 it stays down.
- Robust trajectory on wheeled base: 300/300 nominal, 300/300 with full realistic uncertainty. Video `best_run_wheels/`.
- Base mass (user: use the estimate): everything except upper arm + stick = 3.0 kg (`base_total_mass`, from RoboMaster
  EP Core ~3.3 kg spec); shoulder motor 0.321 kg on the `Cylinder` body; chassis 1.498 kg (inertia scaled); wheels
  0.232 kg each (now authored in the USD -- the cylinder colliders had made them 0.603 kg); total 4.14 kg.
- Wheel motor torque limit 0.15 N m (back-computed: 4.14 kg * 1.5 m/s^2 / 4 * 0.0512 = 0.08 + drag/margin), 40 rad/s.
- Exact cylinder colliders (`/physics/collisionCylinderCustomGeometry`, set in `_setup_scene`): with the convex
  approximation the light robot bounced and veered (18 deg); now 0.4 deg / 13 mm drift over 2.7 m.
- Robust trajectory on the 4.14 kg wheeled base: 300/300 nominal, 300/300 with full realistic uncertainty.

## 2026-10-17: floor tilt + imperfect ball (realistic uncertainty on, 300 runs each)
- Floor: per-env kinematic platform with random roll/pitch (`floor_tilt_max_deg`), ground plane lowered 0.2 m.
  +-0.2 deg 99.3%, +-0.3 deg 94.0%, +-1 deg 61.3% (a sphere rolls ~5 cm in 0.9 s on 1 deg; indoor floors ~0.1-0.3 deg).
- Ball (exact sphere): mass 137-145 g + CoM offset <=2 mm placed CoM-down -> 100% (placed randomly it rolled up to
  ~10 cm to settle -> 46%, an artifact). Radius 31.0/31.5/32.0/33.0 mm -> 100% each.
- Mesh (ellipsoid) ball: rests 2-5 mm into the floor (exact sphere 0.0 mm) and a ROUND mesh ball already drops to 85%
  -> PhysX convex-mesh contact artifact; ellipsoid results (64-78%) are not trustworthy. Marked experimental in cfg.
- Isaac Lab bug worked around: RigidObject.set_coms_index fails (float32 vs transformf buffer) -> set via root_view.

## 2026-10-20: 400-robot run, failure modes, impact torque, slides
- `render_array.py` (400 robots, 1200 Hz physics, 60 Hz control, ball dx +-3 cm / dy +-1 cm, robust trajectory):
  393/400 = 98.3% (`array_stats.json`, `failure_record.log`). Two failure modes in `failure_closeups/`:
  A = ball ~2.2-2.6 cm farther -> late contact, slides off the lip (#145, #203);
  B = lifted to ~43 cm, then thrown out of the pocket side (#172, #307).
- Impact torque in sim (`best_run_wheels/trajectory_record.npz`, applied torque, 60 Hz): first contact t=0.85 s,
  step shoulder -0.64 / elbow +0.75 N m vs approach sigma 0.006 / 0.04; lasts 1-2 samples. Lift start at 0.9 s gives
  3.5-3.9 N m, which would hide a late contact. Plot `deck_img/torque_impact.png`. Sim net/stick are rigid -> real peak
  likely smaller/smoother.
- Slides: `robomaster_lacrosse_scoop_progress_v2.pptx` (build script `build2.js`, video Drive links in `links.json`,
  filled 2026-10-20 from the shared folder; rebuild needs pptxgenjs).
  Drive folder: https://drive.google.com/drive/folders/1j5eB20g6uCwuTs54XmgiSta9ygdR1BRW (public listing works via
  embeddedfolderview?id=<folder id>).
- Next (user plan): measure real MIT motor torque at impact (>=500 Hz log, no-ball baselines, 0.5/0.9 m/s with ball,
  static calibration with the ball). Then contact-triggered lift in sim (torque threshold), residual RL with torque
  history only if needed. Torque does not help lateral/heading misalignment -> needs vision.

## 2026-10-20: demo, colored stick, animated USD
- Demo (GUI, loops rounds): `./isaaclab.sh -p scripts/demos/robomaster_lacrosse_scoop.py [--grid 10 --speed 0.25]`.
  Headless 4x100 robots: 97.8% (colored) vs 97.5% (grey USD) -> coloring has no physics effect.
- Animated USD: `--viz none --speed 100 --rounds 3 --save_usd ~/lacrosse_demo/scoop_array_anim.usd` (64 robots, 720
  frames, 24 MB); play with `scripts/demos/play_usd_animation.py --usd ...` or Isaac Sim File > Open + Play. Physics
  schemas must be REMOVED (not just disabled) or PhysX keeps the body transforms in Fabric and the robots freeze.
- Stick colors: `color_stick_usd.py --model_dir <extracted Nike rar>` binds MTL materials + diffuse textures
  (`stick_textures/`) to both long-stick USDs. The robot USDs are gitignored (*.usd) -> committed with `git add -f`.
- Env issue since ~10-17: torch cu130 cannot find libnvrtc-builtins.so.13.0 (in env_isaaclab/.../nvidia/cu13/lib);
  demo preloads it, other scripts need LD_LIBRARY_PATH=<that dir>.

## 2026-10-20: residual RL with sparse success reward
- Task `Isaac-RoboMaster-Lacrosse-Scoop-RL-Direct-v0` (`robomaster_lacrosse_scoop_rl_env[_cfg].py`): action = open-loop
  trajectory + residual (speed +-0.2 m/s, lift increment +-0.33 after t_lift only -- before t_lift the monotonic arm
  ratchets noise upward and lifts the lip: 0% success). Reward = 1 at time-out if ball in net for the last 1 s.
  Obs selectable with `env.obs_terms=[...]` (arm_pos, phase, arm_vel, arm_torque, base_vel). PPO 4096 envs, 300 it.
- Eval (`eval_scoop_rl.py`, deterministic, ~4100 episodes, one seed each):
  | policy | ball x 0.9-1.1 | ball x 0.97-1.03 | farthest bin 1.06-1.10 |
  | open loop (zero residual) | 78.1% | 97.9% | 25% |
  | A arm angles only | 61.1% | 84.2% | 1% |
  | C arm angles + phase | 92.7% | - | 75% |
  | B arm angles + torque + phase | 96.8% | 100.0% | 86% |
- Behavior (`rl_traces/open_vs_B.png`): B drives ~0.97 m/s, keeps the shoulder down and lifts the elbow ~55 deg
  (open loop 38 deg) -- nearly the same motion at 1.00 and 1.08 m, so most of the gain is a better open-loop motion
  (phase alone gets to 92.7%); torque adds ~4 points, mostly on the farthest balls. Angles-only obs is worse than no
  RL (pre-lift angles are ~constant; normalized noise?). Torque in sim is noise-free -> randomize before trusting.

## 2026-10-21: behavior cloning (pure MLP) + PPO fine-tune
- `env.residual=False`: the MLP outputs (speed, shoulder, elbow) directly. BC labels = trajectory actions, "hold"
  label -0.5 (any <= 0 holds the monotonic arm, margin against regression error). `bc_scoop.py`: 2 expert episodes x
  4096 envs (~2M samples), MSE 60 epochs on the PPO actor (+ normalizer), std 0.1, saved as model_0.pt; fine-tune with
  `train.py --resume --load_run <bc run> --checkpoint model_0.pt env.residual=False ...`.
- Chassis obs added (inaccurate on purpose): base_vel = vx * U(0.9, 1.1) per episode + N(0, 0.05) m/s; base_dist =
  integral of that.
- Results, ball x 0.9-1.1, deterministic (open loop 78%):
  | obs | BC | BC+PPO |
  | arm_pos + phase | 84.0% | 96.2% at it 100 (it 25: 94.9%), then PPO collapsed at ~it 130 -> 57-62% at it 299 |
  | + base_vel + base_dist (noisy) | 63.8% | 62.9% at it 299 (training success still rising slowly) |
- Takeaways: BC reproduces the trajectory; BC+PPO matches the best residual policy (96%) but needs a smaller step
  (lower lr / desired_kl) or checkpoint selection; noisy chassis inputs hurt both cloning and fine-tuning.

## 2026-10-21: within 1 cm, and success heatmap over ball position
- Ball x 0.99-1.01, y +-1 cm (8192 episodes): open loop 98.4%, residual C (angles+phase) 99.3%, BC+PPO it100
  98.4%, residual B (angles+torque+phase) 100.0%.
- `heatmap_scoop.py`: ball on a grid (distance -15..+15 cm in 2 cm, lateral -4..+4 cm in 1 cm), 28 runs per cell,
  one synchronized episode. Plot `heatmap/heatmap_success_vs_ball_pos.png`.
- Lateral tolerance is ASYMMETRIC: every method fails at -2..-4 cm (robot's right) but still works at +4 cm; the
  window is centered near +1.5 cm. Aiming the robot ~1.5 cm to the right of the ball should widen the usable lateral
  tolerance to about +-2.5 cm (cause not checked yet: pocket/lip geometry or stick mount offset). Distance: open loop
  holds -15..+3 cm; the RL/BC policies extend the far side to about +7..+9 cm.

## 2026-10-21: ball centered on the stick head (user)
- The stick head is not on the robot's center line: lip / pocket at y = +0.016 m (STICK_LIP_OFFSET[1]). User: place
  balls in front of the stick head, not the robot. New cfg `ball_spawn_y_center = STICK_LIP_OFFSET[1]` (default); the
  y spawn range is now relative to it. Demo, render_array.py and heatmap_scoop.py follow it. Older results (and the
  grid searches / alignment numbers above) were relative to the robot's center line.
- Stick-centered (`heatmap/heatmap_stick_centered.png`): symmetric. Open loop 100% for lateral -2..+2 cm over
  distance -15..+3 cm; 8192/8192 within 1 cm. Residual B (trained robot-centered) also symmetric, 100% for -2..+2 cm
  over -15..+7 cm.

## 2026-10-21: uneven floor (polygon terrain), ball settles first, rolling resistance
- `env.rough_floor=True` (`rough_floor.py`): per env one 3.2 x 1.0 m triangle-mesh tile (4 cm grid) = smooth random
  bumps + planar tilt; 8 levels, level k = k/7 of max (defaults for a lab floor: 1 mm RMS bumps, 0.15 m correlation,
  0.3 deg tilt). Startup: each env's ball dropped 6x near its tile center, simulated 1.5 s, positions where it rests
  (< 3 mm/s, inside the tile) kept; each episode the ball is put at one of them and the robot 1 m behind it (stick head
  aligned), height/roll/pitch fitted to the floor under its wheels. Success uses the floor height under the ball.
- Rolling resistance (all floors): torque 0.015 * m g r against the ball's rolling while it touches the floor
  (PhysX has none; without it balls never came to rest on any slope). Flat-floor open loop unchanged (4083/4096).
- Performance lessons: (1) envs must not share a tile -- overlapping stick SDFs made 256 envs ~15x slower and broke
  the physics; (2) one mesh per tile, not one huge mesh -- with a single mesh, 1024 envs were 36x slower than flat.
  Now 4096 envs: 70 s vs 27 s per 4 s episode (2.6x), +2.5 min startup for settling, no contact-buffer overflow.
- Open loop on the lab-floor levels (8192 episodes): L0 99.7, L1 98.0, L2 96.5, L3 95.7, L4 94.4, L5 93.5,
  L6 92.2, L7 (1.0 mm, 0.3 deg) 91.8% -> overall 95.2%. Ball rests in >= 93% of drops even at L7.

## 2026-10-22: retarget the scoop to a Franka Research 3 (FK of the stick, IK on FR3)
- Scoop speed (open loop, 16 runs): contact at t = 0.85 s; lip on the floor moving +x at 0.73 m/s (base 0.9 m/s;
  the soft arm lags), stick pitched 48.5 deg; ball 10 cm up 0.18 s and 30 cm up 0.36 s after contact; stick moves
  0.47 m in those 0.36 s, |w| up to 2.1 rad/s. Raw recording has impact shocks (~40 m/s^2, 65 rad/s^2) -> smoothed.
- `fr3/record_stick_traj.py` records the stick (lowerarm) world motion. `fr3/plan_fr3.py` (pure torch, FR3 = Panda
  modified DH): reference smoothed (Gaussian 30 ms); quintic approach from rest onto the recorded motion 0.15 s
  before contact; recording followed exactly 0.4 s after contact; then playback rate ramps 1 -> 0 over 0.6 s (gentle
  stop, ~1.5 m/s^2) and hold. Batch search (16384) over ball distance/bearing, scoop heading, base height, grip
  point on the shaft, grip roll, TCP flip; DLS IK, feasible = < 2 mm / 0.01 rad tracking, joint margin > 0.05 rad,
  qd and qdd < 80% of limits (qdd limits = Panda datasheet values, ASSUMED for FR3).
- Best: FR3 base 0.50 m above the floor, ball 0.56 m from the base axis at bearing 91 deg, scoop heading 47 deg,
  grip 5 cm behind the old elbow, fingers across the shaft; uses 58% of joint velocity and 78% of joint acceleration
  limits. `make_fr3_lacrosse_usd.py` builds fr3_lacrosse.usd (FR3 from the Isaac library + stick link welded to
  fr3_hand at that grip, real stick mass/inertia).
- Physics (`fr3/fr3_scoop_sim.py`, PD 4000/150 + velocity feed-forward, arm gravity off, ball rolling resistance):
  joint tracking < 0.004 rad, lip within ~5 mm of the plan; success 255/256 (ball +-1 cm), 248/256 (+-2 cm),
  207/256 (+-3 cm). Videos fr3/fr3_scoop_realtime.mp4, fr3/fr3_scoop_slowmo6x.mp4.

## 2026-10-22: FR3 on a Clearpath Husky (user's real setup)
- No Husky in the Isaac asset library (Clearpath: Dingo, Jackal, Ridgeback). Took husky_description (BSD) from
  github.com/husky/husky, expanded the xacro, converted to USD (`source/isaaclab_assets/data/husky/`).
- User: FR3 base on the Husky top plate, 0.377 m above the floor (initially said ~0.30). Mount position along the
  Husky not given -> searched (center line, facing forward).
- Planner `--base_h 0.377 --husky`: Husky body as a box (+-0.55 m x, +-0.39 m y incl. wheels/bumpers) below the base
  plane, 3 cm clearance for stick sample points and arm joint frames, ball not under the robot; plus 3 rounds of local
  refinement around the 64 best candidates. Best: FR3 at the FRONT of the top plate (mount_x 0.34 m, the search bound
  0.35), ball front-right (bearing -34 deg, 0.64 m from the FR3 base axis), scoop sweeping to the robot's right
  (heading -91 deg); 50% of joint velocity, 60% of joint acceleration limits.
- Physics (`fr3_scoop_sim.py --husky`, fr3_lacrosse_husky.usd): 252/256 (ball +-1 cm), 243/256 (+-2 cm); tracking
  < 0.005 rad. Videos fr3/fr3_husky_scoop_realtime.mp4, fr3/fr3_husky_scoop_slowmo6x.mp4.
