# Lacrosse Scoop Policy — Browser-Ready Weights

Everything needed to run the trained scoop policy's forward pass in a browser,
no backend required.

| File | What it is |
|---|---|
| `lacrosse_scoop_policy.json` | All weights + observation normalization stats, exported from `lacrosse_runs/dense_v16/dense_v16/checkpoints/best_agent.pt` |
| `policy.js` | Dependency-free JS forward pass matching the JSON exactly (verified against the original PyTorch checkpoint) |
| `export_weights.py` | Re-run this against any other checkpoint to regenerate the JSON |

## Quick start

```html
<script src="policy.js"></script>
<script>
  fetch('lacrosse_scoop_policy.json')
    .then(r => r.json())
    .then(model => {
      const obs = [0, 0, 0, 0, 0, 0, -0.9, 0.6, 0, 0, 0];
      const { action, value } = runPolicy(obs, model);
      console.log(model.action_labels.map((n, i) => `${n}: ${action[i].toFixed(3)}`));
      console.log('value estimate:', value);
    });
</script>
```

## What the 11 inputs and 3 outputs mean

Full task/robot context is in `docs/lacrosse_scoop_readme.md`; the short version:

**Observation (`obs_labels`, in order):**

| Index | Name | Meaning |
|---|---|---|
| 0 | `x` | chassis position (m) |
| 1 | `q1` | shoulder angle (rad) |
| 2 | `q2` | elbow angle (rad) |
| 3 | `x_dot` | chassis velocity (m/s) |
| 4 | `q1_dot` | shoulder angular velocity (rad/s) |
| 5 | `q2_dot` | elbow angular velocity (rad/s) |
| 6 | `ball_dx` | ball x − pocket x (m) — pocket x comes from forward kinematics, see the readme |
| 7 | `ball_dz` | ball z − pocket z (m) |
| 8 | `ball_rot_y` | ball orientation proxy (quaternion y-component) |
| 9 | `ball_vx` | ball velocity x (m/s) |
| 10 | `ball_vz` | ball velocity z (m/s) |

All 11 values must be **raw, un-normalized** physical units — `policy.js` applies
the same running-mean/variance normalization the network was trained with
internally (`obs_normalization` block in the JSON). Don't normalize them yourself
before calling `runPolicy`.

**Action (`action_labels`, in order), each already clipped to [−1, 1]:**

| Index | Name | Meaning |
|---|---|---|
| 0 | `chassis_accel_cmd` | chassis acceleration command — in the real env this gets multiplied by `vx_accel` (≈0.215 for this checkpoint) and integrated into a velocity target |
| 1 | `shoulder_vel_cmd` | shoulder velocity command — multiplied by `vq_scale = 2.0 rad/s` |
| 2 | `elbow_vel_cmd` | elbow velocity command — multiplied by `vq_scale = 2.0 rad/s` |

If your interactive page wants to show physically-meaningful units instead of
raw [−1, 1] actions, multiply by those same constants (`vx_accel`, `vq_scale`)
as documented in `lacrosse_scoop_demo/demo_lacrosse_scoop.py`.

## A few things worth building into an interactive page

- **Sliders for `ball_dx` / `ball_dz` / ball velocity** are the most interesting
  inputs to expose — they're what changes moment-to-moment during a real scoop,
  while `x, q1, q2` sweep from the home pose (all zero) through the motion.
- **The value head output** (`value` in `runPolicy`'s return) is the critic's
  estimate of expected future return from that state — useful to visualize
  alongside the action as a sanity signal (should rise as the state approaches
  a successful hold).
- **`hidden.h1` / `hidden.h2`** (also returned by `runPolicy`) are the 256- and
  128-dim activations after each ELU layer — handy if you want to visualize
  neuron activations rather than just the final 3 action numbers.
- The network was trained on inputs drawn from a fairly narrow real-episode
  distribution (chassis starts at the ball ~0.8–1.0 m away, home joint angles
  are all 0). Feeding it wildly out-of-distribution sliders (e.g. `q1 = 10`)
  will mostly just saturate the output at ±1 — not a bug, just distribution
  shift, same as any other network.
