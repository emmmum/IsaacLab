# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop scoop trajectory grid search: no slip randomization, ball fixed 1.0 m dead ahead."""
import argparse, itertools, json, sys
import gymnasium as gym, torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config
p = argparse.ArgumentParser(); p.add_argument("--task", default="Isaac-RoboMaster-Lacrosse-Direct-v0")
p.add_argument("--reps", type=int, default=150); p.add_argument("--combos", required=True); p.add_argument("--out", default="traj_grid.json")
add_launcher_args(p); a, rem = setup_preset_cli(p); sys.argv = [sys.argv[0]] + rem

V = [0.7, 0.8, 0.9, 1.0]                    # [m/s] drive speed
T_LIFT = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # [s] arm lift start
RATE = [1.0, 2.0, 3.0]                       # [rad/s] lift rate (3 = max increment)
D_SH = [0.0, 0.3, 0.6]                       # [rad] shoulder raise (max 0.647)
D_EL = [0.4, 0.7, 0.99]                      # [rad] elbow raise (max 0.993)
T_BRAKE = [0.2, 0.5]                         # [s] stop driving this long after lift start
WINDOW = 4.0; HOLD = 1.0

@hydra_task_config(a.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    combos = [tuple(r[k] for k in ['v','t_lift','rate','d_sh','d_el','t_brake','pre_el']) for r in json.load(open(a.combos))]
    with launch_simulation(env_cfg, a):
        n = len(combos) * a.reps
        env_cfg.scene.num_envs = n; env_cfg.episode_length_s = WINDOW + 2.0
        env_cfg.base_slip_scale_range = (1.0, 1.0); env_cfg.base_slip_bias_max = (0.0, 0.0, 0.0)
        env_cfg.base_slip_noise_std = (0.0, 0.0, 0.0); env_cfg.base_slip_lag_tau_range = (0.1, 0.1)
        env_cfg.ball_spawn_x_range = (1.0, 1.0); env_cfg.ball_spawn_y_range = (0.0, 0.0)
        env_cfg.arm_locked = False; env_cfg.arm_monotonic = True; env_cfg.base_forward_only = True
        env = gym.make(a.task, cfg=env_cfg).unwrapped; env.reset()
        dev, c = env.device, env.cfg
        P = torch.tensor(combos, device=dev, dtype=torch.float32).repeat(a.reps, 1)
        v, tl, rate, dsh, del_, tb, pre = P.unbind(-1)
        dmax_lift = torch.stack([dsh, torch.maximum(del_, pre)], -1)
        dmax_pre = torch.stack([torch.zeros_like(pre), pre], -1)
        lip_z_at = torch.zeros(n, device=dev)
        default = env._default_arm_joint_pos
        steps = int(round(WINDOW / env.step_dt)); hold_steps = int(round(HOLD / env.step_dt))
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        streak = torch.zeros(n, device=dev); ever_pop = torch.zeros_like(alive)
        nan = float("nan"); pop_fwd = torch.full((n,), nan, device=dev); pop_rv = pop_fwd.clone()
        max_clr = torch.zeros(n, device=dev)
        for k in range(steps):
            t = k * env.step_dt
            act = torch.zeros(n, 5, device=dev)
            act[:, 0] = torch.where(t < tl + tb, 2.0 * v / c.base_lin_vel_scale - 1.0, torch.full_like(v, -1.0))
            dmax = torch.where((t >= tl)[:, None], dmax_lift, dmax_pre)
            remaining = (dmax - (env._arm_target - default)).clamp(min=0.0)
            inc = torch.minimum((rate / (c.arm_increment_max / env.step_dt))[:, None].expand(-1, 2) * c.arm_increment_max, remaining)
            arm_a = (inc / c.arm_increment_max).clamp(0.0, 1.0)
            act[:, 3:5] = arm_a
            _, _, term, trunc, _ = env.step(act)
            if k == int(0.6 / env.step_dt): lip_z_at = (env._lip_pos_w()[:, 2] - env.scene.env_origins[:, 2]).clone()
            alive &= ~(term | trunc)
            innet = env._ball_in_net & alive
            streak = torch.where(innet, streak + 1, torch.zeros_like(streak))
            o = env.scene.env_origins; ball = env.ball.data.root_pos_w.torch
            clr = ball[:, 2] - o[:, 2] - c.ball_radius
            first = innet & ~ever_pop
            pop_fwd[first] = (ball[:, 0] - o[:, 0] - env._ball_spawn_pos[:, 0])[first]
            pop_rv[first] = torch.linalg.norm(env.robot.data.root_lin_vel_w.torch[:, :2], dim=-1)[first]
            ever_pop |= innet
            max_clr = torch.where(alive, torch.maximum(max_clr, clr), max_clr)
        ok = (streak >= hold_steps) & alive
        final_clr = (env.ball.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2] - c.ball_radius)
        robot_x = env.robot.data.root_pos_w.torch[:, 0] - env.scene.env_origins[:, 0]
        nc = len(combos)
        okm = ok.view(a.reps, nc).float().mean(0)
        res = []
        for i, cb in enumerate(combos):
            idx = [i + r * nc for r in range(a.reps)]
            res.append(dict(v=cb[0], t_lift=cb[1], rate=cb[2], d_sh=cb[3], d_el=cb[4], t_brake=cb[5], pre_el=cb[6], lip_z_mean=lip_z_at[idx].mean().item(), lip_z_std=lip_z_at[idx].std().item(),
                            success=okm[i].item(), final_clr=final_clr[idx].mean().item(), max_clr=max_clr[idx].mean().item(),
                            pop_fwd=pop_fwd[idx].nanmean().item(), pop_robot_v=pop_rv[idx].nanmean().item(), robot_x=robot_x[idx].mean().item()))
        json.dump(res, open(a.out, "w"), indent=1)
        print("=" * 70)
        for r in res:
            print(f"VAL pre_el={r['pre_el']:.3f} lip_z@0.6s={r['lip_z_mean']*100:+.2f}cm(std {r['lip_z_std']*1000:.2f}mm) v={r['v']:.2f} t_lift={r['t_lift']:.2f} rate={r['rate']:.0f} d_sh={r['d_sh']:.1f} d_el={r['d_el']:.2f} t_brake={r['t_brake']:.1f}: "
                  f"success {r['success']*100:5.1f}%  final_clr {r['final_clr']:.3f}  pop_fwd {r['pop_fwd']:.3f}  pop_robot_v {r['pop_robot_v']:.2f}")
        print("=" * 70); env.close()
if __name__ == "__main__":
    main()
