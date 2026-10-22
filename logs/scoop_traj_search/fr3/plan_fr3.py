# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Retarget the RoboMaster scoop to a Franka Research 3 holding the stick: FK of the recorded stick motion, IK on FR3.

Pure torch (no simulator). The stick frame is the RoboMaster ``lowerarm`` body frame (stick axis along +x, elbow at
x = 0, lip at STICK_LIP_OFFSET), recorded by ``record_stick_traj.py``.

Desired stick motion, in a frame on the floor at the ball's rest position:

* approach: the FR3 starts at rest and accelerates (quintic) onto the recorded pre-contact motion;
* contact .. contact + ``follow_s``: exactly the recorded motion (same pose and velocity at contact);
* then the recorded motion is played back with a playback rate that ramps smoothly from 1 to 0 (a gentle stop that
  keeps the ball in the pocket), and the stick is held.

Free design choices are searched in batch: where the ball is relative to the FR3 base (distance, bearing, scoop
direction), the height of the FR3 base above the floor, the grip point along the shaft, and the grip roll about the
shaft. The fingers close across the shaft (TCP x along the stick axis). A candidate is feasible if the damped
least-squares IK tracks the TCP within tolerance while staying inside the joint position and velocity limits.

Usage:
  ./isaaclab.sh -p logs/scoop_traj_search/fr3/plan_fr3.py --ref logs/scoop_traj_search/fr3/stick_traj.npz \
      --out logs/scoop_traj_search/fr3/fr3_plan.npz
"""

import argparse
import math

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--ref", required=True)
p.add_argument("--out", required=True)
p.add_argument("--env", type=int, default=0, help="recorded env to retarget")
p.add_argument("--candidates", type=int, default=4096)
p.add_argument("--follow_s", type=float, default=0.4, help="follow the recording exactly this long after contact")
p.add_argument("--stop_s", type=float, default=0.6, help="playback-rate ramp 1 -> 0 duration")
p.add_argument("--pre_s", type=float, default=0.15, help="follow the recording this long before contact")
p.add_argument("--acc_s", type=float, default=0.35, help="acceleration from rest onto the recording")
p.add_argument("--hold_s", type=float, default=0.5)
p.add_argument("--hz", type=float, default=120.0)
p.add_argument(
    "--smooth_s", type=float, default=0.03, help="Gaussian smoothing of the recording (removes the impact shocks)"
)
p.add_argument("--vel_margin", type=float, default=0.8, help="allowed fraction of the joint velocity/acc limits")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--base_h", type=float, default=None, help="fix the FR3 base height above the floor [m]")
p.add_argument(
    "--husky", action="store_true", help="FR3 on a Clearpath Husky: keep arm, stick and ball clear of its body"
)
p.add_argument(
    "--mount_x", type=float, default=None, help="FR3 base position along the Husky (+ = front) [m]; searched if unset"
)
p.add_argument("--clearance", type=float, default=0.02, help="minimum distance to the Husky body [m]")
p.add_argument("--refine_rounds", type=int, default=2, help="local refinement rounds around the best candidates")
p.add_argument(
    "--start_lift", type=float, default=0.07,
    help="also save a start pose with the hand this much higher (a compliant controller sets the head down from it)",
)
a = p.parse_args()
torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- FR3 kinematics (modified DH, identical to Panda) ----------------
DH = torch.tensor(  # a_{i-1}, d_i, alpha_{i-1}
    [[0.0, 0.333, 0.0], [0.0, 0.0, -math.pi / 2], [0.0, 0.316, math.pi / 2], [0.0825, 0.0, math.pi / 2],
     [-0.0825, 0.384, -math.pi / 2], [0.0, 0.0, math.pi / 2], [0.088, 0.0, math.pi / 2]], device=dev
)  # fmt: skip
Q_LO = torch.tensor([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159], device=dev)
Q_HI = torch.tensor([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159], device=dev)
QD_MAX = torch.tensor([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26], device=dev)
# joint acceleration limits: Panda datasheet values (assumed similar for FR3; check before running on hardware)
QDD_MAX = torch.tensor([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0], device=dev)


def _tf(a_, d_, al, th):
    """Modified-DH link transform Rx(alpha) Tx(a) Rz(theta) Tz(d), batched over theta."""
    ca, sa, ct, st = math.cos(al), math.sin(al), torch.cos(th), torch.sin(th)
    T = torch.zeros(*th.shape, 4, 4, device=th.device)
    T[..., 0, 0], T[..., 0, 1], T[..., 0, 3] = ct, -st, a_
    T[..., 1, 0], T[..., 1, 1], T[..., 1, 2], T[..., 1, 3] = st * ca, ct * ca, -sa, -sa * d_
    T[..., 2, 0], T[..., 2, 1], T[..., 2, 2], T[..., 2, 3] = st * sa, ct * sa, ca, ca * d_
    T[..., 3, 3] = 1.0
    return T


FLANGE_TO_TCP = torch.eye(4, device=dev)
_c, _s = math.cos(-math.pi / 4), math.sin(-math.pi / 4)
FLANGE_TO_TCP[:3, :3] = torch.tensor([[_c, -_s, 0], [_s, _c, 0], [0, 0, 1.0]], device=dev)
FLANGE_TO_TCP[2, 3] = 0.107 + 0.1034


def fk_tcp(q):
    """TCP pose (..., 4, 4) in the FR3 base frame."""
    T = torch.eye(4, device=q.device).expand(*q.shape[:-1], 4, 4)
    for i in range(7):
        T = T @ _tf(float(DH[i, 0]), float(DH[i, 1]), float(DH[i, 2]), q[..., i])
    return T @ FLANGE_TO_TCP


def fk_points(q):
    """Origins of joint frames 2..7 and the flange, (..., 7, 3), for clearance checks."""
    T = torch.eye(4, device=q.device).expand(*q.shape[:-1], 4, 4)
    pts = []
    for i in range(7):
        T = T @ _tf(float(DH[i, 0]), float(DH[i, 1]), float(DH[i, 2]), q[..., i])
        if i >= 1:
            pts.append(T[..., :3, 3])
    pts.append((T @ FLANGE_TO_TCP)[..., :3, 3])
    return torch.stack(pts, -2)


def pose_err(T, Td):
    """6-D error (position, rotation vector) taking T to Td."""
    ep = Td[..., :3, 3] - T[..., :3, 3]
    R = Td[..., :3, :3] @ T[..., :3, :3].transpose(-1, -2)
    er = 0.5 * torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    return torch.cat([ep, er], -1)


def jacobian(q, eps=1e-4):
    """Numerical geometric-like Jacobian of pose_err w.r.t. q, (..., 6, 7)."""
    T0 = fk_tcp(q)
    J = []
    for i in range(7):
        dq = torch.zeros_like(q)
        dq[..., i] = eps
        J.append(-pose_err(fk_tcp(q + dq), T0) / eps)
    return torch.stack(J, -1)


def ik_step(q, Td, lam=0.02, k_null=0.05, iters=3):
    q_mid = 0.5 * (Q_LO + Q_HI)
    for _ in range(iters):
        e = pose_err(fk_tcp(q), Td)
        J = jacobian(q)
        JJt = J @ J.transpose(-1, -2) + lam**2 * torch.eye(6, device=q.device)
        dq = (J.transpose(-1, -2) @ torch.linalg.solve(JJt, e[..., None]))[..., 0]
        # null-space pull towards the middle of the joint ranges
        N = torch.eye(7, device=q.device) - J.transpose(-1, -2) @ torch.linalg.solve(JJt, J)
        dq = dq + (N @ (k_null * (q_mid - q))[..., None])[..., 0]
        q = torch.clamp(q + dq, Q_LO + 1e-3, Q_HI - 1e-3)
    return q


# ---------------- reference stick motion ----------------
def quat_xyzw_to_R(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
         2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
         2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1
    ).reshape(*q.shape[:-1], 3, 3)  # fmt: skip


ref = np.load(a.ref)
dt_ref = float(ref["physics_dt"])
e = a.env
k_c = int(np.argmax(ref["contact"][e]))
t_ref = np.arange(ref["stick_pos"].shape[1]) * dt_ref
ball0 = ref["ball"][e, 0]  # rest position before the robot arrives
floor_ball = np.array([ball0[0], ball0[1], 0.0])
# The soft MIT-mode arm is jolted by the impact (~40 m/s^2) and by the stepped lift command; those are reactions, not
# motion to imitate, so the recording is low-pass filtered (positions, and rotations as rotation vectors relative to
# the contact pose).
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

sig = a.smooth_s / dt_ref
pos_np = gaussian_filter1d(ref["stick_pos"][e] - floor_ball, sig, axis=0, mode="nearest")
rot = Rotation.from_quat(ref["stick_quat"][e])  # scipy uses (x, y, z, w)
rot_c = rot[k_c]
rv = gaussian_filter1d((rot_c.inv() * rot).as_rotvec(), sig, axis=0, mode="nearest")
R_np = (rot_c * Rotation.from_rotvec(rv)).as_matrix()
pos_ref = torch.tensor(pos_np, dtype=torch.float32, device=dev)  # stick frame in the ball frame
R_ref = torch.tensor(R_np, dtype=torch.float32, device=dev)
t_c = t_ref[k_c]


def ref_pose(t):
    """Recorded stick pose at (fractional) recording time t [s], linear / nlerp-free (small steps)."""
    x = np.clip(t / dt_ref, 0, len(t_ref) - 1.001)
    i = np.floor(x).astype(int)
    f = torch.tensor(x - i, dtype=torch.float32, device=dev)[:, None]
    pos = pos_ref[i] * (1 - f) + pos_ref[i + 1] * f
    R = R_ref[i] * (1 - f[..., None]) + R_ref[i + 1] * f[..., None]
    U, _, Vt = torch.linalg.svd(R)  # re-orthonormalize the blend
    R = U @ Vt
    T = torch.eye(4, device=dev).repeat(len(t), 1, 1)
    T[:, :3, :3], T[:, :3, 3] = R, pos
    return T


# playback time phi(t) of the recording as a function of plan time t (t = 0 at contact)
dt = 1.0 / a.hz
t_plan = np.arange(-(a.acc_s + a.pre_s), a.follow_s + a.stop_s + a.hold_s + 1e-9, dt)
phi = np.empty_like(t_plan)
for n, t in enumerate(t_plan):
    if t <= a.follow_s:
        phi[n] = t_c + t
    else:
        s = min((t - a.follow_s) / a.stop_s, 1.0)
        # rate 1 - (10 s^3 - 15 s^4 + 6 s^5), integrated
        phi[n] = t_c + a.follow_s + a.stop_s * (s - (2.5 * s**4 - 3 * s**5 + s**6))
T_stick = ref_pose(np.clip(phi, 0.0, None))
# approach: from rest, quintic onto the recording at t = -pre_s (matching position and velocity, zero acceleration)
n0 = int(round(a.acc_s / dt))
p1 = T_stick[n0, :3, 3]
v1 = (T_stick[n0 + 1, :3, 3] - T_stick[n0 - 1, :3, 3]) / (2 * dt)
p0 = p1 - v1 * a.acc_s * 0.5  # constant-jerk-ish ramp covers about half the cruise distance
for n in range(n0):
    s = n * dt / a.acc_s
    # quintic with p(0)=p0, v(0)=0, a(0)=0, p(1)=p1, v(1)=v1*T, a(1)=0
    h = 10 * s**3 - 15 * s**4 + 6 * s**5
    hv = -4 * s**3 + 7 * s**4 - 3 * s**5  # basis for the end velocity
    T_stick[n, :3, 3] = p0 + (p1 - p0) * h + v1 * a.acc_s * hv
    T_stick[n, :3, :3] = T_stick[n0, :3, :3]

# ---------------- candidate placements and grips ----------------
C = a.candidates
rng = torch.Generator(device="cpu").manual_seed(a.seed)


def U(lo, hi):
    return (lo + (hi - lo) * torch.rand(C, generator=rng)).to(dev)


cand = dict(
    dist=U(0.35, 0.75),  # [m] horizontal distance from the FR3 base axis to the ball
    bearing=U(-math.pi, math.pi),  # [rad] direction of the ball seen from the base
    heading=U(-math.pi, math.pi),  # [rad] scoop direction in the base frame
    base_h=U(0.0, 0.5) if a.base_h is None else torch.full((C,), a.base_h, device=dev),  # [m] base above the floor
    mount_x=(U(-0.3, 0.3) if a.mount_x is None else torch.full((C,), a.mount_x, device=dev)),  # [m] on the Husky
    grip_x=U(-0.45, 0.15),  # [m] grip point on the shaft (stick frame x; elbow at 0, butt at -0.504)
    roll=U(-math.pi, math.pi),  # [rad] hand position around the shaft
    flip=(torch.rand(C, generator=rng) < 0.5).float().to(dev) * 2 - 1,  # TCP x along +/- stick axis
)


def ball_frame_in_base(c):
    C = len(c["dist"])
    T = torch.eye(4, device=dev).repeat(C, 1, 1)
    ch, sh = torch.cos(c["heading"]), torch.sin(c["heading"])
    T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1] = ch, -sh, sh, ch
    T[:, 0, 3] = c["dist"] * torch.cos(c["bearing"])
    T[:, 1, 3] = c["dist"] * torch.sin(c["bearing"])
    T[:, 2, 3] = -c["base_h"]
    return T


def tcp_in_stick(c):
    """Grip: TCP at the shaft point (grip_x, 0.016, 0.48), fingers closing across the shaft."""
    C = len(c["dist"])
    T = torch.eye(4, device=dev).repeat(C, 1, 1)
    x = torch.stack([c["flip"], torch.zeros(C, device=dev), torch.zeros(C, device=dev)], -1)
    n = torch.stack([torch.zeros(C, device=dev), torch.sin(c["roll"]), torch.cos(c["roll"])], -1)  # shaft -> hand
    z = -n
    y = torch.cross(z, x, dim=-1)
    T[:, :3, 0], T[:, :3, 1], T[:, :3, 2] = x, y, z
    T[:, :3, 3] = torch.stack([c["grip_x"], torch.full((C,), 0.016, device=dev), torch.full((C,), 0.48, device=dev)], -1)
    return T


def evaluate(cand):
    """IK along the trajectory for every candidate; returns the joint trajectories and feasibility metrics."""
    C = len(cand["dist"])
    Tb = ball_frame_in_base(cand)
    Tg = tcp_in_stick(cand)
    Tn = len(t_plan)
    T_tcp = Tb[:, None] @ T_stick[None] @ Tg[:, None]  # TCP targets (C, Tn, 4, 4)
    # initial solution: several random seeds per candidate, keep the best
    seeds = 8
    qs = Q_LO + (Q_HI - Q_LO) * torch.rand(C, seeds, 7, device=dev)
    Td0 = T_tcp[:, 0][:, None].expand(-1, seeds, -1, -1)
    for _ in range(40):
        qs = ik_step(qs, Td0, iters=1, k_null=0.02)
    err0 = pose_err(fk_tcp(qs), Td0)
    score0 = err0[..., :3].norm(dim=-1) + 0.1 * err0[..., 3:].norm(dim=-1)
    q = qs[torch.arange(C), score0.argmin(-1)]
    Q = torch.empty(C, Tn, 7, device=dev)
    pos_err = torch.empty(C, Tn, device=dev)
    rot_err = torch.empty(C, Tn, device=dev)
    for n in range(Tn):
        q = ik_step(q, T_tcp[:, n], iters=3 if n else 10)
        Q[:, n] = q
        er = pose_err(fk_tcp(q), T_tcp[:, n])
        pos_err[:, n], rot_err[:, n] = er[:, :3].norm(dim=-1), er[:, 3:].norm(dim=-1)
    qd = torch.gradient(Q, spacing=dt, dim=1)[0]
    qdd = torch.gradient(qd, spacing=dt, dim=1)[0]
    vel_ratio = (qd.abs() / QD_MAX).amax(dim=(1, 2))
    acc_ratio = (qdd.abs() / QDD_MAX).amax(dim=(1, 2))
    lim_margin = torch.minimum(Q - Q_LO, Q_HI - Q).amin(dim=(1, 2))
    ratio = torch.nan_to_num(torch.maximum(vel_ratio, acc_ratio), nan=float("inf"))
    pos_err, rot_err = torch.nan_to_num(pos_err, nan=float("inf")), torch.nan_to_num(rot_err, nan=float("inf"))
    # Husky body (Clearpath husky_description: 0.99 x 0.57 m chassis, wheels to +-0.39 m, bumpers to +-0.55 m) as a
    # box below the FR3 base plane, in the FR3 base frame (FR3 on the center line at mount_x, facing forward). The arm
    # links and the stick are capsules (sampled spheres); every sphere must stay `clearance` away from the box. The
    # FR3 base and link 1 stand on the top plate and are not checked.
    clear = torch.ones(C, dtype=torch.bool, device=dev)
    min_dist = torch.full((C,), float("inf"), device=dev)
    if a.husky:
        box_c = torch.stack([-cand["mount_x"], torch.zeros(C, device=dev), -0.5 * cand["base_h"]], -1)
        box_h = torch.stack([torch.full((C,), 0.55, device=dev), torch.full((C,), 0.39, device=dev), 0.5 * cand["base_h"]], -1)
        # arm: spheres along the segments between the joint frames 2..7, the flange and the TCP
        Tq = fk_tcp(Q)[..., :3, 3]
        chain = torch.cat([fk_points(Q), Tq[..., None, :]], -2)  # (C, Tn, 8, 3)
        s = torch.linspace(0, 1, 6, device=dev)[:, None]
        seg = chain[..., :-1, None, :] * (1 - s) + chain[..., 1:, None, :] * s  # (C, Tn, 7, 6, 3)
        arm_pts = seg.flatten(2, 3)
        arm_r = torch.tensor([0.07] * 5 * 6 + [0.06] * 2 * 6, device=dev)  # links, then flange/hand
        # stick: shaft (r 1.5 cm) from the butt to the head, head (r 6 cm) from its throat to the lip
        sx = torch.linspace(-0.50, 0.28, 14, device=dev)
        hx = torch.linspace(0.30, 0.60, 5, device=dev)
        stick_pts = torch.cat([torch.stack([sx, torch.full_like(sx, 0.016), torch.full_like(sx, 0.47)], -1),
                               torch.stack([hx, torch.full_like(hx, 0.016), torch.full_like(hx, 0.46)], -1)])
        stick_r = torch.tensor([0.015] * 14 + [0.06] * 5, device=dev)
        sp_h = torch.cat([stick_pts, torch.ones(len(stick_pts), 1, device=dev)], -1)
        pts_stick = (Tb[:, None] @ T_stick[None] @ sp_h.T)[..., :3, :].transpose(-1, -2)  # (C, Tn, P, 3)
        pts = torch.cat([arm_pts, pts_stick], 2)
        rad = torch.cat([arm_r, stick_r])
        d = (pts - box_c[:, None, None]).abs() - box_h[:, None, None]
        dist = torch.linalg.norm(d.clamp(min=0.0), dim=-1) + d.amax(-1).clamp(max=0.0)  # signed distance to the box
        min_dist = (dist - rad).amin(dim=(1, 2))
        clear = min_dist > a.clearance
        bxy = Tb[:, :2, 3]  # the ball must not start under or against the robot
        clear &= ((bxy - box_c[:, :2]).abs() - box_h[:, :2] > 0.05).any(-1)  # fmt: skip
    track_ok = (pos_err.amax(1) < 0.002) & (rot_err.amax(1) < 0.01) & (lim_margin > 0.05) & clear
    ok = track_ok & (ratio < a.vel_margin)
    score = torch.where(ok, ratio, torch.where(track_ok, 10.0 + ratio, 100.0 + pos_err.amax(1)))
    return dict(
        Q=Q, qd=qd, qdd=qdd, Tb=Tb, Tg=Tg, vel_ratio=vel_ratio, acc_ratio=acc_ratio, lim_margin=lim_margin,
        pos_err=pos_err, rot_err=rot_err, clear=clear, ok=ok, score=score, min_dist=min_dist,
    )  # fmt: skip


def perturb(cand, res, k=64):
    """New candidates around the k best ones (same total count)."""
    C = len(cand["dist"])
    top = res["score"].argsort()[:k]
    idx = top[torch.arange(C, device=dev) % k]
    sig = dict(dist=0.03, bearing=0.08, heading=0.08, base_h=0.03, mount_x=0.04, grip_x=0.04, roll=0.12)
    new = {}
    for key, v in cand.items():
        x = v[idx].clone()
        if key in sig and not (key == "base_h" and a.base_h is not None) and not (key == "mount_x" and a.mount_x is not None):
            x = x + sig[key] * torch.randn(C, device=dev) * (torch.arange(C, device=dev) >= k)  # keep the parents
        new[key] = x
    new["dist"] = new["dist"].clamp(0.3, 0.85)
    new["base_h"] = new["base_h"].clamp(0.0, 0.6)
    new["mount_x"] = new["mount_x"].clamp(-0.35, 0.35)
    new["grip_x"] = new["grip_x"].clamp(-0.48, 0.2)
    return new


res = evaluate(cand)
print(f"PLAN round 0: feasible {int(res['ok'].sum())}/{C}, husky clearance ok {int(res['clear'].sum())}")
for r in range(a.refine_rounds):
    cand = perturb(cand, res)
    res = evaluate(cand)
    best_r = res["score"].min().item()
    print(f"PLAN refine {r + 1}: feasible {int(res['ok'].sum())}/{C}, best score {best_r:.3f}")
Q, qd, qdd, Tb, Tg = res["Q"], res["qd"], res["qdd"], res["Tb"], res["Tg"]
vel_ratio, acc_ratio, lim_margin = res["vel_ratio"], res["acc_ratio"], res["lim_margin"]
pos_err, rot_err, ok, score = res["pos_err"], res["rot_err"], res["ok"], res["score"]
best = int(score.argmin())
print(f"PLAN feasible {int(ok.sum())}/{C} candidates")
for i in score.argsort()[:5].tolist():
    print(
        f"PLAN cand {i}: max|qd|/limit {vel_ratio[i]:.2f}  max|qdd|/limit {acc_ratio[i]:.2f}  pos err"
        f" {1000 * pos_err[i].max():.2f} mm  rot err {rot_err[i].max():.4f} rad  limit margin {lim_margin[i]:.2f} rad  "
        f"husky clearance {100 * res['min_dist'][i]:.1f} cm  "
        + "  ".join(f"{k}={float(v[i]):.2f}" for k, v in cand.items())
    )
# start pose with the stick lifted: the real head may sit lower than the model, so a compliant controller starts above
# the floor and lets the head settle on it during the approach
T_start = Tb[best] @ T_stick[0] @ Tg[best]
T_start[2, 3] += a.start_lift
q_start = Q[best, 0][None].clone()
for _ in range(60):
    q_start = ik_step(q_start, T_start[None], iters=1)
err_start = pose_err(fk_tcp(q_start), T_start[None])[0, :3].norm().item()
print(f"PLAN lifted start pose (+{100 * a.start_lift:.0f} cm): IK error {1000 * err_start:.2f} mm")
np.savez(
    a.out,
    q_start_lifted=q_start[0].cpu().numpy(),
    t=t_plan, phi=phi, dt=dt, q=Q[best].cpu().numpy(), qd=qd[best].cpu().numpy(),
    T_stick_ball=T_stick.cpu().numpy(), T_ball_in_base=Tb[best].cpu().numpy(), T_tcp_in_stick=Tg[best].cpu().numpy(),
    feasible=bool(ok[best]), vel_ratio=float(vel_ratio[best]), acc_ratio=float(acc_ratio[best]), **{k: float(v[best]) for k, v in cand.items()},
)  # fmt: skip
print(f"PLAN saved {a.out} (candidate {best}, feasible {bool(ok[best])})")
