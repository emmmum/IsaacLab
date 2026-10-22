# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute the retargeted scoop on a Franka Research 3 in simulation (physics, ball, floor).

The FR3 (``fr3_lacrosse.usd``: stick welded to the hand at the planned grip) tracks the planned joint trajectory from
``plan_fr3.py`` with stiff joint PD + velocity feed-forward (gravity off on the arm, as with the FR3's own gravity
compensation). Each env gets the ball at the planned spot plus a random offset; success = at the end the ball is more
than 15 cm above the floor and within 25 cm of the pocket reference point.

Usage:
  ./isaaclab.sh -p logs/scoop_traj_search/fr3/fr3_scoop_sim.py --plan logs/scoop_traj_search/fr3/fr3_plan.npz \
      [--num_envs 64 --ball_offset 0.01 --video out.mp4]
"""

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--plan", required=True)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--ball_offset", type=float, default=0.01, help="uniform +- ball offset in the scoop frame [m]")
p.add_argument("--video", default="", help="mp4 of env 0 (needs --enable_cameras)")
p.add_argument("--video_every", type=int, default=6, help="physics steps per frame (6 = real time at 60 fps)")
p.add_argument("--out", default="", help="save per-step signals of env 0 to this .npz")
p.add_argument("--usd", default="fr3_lacrosse.usd", help="FR3 + stick asset (in the robomaster_lacrosse data dir)")
p.add_argument("--husky", action="store_true", help="put the FR3 on a Clearpath Husky (mount_x from the plan)")
p.add_argument("--husky_box", type=float, nargs=2, default=(1.10, 0.78), help="Husky collision box length, width [m]")
p.add_argument(
    "--ctrl", choices=("pd", "impedance"), default="pd",
    help="pd: stiff joint PD on the planned joint trajectory; impedance: compliant task-space control of the lip",
)
p.add_argument(
    "--press_depth", type=float, default=0.10,
    help="impedance: before the lift the lip is pulled towards this far below the modelled floor height [m]",
)
p.add_argument("--press_k", type=float, default=100.0, help="impedance: press spring stiffness [N/m]")
p.add_argument("--press_until", type=float, default=0.05, help="impedance: press phase ends this long after contact [s]")
p.add_argument("--ramp", type=float, default=0.1, help="impedance: z / pitch stiffness ramp-up time after the press [s]")
p.add_argument("--bandwidth", type=float, default=10.0, help="impedance: bandwidth of the stiff axes [Hz]")
p.add_argument(
    "--pitch_frac", type=float, default=1.0,
    help="impedance: pitch stiffness in the press phase as a fraction of the stiff axes (0 = the head tilts freely)",
)
p.add_argument("--k_null", type=float, default=400.0, help="impedance: null-space joint stiffness [1/s^2]")
p.add_argument("--cruise_max", type=float, default=0.2, help="impedance: max wait for a late contact [s]")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import matrix_from_quat, quat_apply  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
USD = os.path.join(REPO, "source/isaaclab_assets/data/robomaster_lacrosse", a.usd)
HUSKY_USD = os.path.join(REPO, "source/isaaclab_assets/data/husky/usd/husky/husky.usda")
HUSKY_BASE_Z = 0.1323  # [m] base_link above the floor (Clearpath husky_description); top plate at 0.3773 m
# Husky collision model: a box (wheels and bumpers included) from the floor up to the FR3 base plane, the same box the
# planner keeps clear of. The converted Husky USD is visual only (its physics variant is not selected).
HUSKY_BOX = tuple(a.husky_box)  # [m] length, width (default: wheels and bumpers included)
BALL_R, BALL_M = 0.0325, 0.141
POCKET = (0.46, 0.016, 0.46)  # [m] pocket in the stick frame (STICK_POCKET_OFFSET)
LIP = (0.640, 0.016, 0.452)  # [m] lip in the stick frame (STICK_LIP_OFFSET)
ROLL_RES, ROLL_W0 = 0.015, 0.05  # rolling resistance, as in the RoboMaster env

plan = np.load(a.plan)
q_plan, qd_plan, t_plan, dt_plan = plan["q"], plan["qd"], plan["t"], float(plan["dt"])
base_h = float(plan["base_h"])
T_ball = plan["T_ball_in_base"]  # ball (floor) frame in the FR3 base frame
# impedance: start with the stick lifted (the planner's IK) and let the head settle on the floor during the approach;
# the real head may sit lower than the model, which a start on the planned floor contact would push into the floor
q0 = plan["q_start_lifted"] if (a.ctrl == "impedance" and "q_start_lifted" in plan) else q_plan[0]
JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    # FR3 on a Husky (top plate at the FR3 base, FR3 mount_x ahead of the Husky center), or on a plain stand
    husky = (
        AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Husky",
            spawn=sim_utils.UsdFileCfg(usd_path=HUSKY_USD),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-float(plan["mount_x"]), 0.0, HUSKY_BASE_Z)),
        )
        if a.husky
        else None
    )
    husky_box = (
        RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/HuskyBox",
            spawn=sim_utils.CuboidCfg(
                size=(*HUSKY_BOX, base_h - 0.002),  # just below the FR3 base so the mount itself is not a contact
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                activate_contact_sensors=True,
                visible=False,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-float(plan["mount_x"]), 0.0, 0.5 * (base_h - 0.002))),
        )
        if a.husky
        else None
    )
    # net contact force on the Husky box: it is kinematic, so the floor never pushes on it; any force comes from the
    # arm, the stick or the ball touching the Husky (the box top is 2 mm below the FR3 base, so the mount is no contact)
    husky_contact = (
        ContactSensorCfg(prim_path="{ENV_REGEX_NS}/HuskyBox")
        if a.husky
        else None
    )
    stand = None if a.husky else AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, max(base_h, 0.01)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5 * base_h)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, base_h),
            joint_pos={**{j: float(v) for j, v in zip(JOINTS, q0)}, "fr3_finger_joint.*": 0.04},
        ),
        actuators={
            # pd: stiff position control; impedance: torque control (the controller below computes the efforts)
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint.*"],
                stiffness=4000.0 if a.ctrl == "pd" else 0.0,
                damping=150.0 if a.ctrl == "pd" else 0.0,
                effort_limit_sim=200.0 if a.ctrl == "pd" else {"fr3_joint[1-4]": 87.0, "fr3_joint[5-7]": 12.0},
                velocity_limit_sim=6.0,
            ),
            "fingers": ImplicitActuatorCfg(joint_names_expr=["fr3_finger_joint.*"], stiffness=2000.0, damping=100.0),
        },
    )  # fmt: skip
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_R,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_M),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8, restitution=0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, BALL_R)),
    )


class ImpedanceController:
    """Compliant, contact-triggered task-space control of the lip, built from the planner's stick model only.

    As on the real robot, the controller does not know the real stick: the control point is the lip as the model
    predicts it from the hand pose (hand -> stick from the plan's grip, lip offset in the stick frame). Each env keeps
    its own plan time ``tau`` (0 = planned contact):

    * approach: ``tau`` runs with the clock; if no contact has been detected when ``tau`` reaches 0, the lip keeps
      cruising at the planned contact velocity (up to ``cruise_max`` s);
    * contact (detected): ``tau`` jumps / resumes at 0 and the lip position measured there becomes the reference of the
      post-contact motion, so a stick that reaches the ball earlier or later than the model does is handled;
    * press (until ``press_until`` after contact): no task stiffness along z (and ``pitch_frac`` of it about the pitch
      axis); a soft spring (``press_k``) pulls the lip towards ``press_depth`` below the modelled floor, so the head
      settles on and presses on the real floor; the lip height / pitch reached there are taken as the reference and
      the z / pitch stiffness ramps up over ``ramp`` for the lift;
    * x, y, roll and yaw stiff (``bandwidth``) with the planned velocity / acceleration as feed-forward; null space:
      joint PD towards the planned configuration.

    Contact detection: in simulation the ball's horizontal speed (> 5 cm/s) stands in for an impact detector; on the
    FR3 the same role is played by its external-wrench estimate from the joint torque sensors.
    """

    def __init__(self, robot, ball, arm_ids, hand_id, origins, dev, n):
        self.robot, self.ball, self.arm_ids, self.hand_id, self.dev, self.n = robot, ball, arm_ids, hand_id, dev, n
        T_hand_tcp = np.eye(4)
        T_hand_tcp[2, 3] = 0.1034
        T_hand_stick = T_hand_tcp @ np.linalg.inv(plan["T_tcp_in_stick"])
        lip_h = T_hand_stick @ np.array([*LIP, 1.0])
        self.lip_in_hand = torch.tensor(lip_h[:3], dtype=torch.float32, device=dev)
        self.R_hand_stick = torch.tensor(T_hand_stick[:3, :3], dtype=torch.float32, device=dev)
        T_w = np.einsum("ij,tjk->tik", T_ball, plan["T_stick_ball"])
        lip = np.einsum("tij,j->ti", T_w, np.array([*LIP, 1.0]))[:, :3] + [0.0, 0.0, base_h]
        v = np.gradient(lip, dt_plan, axis=0)
        acc = np.gradient(v, dt_plan, axis=0)
        R = T_w[:, :3, :3]
        dR = np.gradient(R, dt_plan, axis=0)
        W = np.einsum("tij,tkj->tik", dR, R)  # dR R^T = skew(omega)
        w = np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], -1)
        f = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)  # noqa: E731
        self.lip_d, self.v_d, self.a_d, self.R_d, self.w_d = f(lip), f(v), f(acc), f(R), f(w)
        self.o = origins
        self.S = f(T_ball[:3, :3])  # scoop frame axes in the world
        self.i0 = int(round(-t_plan[0] / dt_plan))  # plan index of the planned contact
        self.v_contact = self.v_d[self.i0]
        wn = 2 * np.pi * a.bandwidth
        self.kp_stiff, self.kd_stiff = wn**2, 2.0 * wn
        self.k_null, self.d_null = a.k_null, 2.0 * np.sqrt(a.k_null)
        z = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.tau = torch.full((n,), float(t_plan[0]), device=dev)
        self.state = torch.zeros(n, dtype=torch.long, device=dev)  # 0 approach, 1 cruise, 2 after contact
        self.cruise = z()  # time spent cruising past the planned contact [s]
        self.off_contact = torch.zeros(n, 3, device=dev)  # lip offset (scoop frame) captured at contact
        self.off_press = torch.zeros(n, 2, device=dev)  # (z, pitch) offset captured at the end of the press
        self.press_done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.t_contact = torch.full((n,), float("nan"), device=dev)

    def _at(self, arr, tau):
        x = ((tau - t_plan[0]) / dt_plan).clamp(0.0, len(t_plan) - 1.001)
        i = x.long()
        fr = (x - i)[:, *([None] * (arr.dim() - 1))]
        return arr[i] * (1 - fr) + arr[i + 1] * fr

    def compute(self, t, dt, q_ref_fn):
        r, ids, n, S = self.robot, self.arm_ids, self.n, self.S
        d = r.data
        q, qd = d.joint_pos.torch[:, ids], d.joint_vel.torch[:, ids]
        J = d.body_link_jacobian_w.torch[:, self.hand_id - 1][:, :, ids]  # fixed base: no root row
        M = d.mass_matrix.torch[:, ids][:, :, ids]
        hp, hq = d.body_pos_w.torch[:, self.hand_id], d.body_quat_w.torch[:, self.hand_id]
        R_h = matrix_from_quat(hq)
        rvec = R_h @ self.lip_in_hand  # hand origin -> modelled lip, world
        p = hp + rvec - self.o
        Jv = J[:, :3] - torch.cross(rvec[:, :, None].expand(-1, -1, 7), J[:, 3:], dim=1)
        Jp = torch.cat([Jv, J[:, 3:]], 1)
        vel = (Jp @ qd[..., None])[..., 0]
        R_s = R_h @ self.R_hand_stick  # modelled stick orientation

        # --- per-env plan time and contact events
        bv = self.ball.data.root_lin_vel_w.torch[:, :2]
        hit = (torch.linalg.norm(bv, dim=-1) > 0.05) & (self.state < 2)
        too_late = (self.state == 1) & (self.cruise > a.cruise_max)
        start = hit | too_late
        self.t_contact = torch.where(start, torch.full_like(self.t_contact, t), self.t_contact)
        self.state = torch.where(start, torch.full_like(self.state, 2), self.state)
        tau_new = torch.where(start, torch.zeros_like(self.tau), self.tau)
        # desired lip at the (possibly re-timed) plan time, plus the cruise distance before contact
        lip_d = self._at(self.lip_d, tau_new) + (self.cruise[:, None] * self.v_contact[None])
        e_pos_s = (lip_d - p) @ S
        R_d = self._at(self.R_d, tau_new)
        U_, _, Vt = torch.linalg.svd(R_d)
        R_d = U_ @ Vt
        Rerr = R_d @ R_s.transpose(1, 2)
        e_rot = 0.5 * torch.stack([Rerr[:, 2, 1] - Rerr[:, 1, 2], Rerr[:, 0, 2] - Rerr[:, 2, 0], Rerr[:, 1, 0] - Rerr[:, 0, 1]], -1)
        e_rot_s = e_rot @ S
        # contact: the measured lip position becomes the reference (x, y now; z / pitch after the press)
        self.off_contact = torch.where(start[:, None], -e_pos_s, self.off_contact)
        self.off_contact[:, 2] = torch.where(start, torch.zeros_like(e_pos_s[:, 2]), self.off_contact[:, 2])
        after = self.state == 2
        press = ~after | (tau_new < a.press_until)
        end_press = after & ~press & ~self.press_done
        self.off_press = torch.where(end_press[:, None], torch.stack([-e_pos_s[:, 2], -e_rot_s[:, 1]], -1), self.off_press)
        self.press_done |= end_press
        e_pos_s = e_pos_s + torch.where(after[:, None], self.off_contact, torch.zeros_like(self.off_contact))
        e_pos_s[:, 2] = e_pos_s[:, 2] + torch.where(self.press_done, self.off_press[:, 0], torch.zeros_like(e_pos_s[:, 2]))
        e_rot_s = e_rot_s.clone()
        e_rot_s[:, 1] = e_rot_s[:, 1] + torch.where(self.press_done, self.off_press[:, 1], torch.zeros_like(e_rot_s[:, 1]))
        # velocity / acceleration feed-forward (cruising keeps the contact velocity)
        v_d = torch.where((self.state == 1)[:, None], self.v_contact[None].expand(n, -1), self._at(self.v_d, tau_new))
        a_d = torch.where((self.state == 1)[:, None], torch.zeros(n, 3, device=self.dev), self._at(self.a_d, tau_new))
        w_d = self._at(self.w_d, tau_new)
        ev_s = (v_d - vel[:, :3]) @ S
        ew_s = (w_d - vel[:, 3:]) @ S
        s_ramp = torch.where(after & ~press, ((tau_new - a.press_until) / a.ramp).clamp(0.0, 1.0), torch.zeros_like(tau_new))
        kp = torch.full((n, 6), self.kp_stiff, device=self.dev)
        kd = torch.full((n, 6), self.kd_stiff, device=self.dev)
        kp[:, 2] = s_ramp * self.kp_stiff
        kd[:, 2] = s_ramp * self.kd_stiff
        kp[:, 4] = (a.pitch_frac + s_ramp * (1.0 - a.pitch_frac)) * self.kp_stiff
        kd[:, 4] = (np.sqrt(a.pitch_frac) + s_ramp * (1.0 - np.sqrt(a.pitch_frac))) * self.kd_stiff
        acc_s = torch.cat([kp[:, :3] * e_pos_s + kd[:, :3] * ev_s, kp[:, 3:] * e_rot_s + kd[:, 3:] * ew_s], -1)
        acc_ff = a_d @ S
        acc_ff[:, 2] = acc_ff[:, 2] * s_ramp  # before the lift the floor decides the height
        acc_s[:, :3] += acc_ff
        acc_w = torch.cat([acc_s[:, :3] @ S.T, acc_s[:, 3:] @ S.T], -1)
        Minv = torch.linalg.inv(M)
        Lam = torch.linalg.inv(Jp @ Minv @ Jp.transpose(1, 2) + 1e-4 * torch.eye(6, device=self.dev))
        F = (Lam @ acc_w[..., None])[..., 0]
        # press spring along world z (= scoop-frame z), in force units, fading out over the ramp
        c_press = 2.0 * np.sqrt(a.press_k * 1.5)  # ~critical for ~1.5 kg effective mass
        e_z_model = (lip_d - p)[:, 2]
        f_press = a.press_k * (e_z_model - a.press_depth) + c_press * (v_d[:, 2] - vel[:, 2])
        F[:, 2] += (1.0 - s_ramp) * f_press
        tau_cmd = (Jp.transpose(1, 2) @ F[..., None])[..., 0]
        Jbar = Minv @ Jp.transpose(1, 2) @ Lam
        N = torch.eye(7, device=self.dev) - Jp.transpose(1, 2) @ Jbar.transpose(1, 2)
        q_ref = q_ref_fn(tau_new)
        tau_null = (M @ (self.k_null * (q_ref - q) - self.d_null * qd)[..., None])[..., 0]
        tau_cmd = tau_cmd + (N @ tau_null[..., None])[..., 0]
        # advance the per-env plan time
        self.cruise = self.cruise + dt * (self.state == 1).float()
        self.tau = tau_new + dt * (self.state != 1).float()
        reach = (self.state == 0) & (self.tau >= 0.0)
        self.state = torch.where(reach, torch.ones_like(self.state), self.state)
        self.tau = torch.where(reach, torch.zeros_like(self.tau), self.tau)
        lim = torch.tensor([87.0] * 4 + [12.0] * 3, device=self.dev)
        return tau_cmd.clamp(-lim, lim)


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1 / 360, render_interval=6, physics=PhysxCfg(gpu_collision_stack_size=2**29))
    )
    scene = InteractiveScene(SceneCfg(num_envs=a.num_envs, env_spacing=3.0))
    sim.reset()
    robot: Articulation = scene["robot"]
    ball: RigidObject = scene["ball"]
    dev, n = sim.device, a.num_envs
    o = scene.env_origins
    arm_ids = robot.find_joints(JOINTS, preserve_order=True)[0]
    stick_id = robot.find_bodies("stick")[0][0]
    hand_id = robot.find_bodies("fr3_hand")[0][0]

    # start the arm at the first planned configuration (fingers open) and hold it there
    jp = robot.data.default_joint_pos.torch.clone()
    robot.write_joint_position_to_sim_index(position=jp)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(jp))
    robot.set_joint_position_target_index(target=jp)
    # ball at the planned spot (+ offset in the scoop frame), at rest on the floor
    Tb = torch.tensor(T_ball, dtype=torch.float32, device=dev)
    off = (torch.rand(n, 2, device=dev) * 2 - 1) * a.ball_offset
    off[0] = 0.0
    off3 = torch.cat([off, torch.zeros(n, 1, device=dev)], -1)
    ball_xy = Tb[:3, 3] + off3 @ Tb[:3, :3].T
    pose = ball.data.default_root_pose.torch.clone()
    pose[:, :3] = o + ball_xy + torch.tensor([0.0, 0.0, base_h + BALL_R], device=dev)
    ball.write_root_pose_to_sim_index(root_pose=pose)
    ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev))
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(sim.get_physics_dt())

    # FK check: hand pose vs the planner's TCP target at t = 0 (TCP = hand + 0.1034 m along hand z)
    hand = robot.data.body_pose_w.torch[0, hand_id]
    tcp = hand[:3] + quat_apply(hand[3:7][None], torch.tensor([[0.0, 0.0, 0.1034]], device=dev))[0] - o[0]
    T_tcp0 = np.asarray(plan["T_ball_in_base"]) @ plan["T_stick_ball"][0] @ plan["T_tcp_in_stick"]
    print(f"SIM FK check: TCP sim {(tcp.cpu().numpy() - [0, 0, base_h]).round(4)} vs plan {T_tcp0[:3, 3].round(4)}")

    phys_dt = sim.get_physics_dt()
    steps = int(round((t_plan[-1] - t_plan[0] + (a.cruise_max if a.ctrl == "impedance" else 0.0)) / phys_dt))
    q_t = torch.tensor(q_plan, dtype=torch.float32, device=dev)
    qd_t = torch.tensor(qd_plan, dtype=torch.float32, device=dev)
    lip_off = torch.tensor([LIP], device=dev).expand(n, -1)
    pocket_off = torch.tensor([POCKET], device=dev).expand(n, -1)
    rec = {k: [] for k in ("t", "q", "q_ref", "lip", "ball")}
    writer = None
    if a.video:
        import imageio

        import omni.replicator.core as rep

        writer = imageio.get_writer(a.video, fps=60, codec="libx264", quality=8, macro_block_size=8)
        # side view of the scoop line: look at the middle between the FR3 base and the ball, from the side
        mid = (o[0] + 0.5 * torch.tensor([float(ball_xy[0, 0]), float(ball_xy[0, 1]), 0.0], device=dev)).tolist()
        head = Tb[:3, 0].tolist()  # scoop direction
        side = [-head[1], head[0]]
        # look from the side of the scoop line that faces away from the robot base / Husky
        if side[0] * (mid[0] - o[0, 0].item()) + side[1] * (mid[1] - o[0, 1].item()) < 0:
            side = [-side[0], -side[1]]
        from pxr import Gf, UsdGeom

        from isaaclab.sim.utils import get_current_stage

        eye = Gf.Vec3d(mid[0] + 2.2 * side[0] - 0.6 * head[0], mid[1] + 2.2 * side[1] - 0.6 * head[1], 1.0)
        cam = UsdGeom.Camera.Define(get_current_stage(), "/World/VideoCam")
        cam.CreateFocalLengthAttr(18.0)
        view = Gf.Matrix4d().SetLookAt(eye, Gf.Vec3d(mid[0], mid[1], 0.35), Gf.Vec3d(0, 0, 1))
        cam.AddTransformOp().Set(view.GetInverse())
        rp = rep.create.render_product("/World/VideoCam", (1280, 720))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([rp])
        for _ in range(10):
            sim.render()

    contact_max = torch.zeros(n, device=dev)
    if a.ctrl == "impedance":
        imp = ImpedanceController(robot, ball, arm_ids, hand_id, o, dev, n)

        def q_ref_at(tau):
            x = ((tau - t_plan[0]) / dt_plan).clamp(0.0, len(t_plan) - 1.001)
            i = x.long()
            fr = (x - i)[:, None]
            return q_t[i] * (1 - fr) + q_t[i + 1] * fr

    for k in range(steps):
        x = min(k * phys_dt / dt_plan, len(t_plan) - 1.001)
        i, f = int(x), x - int(x)
        q_ref = q_t[i] * (1 - f) + q_t[i + 1] * f
        qd_ref = qd_t[i] * (1 - f) + qd_t[i + 1] * f
        if a.ctrl == "pd":
            robot.set_joint_position_target_index(target=q_ref.expand(n, -1), joint_ids=arm_ids)
            robot.set_joint_velocity_target_index(target=qd_ref.expand(n, -1), joint_ids=arm_ids)
        else:
            tau = imp.compute(t_plan[0] + k * phys_dt, phys_dt, q_ref_at)
            robot.set_joint_effort_target_index(target=tau, joint_ids=arm_ids)
        # ball rolling resistance while on the floor
        bp = ball.data.root_pos_w.torch
        on_floor = (bp[:, 2] - o[:, 2] - BALL_R) < 0.002  # the floor is at the env origin height
        w = ball.data.root_ang_vel_w.torch.clone()
        w[:, 2] = 0.0
        wn = torch.linalg.norm(w, dim=-1, keepdim=True)
        tq = -ROLL_RES * BALL_M * 9.81 * BALL_R * torch.tanh(wn / ROLL_W0) * w / wn.clamp(min=1e-9)
        ball.instantaneous_wrench_composer.add_forces_and_torques_index(
            torques=(tq * on_floor[:, None].float())[:, None], body_ids=torch.tensor([0], dtype=torch.int32, device=dev),
            is_global=True,
        )  # fmt: skip
        scene.write_data_to_sim()
        render = writer is not None and k % a.video_every == 0
        sim.step(render=render)
        scene.update(phys_dt)
        if a.husky:
            f_box = scene["husky_contact"].data.net_forces_w.torch  # (n, 1, 3)
            contact_max = torch.maximum(contact_max, torch.linalg.norm(f_box, dim=-1).amax(dim=1))
        sp = robot.data.body_pose_w.torch[:, stick_id]
        if a.out and k % 3 == 0:
            rec["t"].append(t_plan[0] + k * phys_dt)
            rec["q"].append(robot.data.joint_pos.torch[0, arm_ids].cpu().numpy())
            rec["q_ref"].append(q_ref.cpu().numpy())
            rec["lip"].append((sp[0, :3] + quat_apply(sp[:1, 3:7], lip_off[:1])[0] - o[0]).cpu().numpy())
            rec["ball"].append((bp[0] - o[0]).cpu().numpy())
        if render:
            frame = np.asarray(rgb.get_data())
            if frame.size and frame.mean() > 5:
                writer.append_data(frame[..., :3])
    sp = robot.data.body_pose_w.torch[:, stick_id]
    pocket = sp[:, :3] + quat_apply(sp[:, 3:7], pocket_off)
    bp = ball.data.root_pos_w.torch
    height = bp[:, 2] - o[:, 2] - BALL_R
    # the ball rests ~15 cm from the POCKET reference point (it is not the net's lowest point), so allow 25 cm
    near = torch.linalg.norm(bp - pocket, dim=-1) < 0.25
    ok = (height > 0.15) & near
    err = (robot.data.joint_pos.torch[:, arm_ids] - q_t[-1]).abs().max().item()
    print(f"SIM success {int(ok.sum())}/{n} (ball offset +-{100 * a.ball_offset:.0f} cm); env0 ball height "
          f"{height[0]:.3f} m; final joint error {err:.4f} rad")  # fmt: skip
    fail = ~ok
    if fail.any():
        on_floor_end = height < 0.02
        print(f"SIM failures: {int(fail.sum())} (ball on the floor {int((fail & on_floor_end).sum())}, "
              f"up but away from the pocket {int((fail & ~on_floor_end).sum())}); offset of failures "
              f"fwd mean {100 * off[fail, 0].mean():+.2f} cm, lat mean {100 * off[fail, 1].mean():+.2f} cm; "
              f"successes fwd {100 * off[ok, 0].mean():+.2f}, lat {100 * off[ok, 1].mean():+.2f} cm")  # fmt: skip
        dpk = torch.linalg.norm(bp - pocket, dim=-1)
        for e in fail.nonzero().flatten()[:6].tolist():
            print(f"SIM fail env {e}: ball height {height[e]:.3f} m, ball-pocket {dpk[e]:.3f} m, "
                  f"ball xy rel base {(bp[e, :2] - o[e, :2]).tolist()}")
    if a.husky:
        hit = contact_max > 0.1
        print(f"SIM husky contacts: {int(hit.sum())}/{n} envs with any contact on the Husky box > 0.1 N "
              f"(max {contact_max.max():.2f} N)")  # fmt: skip
    if a.out:
        np.savez(a.out, **{k: np.array(v) for k, v in rec.items()})
    if writer is not None:
        writer.close()


main()
app.close()
