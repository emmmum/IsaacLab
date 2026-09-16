"""
export_weights.py — Dump a trained lacrosse_scoop skrl PPO checkpoint to plain JSON
for use in a browser (see policy.js for a matching pure-JS forward pass).

Usage:
    python web/export_weights.py \
        --checkpoint lacrosse_runs/dense_v16/dense_v16/checkpoints/best_agent.pt \
        --out web/lacrosse_scoop_policy.json

Only needs `torch` — does not need Isaac Sim running.
"""

import argparse
import json

import torch

OBS_LABELS = [
    "x", "q1", "q2", "x_dot", "q1_dot", "q2_dot",
    "ball_dx", "ball_dz", "ball_rot_y", "ball_vx", "ball_vz",
]
ACTION_LABELS = ["chassis_accel_cmd", "shoulder_vel_cmd", "elbow_vel_cmd"]


def mat(t, ndigits=8):
    return [[round(float(v), ndigits) for v in row] for row in t.tolist()]


def vec(t, ndigits=8):
    return [round(float(v), ndigits) for v in t.tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd, vd, pp = ckpt["policy"], ckpt["value"], ckpt["observation_preprocessor"]

    out = {
        "meta": {
            "source_checkpoint": args.checkpoint,
            "framework": "skrl PPO (GaussianMixin policy, shared backbone)",
            "note": (
                "weight matrices use PyTorch nn.Linear convention: "
                "shape=(out_features, in_features), y = x @ W^T + b"
            ),
        },
        "obs_dim": 11,
        "act_dim": 3,
        "obs_labels": OBS_LABELS,
        "action_labels": ACTION_LABELS,
        "obs_normalization": {
            "formula": "clip((x - mean) / (sqrt(var) + eps), -clip, clip)",
            "eps": 1e-8,
            "clip": 5.0,
            "mean": vec(pp["running_mean"]),
            "var": vec(pp["running_variance"]),
        },
        "layers": {
            "fc1": {
                "in": 11, "out": 256, "activation": "elu",
                "weight": mat(sd["net_container.0.weight"]),
                "bias": vec(sd["net_container.0.bias"]),
            },
            "fc2": {
                "in": 256, "out": 128, "activation": "elu",
                "weight": mat(sd["net_container.2.weight"]),
                "bias": vec(sd["net_container.2.bias"]),
            },
            "policy_head": {
                "in": 128, "out": 3, "activation": "none (= action mean)",
                "weight": mat(sd["policy_layer.weight"]),
                "bias": vec(sd["policy_layer.bias"]),
            },
            "value_head": {
                "in": 128, "out": 1, "activation": "none",
                "weight": mat(vd["value_layer.weight"]),
                "bias": vec(vd["value_layer.bias"]),
            },
        },
        "log_std": vec(sd["log_std_parameter"]),
        "action_postprocessing": (
            "clip(mean_action, -1, 1); at inference the mean is used directly (no sampling)"
        ),
    }

    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
