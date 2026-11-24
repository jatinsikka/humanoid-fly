"""Check candidate Iron Man hover poses before training on them (no training, ~3 min).

For each pose (joint angles, torso upright, gravity off so it stays put) it prints
  - the angle from vertical of each upper arm, forearm, hand jet and foot jet (0 = pointing straight down)
  - the centre of mass relative to the thrust points
  - the throttles that hover in exactly this pose (total thrust = weight, no pitch / roll torque about the CoM),
    or how far the CoM is off if no throttle split in [0, 1] can do it (then the robot would have to lean)
and renders a front and a side close-up of the posed robot with flames.

    ~/IsaacLab/isaaclab.sh -p scripts/pose_check.py --headless --poses scripts/hover_poses.json --out_dir /tmp/pose_check

--poses: JSON list of {"name": ..., "joints": {regex: rad}, "offsets": {joint: rad from default}}; other joints keep
the cfg's default pose.
Writes <out_dir>/<name>.png (3/4 close-up) and pose_check.json. A pose with "solve_arms": {"out_deg": 10} gets
its arm angles solved for straight arms hanging that far out, hand jets vertical (right arm mirrored).
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Humanoid-Fly-R1-IronMan-Hold-v1")
parser.add_argument("--poses", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--write_target", nargs=2, metavar=("POSE_NAME", "JSON"), default=None,
                    help="write that pose as a hover-pose target (offsets from the default pose, controlled joints)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import humanoid_fly  # noqa: E402,F401
from isaaclab.utils.string import resolve_matching_names_values  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

os.makedirs(args.out_dir, exist_ok=True)
poses = json.load(open(args.poses))

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env_cfg.sim.gravity = (0.0, 0.0, 0.0)  # hold the pose in the air; the weight is accounted for below
env_cfg.air_start_prob = 0.0
env_cfg.joint_target_filter_tau = 0.0
env_cfg.attached_flame_throttle = 0.5
env_cfg.video_recorder.window_width, env_cfg.video_recorder.window_height = 900, 900
# close 3/4 view of the posed robot (single env: visuals.grid_camera looks along the diagonal); pelvis at 2 m
env_cfg.camera_back, env_cfg.camera_height, env_cfg.camera_ahead, env_cfg.camera_look_height = 2.4, 2.2, 0.0, 1.85
env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
u = env.unwrapped
env.reset()
rob = u._robot
dev = u.device
G = 9.81
masses = wp.to_torch(rob.root_view.get_masses())[0].float().to(dev)
mass = float(masses.sum())
names = rob.joint_names
lim = rob.data.joint_pos_limits.torch[0].cpu().numpy()
body = {n: i for i, n in enumerate(rob.body_names)}


def body_pos(name):
    return rob.data.body_link_pos_w.torch[0, body[name]].cpu().numpy()


def angle_from_down(v):
    v = np.asarray(v, dtype=float)
    return float(np.degrees(np.arccos(np.clip(-v[2] / np.linalg.norm(v), -1, 1))))


# what the policy can command: default pose +- each controlled joint's action scale (clamped to the joint limits)
reach = lim.copy()
for k, i in enumerate(u._ctrl_joint_ids):
    d0, sc = float(rob.data.default_joint_pos.torch[0, i]), float(u._joint_scales[k])
    reach[i] = [max(lim[i, 0], d0 - sc), min(lim[i, 1], d0 + sc)]


def place(qc, steps):
    """Pin the base upright at 2 m and hold the joints at qc for a few physics steps (gravity is off)."""
    root = rob.data.default_root_pose.torch[:1].clone()
    root[:, :3] = u._terrain.env_origins[:1] + torch.tensor([0.0, 0.0, 2.0], device=dev)
    e0 = torch.tensor([0], device=dev)
    for _ in range(steps):
        rob.write_root_pose_to_sim_index(root_pose=root, env_ids=e0)
        rob.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=e0)
        rob.write_joint_position_to_sim_index(position=qc.unsqueeze(0), env_ids=e0)
        rob.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(qc).unsqueeze(0), env_ids=e0)
        rob.set_joint_position_target_index(target=qc.unsqueeze(0))
        u._joint_targets[:] = qc.unsqueeze(0)
        rob.write_data_to_sim()
        u.sim.step(render=False)
        u.scene.update(u.physics_dt)
    return root


def arm_cost(out_target_deg):
    """Left arm: upper arm out by out_target_deg and not forward/back, forearm in line with it, hand jet vertical."""
    sh, el, wr = body_pos("left_shoulder_pitch_link"), body_pos("left_elbow_link"), body_pos("left_wrist_roll_link")
    ua, fa = el - sh, wr - el
    ua_out, ua_fwd = np.degrees(np.arctan2(ua[1], -ua[2])), np.degrees(np.arctan2(ua[0], -ua[2]))
    bend = np.degrees(np.arccos(np.clip(np.dot(ua, fa) / np.linalg.norm(ua) / np.linalg.norm(fa), -1, 1)))
    _, push, _ = u._thruster_frames()
    jet = 180 - angle_from_down(push[0, 2].cpu().numpy())
    return (ua_out - out_target_deg) ** 2 + ua_fwd ** 2 + bend ** 2 + 0.5 * jet ** 2, (ua_out, ua_fwd, bend, jet)


def solve_arms(q, spec):
    """Coordinate search over left shoulder pitch / roll / yaw, elbow and wrist roll (right arm mirrored: roll and
    yaw negated) for a straight arm hanging spec["out_deg"] out from the body with the hand jet vertical."""
    J = {k: names.index(f"left_{k}_joint") for k in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")}
    R = {k: names.index(f"right_{k}_joint") for k in J}
    sign = {"shoulder_pitch": 1, "shoulder_roll": -1, "shoulder_yaw": -1, "elbow": 1, "wrist_roll": -1}
    q = q.clone()

    def setv(k, v):  # only angles the policy can reach (default +- action scale)
        v = float(np.clip(v, reach[J[k], 0], reach[J[k], 1]))
        q[J[k]] = v
        q[R[k]] = float(np.clip(sign[k] * v, reach[R[k], 0], reach[R[k], 1]))

    def cost():
        place(q, steps=3)
        return arm_cost(spec.get("out_deg", 10.0))

    best, info = cost()
    for step in (0.4, 0.2, 0.1, 0.05, 0.02, 0.01):
        improved = True
        while improved:
            improved = False
            for k in J:
                for d in (-step, step):
                    old = float(q[J[k]])
                    setv(k, old + d)
                    c, inf = cost()
                    if c < best - 1e-6:
                        best, info, improved = c, inf, True
                    else:
                        setv(k, old)
    print(f"[POSE] solved arms: upper arm out {info[0]:.1f} deg, fwd {info[1]:.1f} deg, elbow bend {info[2]:.1f} deg, "
          f"hand jet {info[3]:.1f} deg from vertical (cost {best:.2f})")
    return q


results = []
default = rob.data.default_joint_pos.torch[0].clone()
for pose in poses:
    q = default.clone()
    for regex, val in pose.get("joints", {}).items():
        ids, _, _ = resolve_matching_names_values({regex: val}, names)
        for i in ids:
            q[i] = val
    for jn, off in pose.get("offsets", {}).items():  # offsets from the default pose, e.g. hover_pose.json
        q[names.index(jn)] = default[names.index(jn)] + off
    qc = torch.max(torch.min(q, torch.as_tensor(lim[:, 1], device=dev)), torch.as_tensor(lim[:, 0], device=dev))
    clipped = [names[i] for i in range(len(names)) if abs(float(qc[i] - q[i])) > 1e-4]
    if "solve_arms" in pose:
        qc = solve_arms(qc, pose["solve_arms"])
        pose["solved"] = {names[i]: round(float(qc[i]), 4) for i in range(len(names)) if any(
            k in names[i] for k in ("shoulder", "elbow", "wrist"))}
    root = place(qc, steps=30)

    bpos = getattr(rob.data, "body_com_pos_w", rob.data.body_link_pos_w).torch[0]
    com = (bpos * masses.unsqueeze(-1)).sum(0).cpu().numpy() / mass
    nozzle, push, _ = u._thruster_frames()
    nozzle, push = nozzle[0].cpu().numpy(), push[0].cpu().numpy()
    tmax = np.array(u.cfg.thruster_max_twr) * mass * G  # u._max_thrust is 0 here: the env scales it by gravity
    # hover balance: throttles t in [0, 1] with sum(t_i Tmax_i push_i)_z = W and zero roll / pitch torque about the CoM
    A = np.zeros((3, 4))
    for i in range(4):
        f = tmax[i] * push[i]
        tau = np.cross(nozzle[i] - com, f)
        A[:, i] = [f[2], tau[0], tau[1]]
    b = np.array([mass * G, 0.0, 0.0])
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    best = None  # also search the [0,1] box for the smallest residual (4 throttles, grid + refine)
    for t0 in np.linspace(0, 1, 21):
        for t1 in np.linspace(0, 1, 21):
            # given feet throttles, solve the hands' two throttles from 2 of the 3 equations, score all 3
            M = A[:, 2:]
            rhs = b - A[:, :2] @ np.array([t0, t1])
            th, *_ = np.linalg.lstsq(M, rhs, rcond=None)
            th = np.clip(th, 0, 1)
            tt = np.array([t0, t1, *th])
            r = A @ tt - b
            score = abs(r[0]) / (mass * G) + (abs(r[1]) + abs(r[2])) / (mass * G * 0.1)
            if best is None or score < best[0]:
                best = (score, tt, r)
    _, tb, rb = best
    # residual torque -> CoM offset the thrust can't cancel -> lean needed to bring the CoM over the thrust centre
    lean_deg = float(np.degrees(np.arctan2(np.hypot(rb[1], rb[2]) / (mass * G), max(com[2] - nozzle[:2, 2].mean(), 1e-3))))
    arms = {}
    for side in ("left", "right"):
        sh, el, wr = body_pos(f"{side}_shoulder_pitch_link"), body_pos(f"{side}_elbow_link"), body_pos(f"{side}_wrist_roll_link")
        ua, fa = el - sh, wr - el
        out = 1.0 if side == "left" else -1.0  # +y is the robot's left
        arms[side] = {"upper_arm_from_down_deg": angle_from_down(ua), "forearm_from_down_deg": angle_from_down(fa),
                      "upper_arm_out_fwd_deg": [round(float(np.degrees(np.arctan2(out * ua[1], -ua[2]))), 1),
                                                round(float(np.degrees(np.arctan2(ua[0], -ua[2]))), 1)],
                      "forearm_out_fwd_deg": [round(float(np.degrees(np.arctan2(out * fa[1], -fa[2]))), 1),
                                              round(float(np.degrees(np.arctan2(fa[0], -fa[2]))), 1)]}
    unreachable = [f"{names[i]} {float(qc[i]):+.2f} (reach {reach[i, 0]:+.2f}..{reach[i, 1]:+.2f})"
                   for i in u._ctrl_joint_ids if not reach[i, 0] - 1e-3 <= float(qc[i]) <= reach[i, 1] + 1e-3]
    res = {
        "name": pose["name"],
        "unreachable_by_policy": unreachable,
        "clipped_to_limits": clipped,
        "upper_arm_from_down_deg": [round(arms[s]["upper_arm_from_down_deg"], 1) for s in ("left", "right")],
        "forearm_from_down_deg": [round(arms[s]["forearm_from_down_deg"], 1) for s in ("left", "right")],
        "upper_arm_out/fwd_deg (L, R)": [arms[s]["upper_arm_out_fwd_deg"] for s in ("left", "right")],
        "forearm_out/fwd_deg (L, R)": [arms[s]["forearm_out_fwd_deg"] for s in ("left", "right")],
        "jet_push_from_up_deg (L foot, R foot, L hand, R hand)": [round(180 - angle_from_down(p), 1) for p in push],
        "com_minus_feet_nozzles_xy_cm": [round(100 * float(c), 1) for c in (com[:2] - nozzle[:2, :2].mean(0))],
        "com_minus_hand_nozzles_xy_cm": [round(100 * float(c), 1) for c in (com[:2] - nozzle[2:, :2].mean(0))],
        "hover_throttles (L foot, R foot, L hand, R hand)": [round(float(x), 3) for x in tb],
        "hover_residual (lift N, roll Nm, pitch Nm)": [round(float(x), 2) for x in rb],
        "lean_needed_deg": round(lean_deg, 1),
        "mass_kg": round(mass, 1),
    }
    results.append(res)
    if args.write_target and args.write_target[0] == pose["name"]:
        assert not unreachable, f"target pose has joints the policy can't reach: {unreachable}"
        ctrl = [names[i] for i in u._ctrl_joint_ids]
        tgt = {"offsets": {n: round(float(qc[names.index(n)] - default[names.index(n)]), 5) for n in ctrl},
               "absolute": {n: round(float(qc[names.index(n)]), 5) for n in ctrl},
               "source": f"scripts/pose_check.py pose '{pose['name']}'", "check": res}
        json.dump(tgt, open(args.write_target[1], "w"), indent=2)
        print(f"[POSE] wrote target {args.write_target[1]}")
    print(f"[POSE] {json.dumps(res)}")

    # close-up (env camera, 3/4 view); the first frames after a pose change can lag, so render a few
    for _ in range(6):
        frame = u.render()
    imageio.imwrite(os.path.join(args.out_dir, f"{pose['name']}.png"), frame)

json.dump({"poses": poses, "results": results}, open(os.path.join(args.out_dir, "pose_check.json"), "w"), indent=2)
print(f"[POSE] wrote {args.out_dir}")
env.close()
app.close()
