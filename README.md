# Humanoid_fly: teaching a humanoid robot to fly with thrusters (Isaac Lab 3.0 + PPO)

![Iron Man R1 flying a waypoint tour](media/ironman_final.gif)

A Unitree R1 humanoid (29 kg, 26 joints) learns to take off, fly to 3D waypoints and hover, trained with
reinforcement learning (PPO, 4096 parallel simulations on one GPU). Two variants:

- **Iron Man**: thrusters fixed in the boots and gloves, no gimbals. The policy controls 4 throttles + 24 joints and
  steers only by pointing its feet and swinging its arms. ([full tour video](media/ironman_final_tour.mp4))
- **Jet pack**: two gimballed boosters on the back. ([demo video](media/jet_booster_demo.mp4))

Robustness eval (`scripts/eval.py`, 2026-11-24 retrain): 512 robots x 12 s per condition, random goals, deterministic
policy. Full tables: `media/eval/*.md`.

| policy | crashes: take-off / mid-air start | hover error | roll rate holding | crashes under velocity kicks | crashes with sensor noise |
|---|---|---|---|---|---|
| Iron Man, stage 3 | 0 % / 0 % | 1.0 cm | 0.16 rad/s | 0.2 % | 0 % |
| Jet pack, stage 2 | 0 % / 0 % | 1.1 cm | 1.30 rad/s | 0.2 % | 36 % |

Both reach every goal, but neither holds still once there: the jet rocks in roll/pitch/yaw, and Iron Man keeps
moving its arms and legs. Stage 3 of the jet (`Jet-Hold-v0`) and stage 4 of Iron Man (`IronMan-Hold-v1`) add
hold-at-goal terms for that (see "Reward terms" below).
How it was built, what went wrong and the lessons: [notes/NOTES.md](notes/NOTES.md) (both variants, Iron Man detail)
and [notes/JET_BOOSTERS.md](notes/JET_BOOSTERS.md) (jet design, run-by-run history, open issues).
Training progression videos: [iter 200](media/ironman_progress_1_iter200.mp4),
[iter 800](media/ironman_progress_2_iter800.mp4), [with heading command](media/ironman_progress_3_heading_iter500.mp4).
Best policies with their exact configs: `checkpoints/`. Short training clips: `media/training_clips/`.
Every raw checkpoint, log and video (886 MB): the `raw-logs-2026-11-09` GitHub release (extract into the repo root).

## The RL problem (both variants)
- **Timing:** policy at 50 Hz, physics at 200 Hz (each action held for 4 physics steps). Episodes up to 12 s
  (600 decisions), 4096 robots in parallel.
- **Starts and goals:** half the resets start standing on the ground, half mid-air (tilted up to ~17 deg, moving up to
  1 m/s, spinning up to 0.5 rad/s, throttle at hover). A goal is a random point within +-3 m horizontally and 1-4 m
  up; a new one every 6 s.
- **Observations** (body frame): linear velocity, angular velocity, gravity direction, vector to the goal, height,
  throttle state, joint positions and velocities, previous action. Iron Man adds each limb's thrust direction (and,
  in the heading tasks, sin/cos of the heading error): 109 / 111 values. The jet adds its gimbal angles: 77 values.
- **Actions** in [-1, 1]: Iron Man 28 (4 throttles + 24 joint offsets from the default pose; the thrusters can't
  swivel, so it steers by moving its limbs). Jet 6 (2 throttles + pitch/roll gimbal per booster, +-25 deg).
- **Termination (crash):** pelvis below 0.35 m, above 8 m, more than 6 m sideways from its start, or tilted more than
  ~75 deg. Reaching 12 s is a time-out, not a crash.
- **PPO:** gamma 0.99 (horizon ~2 s at 50 Hz), lambda 0.95, 24 steps per robot per iteration.

### Reward terms
Each step adds `sum(term * weight) * dt`, so the weights are rates per second (the crash penalty is one-off).
Stages are fine-tunes: each resumes the previous one and changes only the weights listed.

| term | what it measures | Iron Man stage 1 → 2 → 3 → 4 | Jet stage 1 → 2 → 3 |
|---|---|---|---|
| `pos_coarse` | `1 - tanh(dist / 1.5 m)` | +10 | +10 |
| `pos_fine` | `1 - tanh(dist / 0.3 m)` | +6 | +6 |
| `progress` | speed toward the goal, clipped to +-2 m/s | +3 | +3 |
| `upright` | tilt (Iron Man: of the pelvis; jet: of the neutral thrust axis) | -3 → -5 | -3 → -5 |
| `ang_vel` | body angular velocity squared | -0.08 → -0.15 | -0.08 → -0.15 |
| `lin_vel` | body linear velocity squared | -0.02 | -0.02 |
| `action_rate` | change of the action vector squared | -0.01 → -0.04 | -0.01 → -0.05 |
| `throttle` | mean throttle (fuel) | -0.05 | -0.05 |
| `action_bound` | policy outputs past +-1 (keeps them inside the clamp) | -1 | -1 |
| `joint_dev` | distance from the default pose (Iron Man) | -0.1 → -0.3 | - |
| `joint_vel` | joint speeds squared (Iron Man) | -0.0005 → -0.002 | - |
| `heading` | `((1 + cos yaw error) / 2) ^ sharpness`, face the next goal (Iron Man v1) | +2 (sharpness 1) → 3 (sharpness 4) in stage 3 | - |
| `yaw_rate` | Iron Man: yaw rate vs. a target turn rate `clamp(gain * yaw error, +-1.5)`; jet: plain yaw-rate penalty | -0.1 → -0.5 (gain 2) in stage 3 | 0 → -0.1 |
| `hold_ang_vel` | angular velocity squared, near the goal only | 0 → -1.0 in stage 4 | 0 → -1.0 in stage 3 |
| `hold_lin_vel` | linear velocity squared, near the goal only | - | 0 → -0.5 in stage 3 |
| `hold_joint_vel` | joint speeds squared, near the goal only | 0 → -0.03 in stage 4 | - |
| `hold_action_rate` | action change squared, near the goal only | 0 → -0.5 in stage 4 | 0 → -0.1 in stage 3 |
| `hold_pose` | distance from the recorded hover pose, near the goal only | 0 → -3.0 in stage 4 | - |
| `termination` | crash (one-off) | -5 | -5 |

"Near the goal only" = multiplied by `exp(-(dist / 0.5 m)^2)`: 1 at the goal, 0.37 at 0.5 m, ~0 past 1 m, so the
hold terms don't slow the flight between goals. That matters most for Iron Man, whose limbs are its steering.
Stage 4 of Iron Man also low-pass filters the joint targets (`joint_target_filter_tau = 0.05` s), so fast limb
jitter can't reach the joints, and pulls the limbs toward its own hover pose: `scripts/hover_pose.py` records the
stage-3 policy's average limb pose while holding at its goals into `humanoid_fly/tasks/r1_ironman/hover_pose.json`.
Why: at the end of stage 3 all the limb-motion penalties together were ~2 % of the reward (joint speed RMS while
holding ~1.7 rad/s), so fidgeting was essentially free.

## Jet pack
The R1 (29 kg) gets two gimballed jet boosters on the back of its torso and learns to fly to 3D waypoints with PPO.

### Tasks
| id | actions | notes |
|---|---|---|
| `Humanoid-Fly-R1-Jet-v0` | 6: throttle L/R, gimbal pitch L/R, gimbal roll L/R | body held in its default pose by PD |
| `Humanoid-Fly-R1-Jet-Smooth-v0` | 6 | stage 2 (`--resume`), smoother; final policy `checkpoints/jet/model_1499.pt` |
| `Humanoid-Fly-R1-Jet-Hold-v0` | 6 | stage 3 (`--resume` the stage-2 final): hold-at-goal terms, settle instead of rocking |
| `Humanoid-Fly-R1-Jet-WholeBody-v0` | 6 + 24 joint offsets | policy also moves arms/legs/waist |

Boosters: mounted 0.40 m apart (v3, 2026-11-23; the trained v2 policies in `checkpoints/jet/` used 0.24 m and roll
~1.5 rad/s, see notes/JET_BOOSTERS.md), total thrust-to-weight 2.0 (284 N each), ±25° gimbal, 80 ms spool lag. They're simulated as a force + torque
on `waist_yaw_link`. The jetpack you see (two casings and a pack plate on the back, a flame out of each casing) is
visual only, attached to the torso link so it moves rigidly with the robot; it adds no mass and has no collisions.
Half the episodes start standing on the ground (take-off), half start mid-air with random tilt/velocity (recovery).
A new goal every 6 s; episodes are 12 s long. Tune everything in `humanoid_fly/tasks/r1_jet/r1_jet_env_cfg.py`.

## Iron Man (thrusters in the boots and gloves)
| id | notes |
|---|---|
| `Humanoid-Fly-R1-IronMan-v0` | 4 throttles + 24 joint offsets, fly to 3D goals, yaw free |
| `Humanoid-Fly-R1-IronMan-v1` | same, plus a heading command: face the next goal (sin/cos of the yaw error observed) |
| `Humanoid-Fly-R1-IronMan-Smooth-v0` / `-Smooth-v1` | stage 2 (`--resume` a stage-1 checkpoint): stronger upright/ang-vel/action-rate/pose penalties |
| `Humanoid-Fly-R1-IronMan-Hold-v1` | stage 4 (`--resume` the stage-3 final): stage 3's heading settings built in, plus hold-at-goal terms, hover-pose target and joint-target filter |

Four thrusters fixed to the limbs, no gimbals: one under each foot (`*_ankle_roll_link`, 0.75 x weight each) and one
in each hand (`*_wrist_roll_link`, exhaust out past the fingertips, 0.30 x weight each); total thrust-to-weight 2.1,
60 ms throttle lag. The policy steers like Iron Man does: by pointing its feet and swinging its arms (the thrust is a
force at each nozzle point, so the reaction loads go through the joints). Ankles/arms are stiffer than stock (a "suit").
The flames are visual only and attached to the hand and foot links (no casings), so they render with the limbs.
Config: `humanoid_fly/tasks/r1_ironman/r1_ironman_env_cfg.py`.

Training recipe that produced the demo (each stage resumes the previous one; `--env KEY=VALUE` overrides env cfg fields):
```bash
bash scripts/train.sh --task Humanoid-Fly-R1-IronMan-v1                     # stage 1, model_1300 used
bash scripts/train.sh --task Humanoid-Fly-R1-IronMan-Smooth-v1 --resume logs/r1_ironman_v1/<run>/model_1300.pt \
    --reset_std 0.25                                                        # stage 2, to model_1800
bash scripts/train.sh --task Humanoid-Fly-R1-IronMan-Smooth-v1 --resume logs/r1_ironman_v1_smooth/<run>/model_1800.pt \
    --max_iterations 800 --env yaw_rate_reward_scale=-0.5 yaw_rate_tracking_gain=2.0 heading_reward_scale=3.0 heading_sharpness=4.0
```
The last stage swaps the plain yaw-rate penalty for yaw-rate *tracking* (turn toward the target heading at
`clamp(2 * err, +-1.5)` rad/s) and sharpens the heading reward: a plain yaw-rate penalty either let it wiggle in place
(0.32 rad/s) or made it give up on facing the goal (15 deg steady offset).

Demo: take off from the ground, fly a 7-waypoint tour (4 s per waypoint, 1.5-3.5 m up), chase camera on env 0:
```bash
~/IsaacLab/isaaclab.sh -p scripts/demo.py --headless --task Humanoid-Fly-R1-IronMan-v1 --checkpoint <ckpt> [--wandb]
~/IsaacLab/isaaclab.sh -p scripts/demo.py --headless --task Humanoid-Fly-R1-IronMan-v1 --checkpoint <ckpt> --no_video --num_envs 256  # metrics
```
Results over 256 envs (same tour in every env):

| checkpoint | crashes | waypoints reached | hold error | hold heading error | hold yaw rate | mean tilt |
|---|---|---|---|---|---|---|
| stage 2 `model_1800` | 0 % | 100 % | 1.4 cm | 6.1 deg | 0.32 rad/s | 6.5 deg |
| stage 3 `model_2200` | 0 % | 100 % | 0.7 cm | 3.4 deg | 0.11 rad/s | 8.3 deg |
| **stage 3 `model_2599`** | 0 % | 100 % | 1.4 cm | 2.0 deg | 0.10 rad/s | 9.1 deg |

"Reached" = within 0.3 m at the end of a waypoint's 4 s; "hold" columns = averaged over the second half of each 4 s.
Final demo: `media/ironman_final_tour.mp4` (policy: `checkpoints/ironman/model_2599.pt`), also on W&B (`humanoid_fly`, run `yw61ij4f`,
job type `demo`). Training runs in the same project: `p34hhgdt` (stage 1), `3mzn62ud` (stage 2), `2sirb1d0` (stage 3).

## Run (from this folder)
```bash
~/IsaacLab/isaaclab.sh -p -m pip install -e .        # once (r1_factory must be installed too)
~/IsaacLab/isaaclab.sh -p scripts/test_env.py --headless   # smoke test: open-loop thrust
bash scripts/train.sh                                 # headless, 4096 envs, ~100 min on the L40S, logs to W&B
bash scripts/train.sh --no_video --logger tensorboard # ~20% faster, no W&B
bash scripts/play.sh                                  # GUI, latest checkpoint; also exports policy.pt
```
Logs/checkpoints: `logs/r1_jet/<timestamp>/`. W&B project `humanoid_fly`: curves plus a 6 s clip of env 0 every
100 iterations (Media panel, key `video`). The camera is aimed in `R1JetEnv._setup_scene`.

Isaac Lab's own `scripts/reinforcement_learning/rsl_rl/train.py` crashes at startup on this machine
(`free(): invalid pointer`, also with the stock quadcopter task), so `scripts/train.py` / `play.py` are used instead.

## Visuals (no physics effect)
- **Flames and the jetpack are parented to the robot's links** (`humanoid_fly/tasks/visuals.py`), so they render in
  the same frame as the robot. The old version drew them as VisualizationMarkers, which trailed the limbs by a frame
  or more at speed (the red boxes "flying off" the hands, the jetpack wobbling above the shoulders).
- In training flames have a fixed length (`attached_flame_throttle`; `live_flames = False`): on Isaac Lab versions
  that keep all 4096 robots in USD, animating them stalled training. `scripts/showcase.py` turns live flames on for its
  36-robot grid, so there they follow each robot's throttle (and the jet's gimbal).
- Training clips (W&B, `logs/<exp>/<run>/videos/train/`) are shot from just outside env 0, looking across the grid:
  one robot in front, the others training behind it. Tune with `camera_back/height/ahead/look_height` in the cfgs.
- Jet: the casings use `booster_vis_cant_deg` (-13.5). If the log prints `booster casings are drawn ... off the real
  neutral thrust axis`, put the printed value there. If the casings clip into the torso, raise `booster_vis_back_offset`.

## Retraining runbook (both variants, then showcase + curves)
Run on the GPU box (L40S, ~4 s per iteration with 4096 robots: Iron Man 1300 iterations ~100 min, jet 1200 ~100 min;
~5 min of startup per stage), from this folder. The 2026-11-24 retrain used exactly these commands, chained in one
script; W&B runs are in project `humanoid_fly`.

Gotchas on that box: `isaaclab.sh` exits 0 even when the Python script raised, so check that each stage reached its
last checkpoint (and grep the log for `Traceback`) before starting the next. rsl_rl 5 opens one W&B run per
`runner.learn()` call, so `scripts/train.py` calls it once (the early checkpoints come from a per-iteration hook).
Fresh runs save extra checkpoints every 5 iterations up to iteration 60 (`--early_save_every/--early_save_until`) so
the showcase can show the robot learning to balance (it goes from falling over to flying in ~50 iterations).

**0. Visual check first (2 min, no training):** render the shipped policies and look at the flames / jetpack.
```bash
~/IsaacLab/isaaclab.sh -p scripts/showcase.py --headless --task Humanoid-Fly-R1-IronMan-Smooth-v1 \
    --checkpoints checkpoints/ironman/model_2599.pt --labels "Iron Man (old policy)" --out_dir /tmp/vis_ironman --seconds 6
~/IsaacLab/isaaclab.sh -p scripts/showcase.py --headless --task Humanoid-Fly-R1-Jet-Smooth-v0 \
    --checkpoints checkpoints/jet/model_1499.pt --labels "Jet pack (old policy)" --out_dir /tmp/vis_jet --seconds 6 \
    --env "booster_mounts=((-0.10, 0.12, 0.30), (-0.10, -0.12, 0.30))"   # the old policy was trained with the narrow mount
```
Look for `flames attached` / `jetpack attached` and `live flames on` in the log, and no `WARNING`.

**1. Iron Man** (3 stages, each resumes the previous one). `last` = a stage's newest checkpoint (the exact final
number, e.g. 1299 vs 1300, depends on the rsl_rl version):
```bash
last() { ls -v "$1"/model_*.pt | tail -1; }
bash scripts/train.sh --task Humanoid-Fly-R1-IronMan-v1 --max_iterations 1300                      # -> logs/r1_ironman_v1/<A>
bash scripts/train.sh --task Humanoid-Fly-R1-IronMan-Smooth-v1 --resume $(last logs/r1_ironman_v1/<A>) \
    --reset_std 0.25 --max_iterations 500                                                            # -> logs/r1_ironman_v1_smooth/<B>
bash scripts/train.sh --task Humanoid-Fly-R1-IronMan-Smooth-v1 --resume $(last logs/r1_ironman_v1_smooth/<B>) \
    --max_iterations 800 --env yaw_rate_reward_scale=-0.5 yaw_rate_tracking_gain=2.0 heading_reward_scale=3.0 heading_sharpness=4.0
                                                                                                     # -> logs/r1_ironman_v1_smooth/<C>; stage-3 final = $(last logs/r1_ironman_v1_smooth/<C>)
# stage 4, hold still at the goal: record the stage-3 hover pose (commit the json), then fine-tune
~/IsaacLab/isaaclab.sh -p scripts/hover_pose.py --headless --task Humanoid-Fly-R1-IronMan-Smooth-v1 \
    --checkpoint $(last logs/r1_ironman_v1_smooth/<C>) --out humanoid_fly/tasks/r1_ironman/hover_pose.json
bash scripts/train.sh --task Humanoid-Fly-R1-IronMan-Hold-v1 --resume $(last logs/r1_ironman_v1_smooth/<C>) \
    --max_iterations 400                                                                             # -> logs/r1_ironman_hold/<H>; final = $(last logs/r1_ironman_hold/<H>)
```

**2. Jet pack** (2 stages):
```bash
bash scripts/train.sh --task Humanoid-Fly-R1-Jet-v0 --max_iterations 1200                          # -> logs/r1_jet/<D>
bash scripts/train.sh --task Humanoid-Fly-R1-Jet-Smooth-v0 --resume $(last logs/r1_jet/<D>) --max_iterations 300
                                                                                                     # -> logs/r1_jet_smooth/<E>
bash scripts/train.sh --task Humanoid-Fly-R1-Jet-Hold-v0 --resume $(last logs/r1_jet_smooth/<E>) --max_iterations 400
                                                                                                     # -> logs/r1_jet_hold/<J>; final = $(last logs/r1_jet_hold/<J>)
```

**3. Evaluation.** `scripts/eval.py` is the one to report: 512 robots x 12 s under each condition (standing take-off
with random goals, mid-air starts, velocity kicks, thrust -10% / +10%, observation noise), crash rate, goals reached,
hover error, tilt, roll rate, limb speed and action change while holding (how much it fidgets), and mid-air recovery
time, as a markdown table. Training never saw pushes, model error or
sensor noise, so those rows show what to train on next.
```bash
~/IsaacLab/isaaclab.sh -p scripts/eval.py --headless --task Humanoid-Fly-R1-IronMan-Smooth-v1 --checkpoint <stage-3 final> --out media/eval/ironman
~/IsaacLab/isaaclab.sh -p scripts/eval.py --headless --task Humanoid-Fly-R1-IronMan-Hold-v1 --checkpoint <stage-4 final> --out media/eval/ironman_hold
~/IsaacLab/isaaclab.sh -p scripts/eval.py --headless --task Humanoid-Fly-R1-Jet-Smooth-v0 --checkpoint <stage-2 final> --out media/eval/jet
~/IsaacLab/isaaclab.sh -p scripts/eval.py --headless --task Humanoid-Fly-R1-Jet-Hold-v0 --checkpoint <stage-3 final> --out media/eval/jet_hold
# the old (v2, narrow mount) jet policy, for comparison:
~/IsaacLab/isaaclab.sh -p scripts/eval.py --headless --task Humanoid-Fly-R1-Jet-Smooth-v0 --checkpoint checkpoints/jet/model_1499.pt \
    --out media/eval/jet_v2 --env "booster_mounts=((-0.10, 0.12, 0.30), (-0.10, -0.12, 0.30))"
```
`scripts/demo.py` (the fixed 7-waypoint tour with a chase camera) is still there for the tour video; its "0 / 256
crashes" is not a robustness number (see below).

**4. Showcase videos** (36 robots taking off from the ground; same starts and goals per checkpoint):
```bash
~/IsaacLab/isaaclab.sh -p scripts/showcase.py --headless --task Humanoid-Fly-R1-IronMan-Hold-v1 \
    --checkpoints logs/r1_ironman_v1/<A>/model_0.pt logs/r1_ironman_v1/<A>/model_25.pt <final> \
    --labels "Iteration 0" "Iteration 25" "Final" --out_dir media/showcase/ironman_hold
~/IsaacLab/isaaclab.sh -p scripts/showcase.py --headless --task Humanoid-Fly-R1-Jet-Hold-v0 \
    --checkpoints logs/r1_jet/<D>/model_0.pt logs/r1_jet/<D>/model_20.pt logs/r1_jet/<D>/model_30.pt <final> \
    --labels "Iteration 0" "Iteration 20" "Iteration 30" "Final" --out_dir media/showcase/jet_hold
```
Pick the middle checkpoints from the curves: just before episode length shoots up to 12 s (2026-11-24: Iron Man
survives full episodes from iteration ~55, the jet from ~35; by then the clips already look like the final policy). Outputs per folder: `clip_<i>.mp4/.gif`,
`before_after.mp4/.gif` (first vs last side by side), `metrics.json` (crashes per clip).

**5. Curves.** `<out>_summary.png` is the one-chart version (episode reward + value loss, stage boundaries dashed);
`<out>_training.png` the 2x2 overview (reward, PPO value/surrogate loss, episode length + crashes, distance to goal +
action std); `<out>_reward_terms.png` every term over training:
```bash
python scripts/plot_curves.py logs/r1_ironman_v1/<A> logs/r1_ironman_v1_smooth/<B> logs/r1_ironman_v1_smooth/<C> \
    logs/r1_ironman_hold/<H> --out media/curves/ironman --title "Iron Man"
python scripts/plot_curves.py logs/r1_jet/<D> logs/r1_jet_smooth/<E> logs/r1_jet_hold/<J> --out media/curves/jet --title "Jet pack"
```
Needs only numpy + matplotlib (runs on a laptop too). The short dips in reward / episode length at each dashed stage
line are a logging artifact of resuming (episode statistics restart), not the policy getting worse. Once nothing
crashes, all robots time out together every 12 s (25 iterations), so the raw value loss oscillates with that period;
the summary chart averages it over one period.

**6. Keep the results:** copy the new final checkpoints + `params/` into `checkpoints/`, commit `media/showcase/*`
(gifs, before_after.mp4, metrics.json), `media/curves/*` and `media/eval/*`, update the result tables above, then
back up every log and checkpoint before shutting the instance down:
```bash
bash scripts/backup_logs.sh            # GitHub release raw-logs-<date> (needs gh auth); restore instructions in the script
```
`eval.py`, `showcase.py` and `plot_curves.py` also take `--wandb` to put their tables / videos / plots in the W&B
project next to the training runs.

**What `demo.py` does and doesn't show.** It flies one fixed 7-waypoint tour from a standing take-off;
across the 256 robots only the starting heading differs, and the policy is deterministic, with no noise, pushes or
randomized physics. "0 / 256 crashes" therefore means "reliably flies this tour", not a robustness result: use
`scripts/eval.py` for that.

