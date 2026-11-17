# Humanoid_fly: teaching a humanoid robot to fly with thrusters (Isaac Lab 3.0 + PPO)

![Iron Man R1 flying a waypoint tour](media/ironman_final.gif)

A Unitree R1 humanoid (29 kg, 26 joints) learns to take off, fly to 3D waypoints and hover, trained with
reinforcement learning (PPO, 4096 parallel simulations on one GPU). Two variants:

- **Iron Man**: thrusters fixed in the boots and gloves, no gimbals. The policy controls 4 throttles + 24 joints and
  steers only by pointing its feet and swinging its arms. ([full tour video](media/ironman_final_tour.mp4))
- **Jet pack**: two gimballed boosters on the back. ([demo video](media/jet_booster_demo.mp4))

| task | crashes | waypoints reached | hold error | notes |
|---|---|---|---|---|
| Iron Man (final) | 0 / 256 | 100 % | 1.4 cm | faces each goal within 2 deg while holding |
| Jet pack (final) | 0 / 256 | 100 % | 1.3 cm | flies leaning forward (~28 deg mean tilt) |

Each eval: 256 robots fly the same 7-waypoint tour from a standing take-off (1.5-3.5 m altitude, 4 s per waypoint).
How it was built, what went wrong and the lessons: [notes/NOTES.md](notes/NOTES.md) (both variants, Iron Man detail)
and [notes/JET_BOOSTERS.md](notes/JET_BOOSTERS.md) (jet design, run-by-run history, open issues).
Training progression videos: [iter 200](media/ironman_progress_1_iter200.mp4),
[iter 800](media/ironman_progress_2_iter800.mp4), [with heading command](media/ironman_progress_3_heading_iter500.mp4).
Best policies with their exact configs: `checkpoints/`. Short training clips: `media/training_clips/`.
Every raw checkpoint, log and video (886 MB): the `raw-logs-2026-11-09` GitHub release (extract into the repo root).

## Jet pack
The R1 (29 kg) gets two gimballed jet boosters on the back of its torso and learns to fly to 3D waypoints with PPO.

### Tasks
| id | actions | notes |
|---|---|---|
| `Humanoid-Fly-R1-Jet-v0` | 6: throttle L/R, gimbal pitch L/R, gimbal roll L/R | body held in its default pose by PD |
| `Humanoid-Fly-R1-Jet-Smooth-v0` | 6 | stage 2 (`--resume`), smoother; final policy `checkpoints/jet/model_1499.pt` |
| `Humanoid-Fly-R1-Jet-WholeBody-v0` | 6 + 24 joint offsets | policy also moves arms/legs/waist |

Boosters: total thrust-to-weight 2.0 (284 N each), ±25° gimbal, 80 ms spool lag. They're simulated as a force + torque
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
- In training (4096 robots, cloned in Fabric) flames have a fixed length (`attached_flame_throttle`). In
  `scripts/showcase.py` every robot is in USD, so flames follow each robot's throttle (and the jet's gimbal).
- Training clips (W&B, `logs/<exp>/<run>/videos/train/`) are shot from just outside env 0, looking across the grid:
  one robot in front, the others training behind it. Tune with `camera_back/height/ahead/look_height` in the cfgs.
- Jet: the casings use `booster_vis_cant_deg` (-13.5). If the log prints `booster casings are drawn ... off the real
  neutral thrust axis`, put the printed value there. If the casings clip into the torso, raise `booster_vis_back_offset`.

## Retraining runbook (both variants, then showcase + curves)
Run on the GPU box (L40S, ~100 min per 2000 Iron Man iterations, ~1 h per 1200 jet iterations), from this folder.
Fresh runs save extra checkpoints every 5 iterations up to iteration 60 (`--early_save_every/--early_save_until`) so
the showcase can show the robot learning to balance (it goes from falling over to flying in ~50 iterations).

**0. Visual check first (2 min, no training):** render the shipped policies and look at the flames / jetpack.
```bash
~/IsaacLab/isaaclab.sh -p scripts/showcase.py --headless --task Humanoid-Fly-R1-IronMan-Smooth-v1 \
    --checkpoints checkpoints/ironman/model_2599.pt --labels "Iron Man (old policy)" --out_dir /tmp/vis_ironman --seconds 6
~/IsaacLab/isaaclab.sh -p scripts/showcase.py --headless --task Humanoid-Fly-R1-Jet-Smooth-v0 \
    --checkpoints checkpoints/jet/model_1499.pt --labels "Jet pack (old policy)" --out_dir /tmp/vis_jet --seconds 6
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
                                                                                                     # -> logs/r1_ironman_v1_smooth/<C>; final = $(last logs/r1_ironman_v1_smooth/<C>)
```

**2. Jet pack** (2 stages):
```bash
bash scripts/train.sh --task Humanoid-Fly-R1-Jet-v0 --max_iterations 1200                          # -> logs/r1_jet/<D>
bash scripts/train.sh --task Humanoid-Fly-R1-Jet-Smooth-v0 --resume $(last logs/r1_jet/<D>) --max_iterations 300
                                                                                                     # -> logs/r1_jet_smooth/<E>; final = $(last logs/r1_jet_smooth/<E>)
```

**3. Evaluation** (waypoint tour from a standing take-off; note what it does and doesn't test, below):
```bash
~/IsaacLab/isaaclab.sh -p scripts/demo.py --headless --task Humanoid-Fly-R1-IronMan-v1 --checkpoint <final> --no_video --num_envs 256
~/IsaacLab/isaaclab.sh -p scripts/demo.py --headless --task Humanoid-Fly-R1-Jet-v0 --checkpoint <final> --no_video --num_envs 256
```

**4. Showcase videos** (36 robots taking off from the ground; same starts and goals per checkpoint):
```bash
~/IsaacLab/isaaclab.sh -p scripts/showcase.py --headless --task Humanoid-Fly-R1-IronMan-Smooth-v1 \
    --checkpoints logs/r1_ironman_v1/<A>/model_0.pt logs/r1_ironman_v1/<A>/model_25.pt logs/r1_ironman_v1/<A>/model_50.pt <final> \
    --labels "Iteration 0" "Iteration 25" "Iteration 50" "Final" --out_dir media/showcase/ironman
~/IsaacLab/isaaclab.sh -p scripts/showcase.py --headless --task Humanoid-Fly-R1-Jet-Smooth-v0 \
    --checkpoints logs/r1_jet/<D>/model_0.pt logs/r1_jet/<D>/model_50.pt <final> \
    --labels "Iteration 0" "Iteration 50" "Final" --out_dir media/showcase/jet
```
Pick the middle checkpoints from the curves (where episode length shoots up). Outputs per folder: `clip_<i>.mp4/.gif`,
`before_after.mp4/.gif` (first vs last side by side), `metrics.json` (crashes per clip).

**5. Curves** (reward, PPO value/surrogate loss, episode length + crashes, distance to goal + action std, reward terms):
```bash
python scripts/plot_curves.py logs/r1_ironman_v1/<A> logs/r1_ironman_v1_smooth/<B> logs/r1_ironman_v1_smooth/<C> \
    --out media/curves/ironman --title "Iron Man"
python scripts/plot_curves.py logs/r1_jet/<D> logs/r1_jet_smooth/<E> --out media/curves/jet --title "Jet pack"
```
Needs only numpy + matplotlib (runs on a laptop too). The short dips in reward / episode length at each dashed stage
line are a logging artifact of resuming (episode statistics restart), not the policy getting worse.

**6. Keep the results:** copy the new final checkpoints + `params/` into `checkpoints/`, commit `media/showcase/*`
(gifs, before_after.mp4, metrics.json) and `media/curves/*`, update the result tables above, and back up `logs/` as a
GitHub release before shutting the instance down.

**What the evaluation does and doesn't show.** `demo.py` flies one fixed 7-waypoint tour from a standing take-off;
across the 256 robots only the starting heading differs, and the policy is deterministic, with no noise, pushes or
randomized physics. "0 / 256 crashes" therefore means "reliably flies this tour", not a robustness result. Training
itself does include mid-air recovery starts and random goals.

