# Project notes: teaching a humanoid to fly with thrusters (Nov 8-9, 2026)

Unitree R1 humanoid (28.9 kg, 26 joints) in Isaac Lab 3.0, trained with PPO (RSL-RL), 4096 parallel envs on one L40S.
Two variants, both flying to random 3D goals and recovering from random mid-air starts:

1. **Jet pack** (`Humanoid-Fly-R1-Jet-*`): two gimballed boosters on the back ("Pacific Rim" style).
2. **Iron Man** (`Humanoid-Fly-R1-IronMan-*`): four thrusters fixed in the boots and gloves, no gimbals. The robot
   steers only by moving its body: pointing its feet and swinging its arms, the way Iron Man does.

Everything (training curves, periodic training clips, demo videos + metrics) is logged to one W&B project,
`humanoid_fly`.

## Iron Man: design
- Thrusters: forces applied at the nozzle point of `*_ankle_roll_link` (0.75 x weight each, pushing along the sole
  normal) and `*_wrist_roll_link` (0.30 x weight each, exhaust out past the fingertips). Total thrust-to-weight 2.1,
  60 ms throttle lag. The force acts at the nozzle, so reaction loads go through the joints just like a real suit.
- Actions (28): 4 throttles + 24 joint position offsets (legs, waist, arms) around a flight pose
  (legs straight, arms down and slightly out). Ankles/arms are stiffer than stock (a "suit") so the hand jets
  don't fold the arms.
- Observations (111): body velocities, gravity, goal in body frame, height, throttles, the 4 thrust directions in
  body frame, joint states, last action, and sin/cos of the heading error.
- Rewards: coarse + fine distance-to-goal, progress, uprightness, smoothness (ang vel, action rate, joint vel),
  stay near the flight pose, throttle cost, a heading term (face the next goal) and yaw-rate tracking.
- Episodes: 12 s, new goal every 6 s; half start standing (take-off), half start mid-air with random tilt/velocity.

## Iron Man: what happened, in order
| stage | what changed | result |
|---|---|---|
| v0 (no heading) | first version, yaw left free | flew to goals within ~200 iterations, but spun/drifted in yaw: looks unnatural |
| v1 stage 1 | + heading command (face the next goal) | flies and faces the goal; jittery limbs |
| stage 2 "smooth" | resume, stronger smoothness/upright/pose penalties, policy std reset to 0.25 | 1.4 cm hold error, but wiggles in yaw while hovering (0.32 rad/s) |
| stage 3a | + plain yaw-rate penalty | wiggle gone (0.09 rad/s) but it gave up facing the goal (15 deg offset) |
| **stage 3b** | yaw-rate **tracking**: turn toward the target at clamp(2 x error, +-1.5 rad/s); sharper heading reward | **0 crashes, 100 % waypoints, 1.4 cm hold, 2 deg heading, 0.10 rad/s** |

Final evaluation: 256 copies fly the same 7-waypoint tour (take-off from the ground, 1.5-3.5 m altitude, 4 s per
waypoint). Best checkpoint: `checkpoints/ironman/model_2599.pt`.

Jet pack design and run-by-run history: [JET_BOOSTERS.md](JET_BOOSTERS.md).

## Lessons learned
- **A plain penalty can fight the task.** Penalising yaw rate stopped the wiggle but also stopped the robot turning
  toward its goal. Tracking a *desired* yaw rate (proportional to the heading error, clamped) fixed both.
- **Check your eval before trusting it.** Isaac Lab staggers initial episode lengths, so the first demo eval timed
  out mid-tour and every env "missed" a waypoint. Zeroing `episode_length_buf` at the start of the demo fixed it.
- **Geometry first (jet).** The first booster mount was 0.22 m above / 0.18 m behind the centre of mass and needed
  38 deg of gimbal to thrust through it (limit 25 deg), so there was no static hover and every policy spun at
  ~6 rad/s. Mounting at shoulder-blade height with the neutral axis canted through the CoM fixed it.
- **Entropy and action bounds.** Entropy 0.005 without a penalty for out-of-range actions let the policy std grow
  to 11 (bang-bang control); an `action_bound` penalty fixes it. Entropy 0 from scratch collapses exploration and
  never learns to fly. Entropy 0 is fine for the fine-tuning stages.
- **Curriculum by resuming.** Train for capability first (loose penalties), then resume with the smoothness
  penalties turned up (and the policy std reset low), so the strict penalties never block learning to fly.
- **Ops.** Isaac Lab's stock `rsl_rl/train.py` crashes at Kit startup on this machine (`free(): invalid pointer`),
  so `scripts/train.py` starts the AppLauncher itself. Long runs are launched with `setsid nohup ... < /dev/null &`
  so they survive the shell; never `pkill -f` a pattern that also matches your own command.

## Where things are
- `humanoid_fly/tasks/r1_ironman/`, `humanoid_fly/tasks/r1_jet/`: environments, configs, PPO configs
- `scripts/train.py` (`--env KEY=VALUE` overrides env cfg fields), `scripts/demo.py` (waypoint tour + metrics +
  video, `--wandb` uploads), `scripts/play.py`, smoke tests `test_env.py` / `test_ironman.py`
- `checkpoints/`: best policies with the exact env/agent configs they were trained with (`params/`)
- `media/`: demo videos; `notes/training_curves/`: per-100-iteration summaries of each Iron Man training stage
- W&B project `humanoid_fly` (entity jsikka-the-university-of-texas-at-austin): Iron Man training runs
  `p34hhgdt` (stage 1), `3mzn62ud` (stage 2), `9fg7gwbx` (stage 3a), `2sirb1d0` (stage 3b), final demo `yw61ij4f`
