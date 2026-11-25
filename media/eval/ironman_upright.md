Checkpoint `logs/r1_ironman_upright/2026-11-24_19-54-47/model_3595.pt` on `Humanoid-Fly-R1-IronMan-Upright-v1`, 512 robots x 12 s per condition, deterministic policy. Crash rate = robots that crashed at least once. Hover error / tilt / roll rate = last 1.5 s of each goal, robots that hadn't crashed. Goals reached = within 0.3 m at the goal switch. Limb speed = RMS joint speed while holding (same window); action change = mean |change| of each action per 20 ms step. Torso tilt and pose error (RMS over the controlled joints, vs. the upright target pose) while holding, Iron Man only.

| condition | crash rate | goals reached | hover error mean / p95 | tilt | roll rate (hover) | limb speed (hover) | action change (hover) | torso tilt (hover) | pose error (hover) | recovery |
|---|---|---|---|---|---|---|---|---|---|---|
| Standing take-off, random goals | 0.0% | 100% | 0.8 / 1.3 cm | 4.2 deg | 0.10 rad/s | 0.11 rad/s | 0.000 | 0.8 deg | 13.1 deg | - |
| Mid-air start (tilt, velocity, spin) | 0.0% | 100% | 0.8 / 1.3 cm | 4.3 deg | 0.10 rad/s | 0.11 rad/s | 0.000 | 0.8 deg | 13.1 deg | 100% in 0.86 s (median) |
| Velocity kicks 0.5-1.5 m/s every ~2 s | 0.0% | 99% | 6.5 / 22.9 cm | 3.8 deg | 0.12 rad/s | 0.40 rad/s | 0.022 | 1.1 deg | 13.1 deg | - |
| Thrust -10% | 0.0% | 100% | 3.4 / 3.7 cm | 3.3 deg | 0.09 rad/s | 0.10 rad/s | 0.000 | 1.0 deg | 12.5 deg | - |
| Thrust +10% | 0.0% | 100% | 2.5 / 3.1 cm | 4.8 deg | 0.11 rad/s | 0.11 rad/s | 0.000 | 0.5 deg | 13.5 deg | - |
| Observation noise (std 0.05) | 0.0% | 100% | 8.3 / 13.5 cm | 4.6 deg | 0.35 rad/s | 0.97 rad/s | 0.385 | 3.9 deg | 13.4 deg | - |
