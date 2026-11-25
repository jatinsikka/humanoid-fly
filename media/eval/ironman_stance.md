Checkpoint `logs/r1_ironman_stance/2026-11-25_00-29-01//model_4194.pt` on `Humanoid-Fly-R1-IronMan-Stance-v1`, 512 robots x 12 s per condition, deterministic policy. Crash rate = robots that crashed at least once. Hover error / tilt / roll rate = last 1.5 s of each goal, robots that hadn't crashed. Goals reached = within 0.3 m at the goal switch. Limb speed = RMS joint speed while holding (same window); action change = mean |change| of each action per 20 ms step. Torso tilt and pose error (RMS over the controlled joints, vs. the upright target pose) while holding, Iron Man only.

| condition | crash rate | goals reached | hover error mean / p95 | tilt | roll rate (hover) | limb speed (hover) | action change (hover) | torso tilt (hover) | pose error (hover) | recovery |
|---|---|---|---|---|---|---|---|---|---|---|
| Standing take-off, random goals | 0.0% | 100% | 1.5 / 2.5 cm | 4.6 deg | 0.10 rad/s | 0.39 rad/s | 0.018 | 0.9 deg | 14.0 deg | - |
| Mid-air start (tilt, velocity, spin) | 0.0% | 100% | 1.6 / 2.5 cm | 4.6 deg | 0.10 rad/s | 0.39 rad/s | 0.018 | 0.9 deg | 14.0 deg | 100% in 0.84 s (median) |
| Velocity kicks 0.5-1.5 m/s every ~2 s | 0.0% | 99% | 6.9 / 22.9 cm | 4.5 deg | 0.13 rad/s | 0.62 rad/s | 0.038 | 1.3 deg | 14.3 deg | - |
| Thrust -10% | 0.0% | 100% | 3.8 / 4.7 cm | 4.7 deg | 0.10 rad/s | 0.29 rad/s | 0.013 | 1.2 deg | 14.1 deg | - |
| Thrust +10% | 0.0% | 100% | 2.0 / 3.3 cm | 4.6 deg | 0.11 rad/s | 0.44 rad/s | 0.022 | 0.7 deg | 13.9 deg | - |
| Observation noise (std 0.05) | 0.0% | 100% | 9.8 / 15.2 cm | 5.1 deg | 0.37 rad/s | 1.08 rad/s | 0.424 | 4.1 deg | 14.1 deg | - |
