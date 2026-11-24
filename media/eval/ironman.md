Checkpoint `logs/r1_ironman_v1_smooth/2026-11-24_06-46-05/model_2597.pt` on `Humanoid-Fly-R1-IronMan-Smooth-v1`, 512 robots x 12 s per condition, deterministic policy. Crash rate = robots that crashed at least once. Hover error / tilt / roll rate = last 1.5 s of each goal, robots that hadn't crashed. Goals reached = within 0.3 m at the goal switch. Limb speed = RMS joint speed while holding (same window); action change = mean |change| of each action per 20 ms step.

| condition | crash rate | goals reached | hover error mean / p95 | tilt | roll rate (hover) | limb speed (hover) | action change (hover) | recovery |
|---|---|---|---|---|---|---|---|---|
| Standing take-off, random goals | 0.0% | 100% | 1.0 / 1.9 cm | 7.6 deg | 0.16 rad/s | 1.19 rad/s | 0.056 | - |
| Mid-air start (tilt, velocity, spin) | 0.0% | 100% | 1.0 / 1.8 cm | 7.5 deg | 0.16 rad/s | 1.19 rad/s | 0.056 | 100% in 1.18 s (median) |
| Velocity kicks 0.5-1.5 m/s every ~2 s | 0.2% | 100% | 4.6 / 17.6 cm | 6.5 deg | 0.26 rad/s | 1.26 rad/s | 0.063 | - |
| Thrust -10% | 0.0% | 100% | 3.3 / 4.3 cm | 7.7 deg | 0.16 rad/s | 1.19 rad/s | 0.055 | - |
| Thrust +10% | 0.0% | 100% | 2.4 / 3.4 cm | 7.0 deg | 0.18 rad/s | 1.16 rad/s | 0.056 | - |
| Observation noise (std 0.05) | 0.0% | 100% | 9.1 / 14.9 cm | 6.6 deg | 0.64 rad/s | 2.36 rad/s | 0.356 | - |
