Checkpoint `logs/r1_ironman_hold/2026-11-24_18-39-09//model_2996.pt` on `Humanoid-Fly-R1-IronMan-Hold-v1`, 512 robots x 12 s per condition, deterministic policy. Crash rate = robots that crashed at least once. Hover error / tilt / roll rate = last 1.5 s of each goal, robots that hadn't crashed. Goals reached = within 0.3 m at the goal switch. Limb speed = RMS joint speed while holding (same window); action change = mean |change| of each action per 20 ms step.

| condition | crash rate | goals reached | hover error mean / p95 | tilt | roll rate (hover) | limb speed (hover) | action change (hover) | recovery |
|---|---|---|---|---|---|---|---|---|
| Standing take-off, random goals | 0.0% | 100% | 1.6 / 2.3 cm | 5.2 deg | 0.04 rad/s | 0.08 rad/s | 0.001 | - |
| Mid-air start (tilt, velocity, spin) | 0.0% | 100% | 1.6 / 2.3 cm | 5.2 deg | 0.04 rad/s | 0.08 rad/s | 0.001 | 100% in 1.00 s (median) |
| Velocity kicks 0.5-1.5 m/s every ~2 s | 0.0% | 99% | 6.4 / 23.1 cm | 5.9 deg | 0.13 rad/s | 0.48 rad/s | 0.025 | - |
| Thrust -10% | 0.0% | 100% | 5.0 / 5.9 cm | 7.9 deg | 0.04 rad/s | 0.09 rad/s | 0.001 | - |
| Thrust +10% | 0.0% | 100% | 4.0 / 5.0 cm | 4.8 deg | 0.05 rad/s | 0.09 rad/s | 0.001 | - |
| Observation noise (std 0.05) | 0.0% | 100% | 9.1 / 16.2 cm | 6.5 deg | 0.38 rad/s | 0.89 rad/s | 0.330 | - |
