Checkpoint `checkpoints/jet/model_1499.pt` on `Humanoid-Fly-R1-Jet-Smooth-v0`, 512 robots x 12 s per condition, deterministic policy. Crash rate = robots that crashed at least once. Hover error / tilt / roll rate = last 1.5 s of each goal, robots that hadn't crashed. Goals reached = within 0.3 m at the goal switch.

| condition | crash rate | goals reached | hover error mean / p95 | tilt | roll rate (hover) | recovery |
|---|---|---|---|---|---|---|
| Standing take-off, random goals | 0.0% | 100% | 1.3 / 1.9 cm | 26.1 deg | 1.73 rad/s | - |
| Mid-air start (tilt, velocity, spin) | 0.0% | 100% | 1.3 / 1.9 cm | 26.3 deg | 1.71 rad/s | 0% in 3.42 s (median) |
| Velocity kicks 0.5-1.5 m/s every ~2 s | 0.2% | 100% | 6.0 / 21.2 cm | 26.8 deg | 1.57 rad/s | - |
| Thrust -10% | 0.0% | 100% | 1.8 / 2.7 cm | 25.5 deg | 1.62 rad/s | - |
| Thrust +10% | 0.0% | 100% | 2.5 / 3.2 cm | 26.5 deg | 1.77 rad/s | - |
| Observation noise (std 0.05) | 18.4% | 32% | 40.2 / 76.4 cm | 31.0 deg | 1.13 rad/s | - |
