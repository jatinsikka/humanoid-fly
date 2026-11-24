Checkpoint `logs/r1_jet_smooth/2026-11-24_09-30-07/model_1498.pt` on `Humanoid-Fly-R1-Jet-Smooth-v0`, 512 robots x 12 s per condition, deterministic policy. Crash rate = robots that crashed at least once. Hover error / tilt / roll rate = last 1.5 s of each goal, robots that hadn't crashed. Goals reached = within 0.3 m at the goal switch.

| condition | crash rate | goals reached | hover error mean / p95 | tilt | roll rate (hover) | recovery |
|---|---|---|---|---|---|---|
| Standing take-off, random goals | 0.0% | 100% | 1.1 / 1.6 cm | 22.6 deg | 1.30 rad/s | - |
| Mid-air start (tilt, velocity, spin) | 0.0% | 100% | 1.1 / 1.6 cm | 22.7 deg | 1.31 rad/s | 0% in nan s (median) |
| Velocity kicks 0.5-1.5 m/s every ~2 s | 0.2% | 100% | 5.3 / 18.9 cm | 24.0 deg | 1.27 rad/s | - |
| Thrust -10% | 0.0% | 100% | 1.7 / 2.4 cm | 23.0 deg | 1.32 rad/s | - |
| Thrust +10% | 0.0% | 100% | 2.1 / 2.8 cm | 22.5 deg | 1.31 rad/s | - |
| Observation noise (std 0.05) | 36.1% | 3% | 70.6 / 120.0 cm | 32.5 deg | 1.22 rad/s | - |
