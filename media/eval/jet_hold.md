Checkpoint `logs/r1_jet_hold/2026-11-24_17-48-26//model_1897.pt` on `Humanoid-Fly-R1-Jet-Hold-v0`, 512 robots x 12 s per condition, deterministic policy. Crash rate = robots that crashed at least once. Hover error / tilt / roll rate = last 1.5 s of each goal, robots that hadn't crashed. Goals reached = within 0.3 m at the goal switch. Limb speed = RMS joint speed while holding (same window); action change = mean |change| of each action per 20 ms step.

| condition | crash rate | goals reached | hover error mean / p95 | tilt | roll rate (hover) | limb speed (hover) | action change (hover) | recovery |
|---|---|---|---|---|---|---|---|---|
| Standing take-off, random goals | 0.0% | 100% | 1.1 / 2.0 cm | 19.2 deg | 0.43 rad/s | 0.11 rad/s | 0.140 | - |
| Mid-air start (tilt, velocity, spin) | 0.0% | 100% | 1.1 / 2.0 cm | 19.2 deg | 0.44 rad/s | 0.11 rad/s | 0.139 | 7% in 1.38 s (median) |
| Velocity kicks 0.5-1.5 m/s every ~2 s | 0.2% | 100% | 6.7 / 22.3 cm | 20.0 deg | 0.46 rad/s | 0.10 rad/s | 0.115 | - |
| Thrust -10% | 0.0% | 100% | 2.6 / 3.4 cm | 18.8 deg | 0.35 rad/s | 0.11 rad/s | 0.139 | - |
| Thrust +10% | 0.0% | 100% | 2.1 / 2.9 cm | 19.0 deg | 0.39 rad/s | 0.12 rad/s | 0.150 | - |
| Observation noise (std 0.05) | 10.5% | 3% | 62.5 / 98.9 cm | 25.1 deg | 0.83 rad/s | 0.21 rad/s | 0.792 | - |
