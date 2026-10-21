# Part 1: 200 fighters in a confined arena, one decider

The question: can a single server resolve a dense melee of 200 players, every hit decided on the server against multi-capsule hurtboxes, inside a 33 ms tick at 30 Hz? And where does one machine stop?

Two fixtures were built independently, in different languages and with different algorithms, on the same desktop (Intel i5-13400F, Windows 11). They agree.

## 1. The two fixtures

| | Native lab (`part1/native-lab`) | numpy harness (`part1/numpy-harness`) |
|---|---|---|
| Language | C++20, one simulation thread, Zig 0.16 toolchain, AVX2 | Python 3.12 + numpy, one pinned core |
| Bodies | 200 active every tick, 32 hurtboxes each (6,400), 3 weapon capsules each | 200 in two teams, 15-tick swing (6 windup, 4 active, 5 recover), four weapon shapes |
| Extras | 1,000 projectiles with continuous collision detection, area effect 200 × 199 pairs per tick, 8 temporal intervals per tick | lag-compensated sweeps rewound 1 to 9 ticks by attacker ping, area effect on a quarter of swings, 32 status slots per body, 2% ack loss |
| Network | 200 real UDP sockets in a second process, full state to everyone every tick | loopback UDP, delta against acknowledged baselines |
| Correctness | moving-sphere time of impact against the analytic answer (20,000 cases, 0 mismatches), capsule contacts against brute force (4,000 cases, 0 missed), 5.76 million packets complete | exhaustive float64 oracle on every recorded tick: 0 extra hits, 0 missing hits, 0 decode mismatches |
| Headline | tick p99 about 24 ms over a 3,600-tick soak (two minutes) with the network on | tick p99 about 26 ms in the worst case (dense blob, everyone swinging on the same tick) |

Both land inside the 33.3 ms budget with the same margin, from different code. That is the strongest claim in this part. It is a fixture claim, not a shipped-game claim: no engine, no rendered client, no human.

## 2. What the native lab taught, round by round

The lab was not built right the first time, and the failures are the useful part.

| Round | Change | Outcome |
|---|---|---|
| A | native CCD plus two-level bounds, 16 hurtboxes, 4 intervals | fast, 0 unresolved contacts |
| B | 32 hurtboxes, 8 intervals, denser start | still fast, **1 unresolved contact, a 29 mm overlap**. Failed correctness while passing the budget. |
| C | interval fallback fixed, crowd solver replaced | 0 unresolved, overlap 0.03 mm, still in budget |
| D | 200 UDP endpoints attached, projectiles as floats | **tick p99 37 ms, over budget in 15 ticks of 300.** A CPU pass does not mean a network pass. |
| E | projectiles quantised to 6 bytes, high-resolution pacing timer | back in budget, 1.44 million packets complete |
| F | two-minute soak | p99 24 ms, one tick over budget out of 3,600 (a desktop OS spike) |
| G | solver checked against an independent reference | 0 mismatches: the speed did not come from dropping collisions |
| I | a pose cache | compiled, then blocked by the OS's application control. Recorded as **unmeasured**, not as a speedup. |

Two lessons generalise. Report the maximum and the count of ticks over budget, not only the p99: the soak's single 36 ms spike would be invisible otherwise. And never disable a security control to make a benchmark run; record the number as unmeasured instead.

## 3. Techniques that measurably paid off

| Technique | Effect |
|---|---|
| Four-layer collision culling (root pairs, capsule pairs, sphere prefilter, exact kernel) | exact kernel tests cut by about 70%, hit stage p99 down by about a third, contact count unchanged |
| Candidate search radius widened by the history distance | targets that ran out of their cell between the swing and the rewound tick are still found; a counterexample killed the naive version |
| Delta encoding against groups of acknowledged baselines | 200 clients collapse to about a dozen distinct baselines, snapshot encode time falls by an order of magnitude, 0.2 to 0.3 Mbit/s per client for player state |
| Projectile records quantised to 6 bytes | datagrams per snapshot nearly halved, position error under 0.5 mm |
| Conservative-advancement CCD with a surface-exact fallback | unresolved contacts 1 to 0 |
| Projected Gauss-Seidel up to 64 iterations for the crowd | body overlap from several centimetres to 0.03 mm |
| A high-resolution waitable timer | tick start lateness p99 from up to 17 ms to under 1 ms |

The most important single finding: **how many points you sample along a blade is a content contract and the largest CPU lever after density.** Three sample points per tick leave a greatsword's edge tens of centimetres uncertain; twelve bring it to about 6 cm at roughly three times the hit cost. A single line between the blade tip's positions on two ticks misses a half-circle swing entirely (a counterexample, not an opinion). Bake the sample count per animation and let a tool check the tolerance every time an animation changes.

And the crowd solver is not cosmetic: one Jacobi iteration against eight raised the number of candidate pairs by 40% and the hit time by about half, because overlapping bodies look like more neighbours. A gate with two sides pushing in needs a rule, not just a solver.

## 4. Scaling and the single-machine ceiling

At constant density (about one body per square metre) the deciding work grows roughly linearly with N, because the number of bodies a swing can reach stays at about 13 to 15 whatever the crowd size. Only when every body stands on the same point does it become quadratic.

What grows faster is the sending side. One datagram per client per tick is one syscall each, at roughly 7 to 8 µs. By 800 mutually visible players that alone is around 20 ms of a 33 ms tick; by 1,600 it exceeds the tick on its own. Bandwidth per client grows with the number of visible players too (about 1.1 kbit/s per visible player per client in the harness).

| Mutually visible players | Deciding (native, one core, estimate) | Sending | Verdict |
|---|---|---|---|
| 200 | a few ms | trivial | comfortable at 30 Hz, probably 60 Hz |
| 400 | under 10 ms | a few ms | fine at 30 Hz |
| 800 | fits on several cores | tens of ms of syscalls | needs batched sends and a fast link |
| 1,600 | fits on several cores | exceeds the tick | needs network level of detail or more than one sending machine |

So the engineering ceiling of one machine for a dense arena at full fidelity sits around 400 to 800 players at 30 Hz, and it is the send side that ends it. Part 2 is about moving the send side off the deciding machine; Part 3 is about splitting the deciding itself.

Five bandwidth numbers exist for "one player", from 0.2 to 3.4 Mbit/s, and they differ because of what they assume (full state every tick versus deltas, players only versus projectiles, with or without events). The real figure for a shipped game sits between them. Keep all of them; do not quote one as a guarantee.

## 5. Claims that did not survive

Earlier reports on the same question, written before the fixtures existed, made claims the fixtures then tested. Keeping the failures visible is part of the method.

| Claim | What the fixture said |
|---|---|
| "10 ms per tick, proven" | the table behind it summed to 9 ms and had no executable, trace or baseline. An estimate presented as a measurement. |
| "One Jacobi iteration is stable, no oscillation" | one iteration overlapped bodies by 11 cm; 64 Gauss-Seidel iterations by 0.03 mm |
| "AVX-512 sweeps 16 capsule pairs per instruction" | 16 lanes is not a finished geometry solve. AVX2 adds 9 to 14% over SSE2 in the published Box2D numbers. The test machine has no AVX-512 and passed anyway. |
| "One encoded snapshot for everyone, 66 ms to 2 ms" | every client has a different acknowledged baseline, so identical bytes are impossible. Sharing per baseline group is possible, and is what the harness does. |
| "Resending UDP three times keeps a parry at 600 ms round trip" | burst loss takes all three copies, and priority removes no queueing or propagation delay |

## 6. Rules the measurements left behind

- Sample the blade sparsely only with a radius inflated to the measured tolerance; never a single line between two tick positions.
- Never cull candidates from current positions without widening by the history distance.
- Never drop a query when an iteration budget runs out; it is faster and it loses hits.
- Never encode a snapshot per client when baselines can be grouped; never send full state to everyone every tick.
- Never run one crowd iteration in a tight space.
- Give a dense fight its own core; two instances sharing a cache cost each other multiples.
- Do not allocate in the send path every tick.
- Do not cap targets, thin blade samples or shed players under load: the gameplay result then depends on how many people are present.
- Never report a p99 alone.
- Never call a 2D or loopback result a production test.

## 7. Shipped games as a yardstick

| Game | What it shows | What it does not show |
|---|---|---|
| New World (30 Hz, several simulation hubs, 2,500 per world) | a product decision to cap wars at 50 versus 50; the engine was not the stated limit | 200 in one spot |
| Photon Fusion BR200 sample (200 players at 60 Hz with lag compensation) | 200 live connections are ordinary; bare metal versus some VMs is a 3× difference | a shooter, not a melee blob |
| Valorant (server frame from 50 ms to under 2 ms) | the same work can get 25× cheaper by engineering; instances sharing a core interfere | 10 players per match |
| Black Desert Online (300 versus 300 node wars, 1,400-player sieges) | action combat with 600 in one battle ships | the field is a region, not an arena; no technical numbers published |
| Mount & Blade II: Bannerlord (official servers up to 600 players) | directional melee at 600 per server | tick rate and hit method undisclosed |
| Holdfast (150), Albion (300 to 350 per map), PlanetSide 2 (thousands with client-side hits), EVE (6,500 with time dilation) | scale ships in several ways, each with a trade | hit registration complaints, cheating exposure, or tab-target rules |

No shipped game we could find discloses "melee hitboxes, every hit server-decided, 200 in a confined arena, 30 Hz". That is the gap these fixtures measure.

## 8. Limits

One desktop, Windows scheduling, no noisy neighbours. Loopback or local sockets only: no real loss or jitter on the player's line. Synthetic geometry and bots that never stop swinging, with simplified guard and parry rules. The numpy harness has no SIMD, so its times are upper bounds. The native lab interpolates rotation between knots, tests rewind on one regression case, and has no death or despawn. Neither fixture has an engine, a client that renders 199 other players, or a human.

## 9. Reproduce

```bash
python part1/model/dense_combat_model.py            # 2D model and counterexamples, standard library only
python part1/numpy-harness/dense_melee_bench.py all  # about 15 minutes, writes a results markdown and JSON beside the script
```

The native lab needs the Zig 0.16 toolchain under `build/toolchain/` (the SHA-256 of the archive is in `build.cmd`) and an AVX2 CPU; `build.cmd` stamps the source hash and the runners refuse to start a binary whose hash does not match. Do not run the two fixtures at the same time; they compete for the same core.
