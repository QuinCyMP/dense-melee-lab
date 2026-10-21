# Part 3: several deciders for one fight

Part 2 moves the sending off the deciding machine. The deciding itself is still one process, and Part 1 says it grows linearly with the crowd. The question here is the one the literature calls impossible: when many players interact tightly in one area, can the *deciding* be split across machines without massive communication, latency or inconsistency?

Several deciders per *world* exist everywhere (regions per server, migrating entity authority, worker overlap zones). Several deciders per *fight at a point* did not, as far as we could find. These fixtures ask three things: which split of responsibility keeps the cross-decider share small (R7a), whether D deciders give the same outcome as one (R7d), and what the exchange costs in real processes (R7c).

## 1. The fixture

A trace-driven simplification of the Part 1 harness, in numpy: a disc at about one body per square metre, 30 Hz, two teams in a front layout or mixed. Each tick: input from a recorded trace, move, crowd push (4 Jacobi iterations on 0.7 m discs), swing resolution, damage (120 per hit, 4,000 HP), death. A swing is 15 ticks with 4 active. Every number is a float sent as raw bits; quantising breaks bit-exactness.

Deciders are **bands** across the front line. A body belongs to the band it stands in at the start of the tick, and every decider holds read copies of bodies within a halo beyond its band.

## 2. R7a: which split keeps the cross share small

Seven responsibility rules were compared at 2,000, 10,000 and 20,000 bodies with 8 to 64 deciders: square cells, overlapping circles, sticky per-entity owners, per-interaction pairs, and a spine-and-limbs role split. Cells cut across the front line put most hits across deciders. Bands cut *along* the front line (one band per stretch of the line, every decider holding both teams in its stretch) keep the cross share in the single digits at moderate D and grow it slowly with D. Pairs and roles have no cross writes by construction but pay for it elsewhere. The band rule is what R7d and R7c use.

## 3. R7d: does D deciders equal one decider?

Replay one recorded fight through 1, 2, 4 and 8 deciders and diff every tick. Nearly 300 configurations.

**Divergence is exactly 0 on hits, HP, positions and death ticks, every tick for 40 seconds, for D = 2, 4 and 8**, at 2,000 and 10,000 bodies in both layouts, with cross-decider hit shares from under 1% to about 18%, when three conditions hold:

1. **The boundary exchange is a barrier inside the tick.** Read copies are current.
2. **The halo is at least 3 m.** Four crowd iterations reach 2.8 m; 2 m and below diverge at tick 1.
3. **Hits that land on one target in one tick are applied in the single decider's order**, (swing tick, attacker id). Without this, HP and positions still match but kill credit differs for a few dozen attackers.

The negative controls bite: a 1 m halo breaks everything at tick 1; so does skipping the ordering rule, in the way just described.

**Any delay of 1 or 2 ticks on boundary positions breaks the equivalence completely**, whatever the hit rule. The first position difference appears at tick 1, the first hit difference at tick 7, and by tick 300 nearly every body is more than a centimetre off. The size of that divergence equals the **noise floor**: run the single decider again with one body moved by one micrometre and the fight diverges by the same amount. So for a lethal dense fight there is no useful "acceptable bound" below exact. Only 0 or a different, statistically similar fight, and the delayed fight is measurably less lethal than the noise floor, which is a fairness problem, not just a correctness one.

A delay on the **hits only** (positions exact) is bounded while nobody can die, but a kill that lands late lets the dead target keep moving and swinging, and the fight decorrelates from about tick 60 to 90. Pose rewind (lag compensation) is required as well.

**One construction restores exactness under delay:** a uniform hit latency L = d with rewind, the ordering rule, and exact position exchange. Every hit, local ones included, takes effect L ticks after its swing. Compared with a single decider that has the same L, the result is 0 / 0 / 0 again for D = 2, 4, 8, both layouts, with and without deaths. The price is L × 33 ms on every hit, and it does not remove the need for current positions.

Honest caveat: exactness at delay 0 is largely determinism (same inputs into the same function). The informative results are which conditions are necessary, how much each missing condition costs, and the noise-floor yardstick.

## 4. R7c: what the barrier costs in real processes

Each decider is its own OS process (Python multiprocessing, spawn). They exchange through shared-memory mailboxes and synchronise with a spin barrier on shared counters, two barriers per tick (after the move, after the crowd). Four cost classes are timed per tick: compute, exchange, barrier wait, sequencer. Every tick is checked against the R7d single-decider reference. 275 verified runs, D = 2, 4, 8, N = 500 to 20,000, 1,200 ticks each.

1. **Equivalence is exact in real processes**: 0 on hits, HP, positions and death ticks in all 275 runs, billions of body-ticks compared bit for bit. A 1 m halo breaks it at tick 1, so the check bites. The construction needs two barriers per tick, the halo, forward-to-owner attacker snapshots, and a hand-over record when a body changes band. No third round trip is needed for hits.
2. **The barrier mechanism is cheap; the barrier wait is not.** An empty spin barrier costs microseconds to tens of microseconds at D = 8. In the real run the wait is 4 to 25 times that: it is straggler time, from owned-body imbalance and per-tick compute noise, not the mechanism.
3. **Exchange** (packing and unpacking mailboxes) is a fraction of a millisecond; payload grows to about 100 kB per tick over all deciders at N = 20,000, D = 8, which is a few Mbit/s per decider. Most of the measured exchange bucket is Python overhead, not data movement.
4. **The sequencer is not a bottleneck.** With forward-to-owner, ordering is a local sort at the target's owner: p99 well under a millisecond. A central sequencer variant costs the same within noise and keeps equivalence at 0.
5. **Crossover.** Two deciders beat one from about N = 1,100, four from about 1,200, eight from about 2,700. At N = 20,000 the tick goes from about 20 ms (one decider) to 11.5 (two), 7.5 (four) and 6.4 ms (eight): 3.2× faster, 40% of the ideal split. Tail latency falls too: p99 from about 30 ms to under 10 ms at 20,000, because the slowest decider has fewer bodies. Below about 1,000 bodies one decider is as fast or faster.
6. **What limits the gain** at N = 20,000, D = 8: compute above the ideal (halo bodies are simulated twice, plus a per-tick numpy floor), then barrier wait, then exchange, then the slowest-decider effect, and the sequencer last.
7. **Against the native serial bound.** A native single decider is estimated to need tens of deciders at 10,000 to 20,000 bodies to fit the budget with zero overhead. The measured Python overhead at D = 8 is 3 to 9% of the budget, negligible against that, and it grows by roughly 0.16 ms per decider, so beyond about 30 deciders it stops being small (an extrapolation).

The chart in the README plots result 5.

## 5. The protocol that follows (design)

A tick schedule for D deciders behind one replica, written against the measured rules: the target's owner decides every interaction, boundary positions exchanged inside every tick with a 3 m halo (6 m for area effects), hits on one target ordered by (swing tick, attacker id), back-writes to an attacker (a parry cutting a swing) forwarded to its owner and applied before any later phase of the same tick that reads it, a hand-over record with state and dedupe keys when a body changes band, and dual command delivery near band edges. Message shapes are fixed-size little-endian records with floats as raw bits. Lockstep has no slack: a lost or late exchange message stalls the tick and cannot be skipped. Ownership migration inside one tick is possible when the receiving decider already holds a read copy, which the halo guarantees.

## 6. Limits, read before quoting

- Python and numpy: absolute times are an upper bound and the exchange cost is mostly interpreter overhead. A native port would move every number except the equivalence.
- One machine. Deciders exchange through shared memory with a spin barrier; no network hop, no jitter, no loss between them. R7d says any delay on positions breaks equivalence, so a real hop needs the uniform-latency construction or a lockstep transport, and that cost is unmeasured.
- One recorded fight per population and layout, one seed.
- The fixture is a simplification of Part 1's harness: no guard or parry in R7c, no area effect, simplified crowd.
- D up to 8 measured; the protocol is D-general but the band re-cut under load, three deciders meeting at a corner, and migration cost in an engine are open.

## 7. Reproduce

```bash
python part3/r7d/r7d_replay.py --n 2000 --deciders 1,2,4,8 --delay 0,1,2
python part3/r7d/r7d_replay.py --n 2000 --deciders 1 --delay 0 --variants perturb --perturb-k 20   # the noise floor
python part3/r7d/r7d_replay.py --summarize

python part3/r7c/r7c_barrier.py --n 500,1000,2000,5000,10000,20000 --deciders 1,2,4,8
python part3/r7c/r7c_barrier.py --summarize
```

Both scripts write one JSON per configuration under `raw/` beside themselves, refuse to overwrite, and print the tables from those files. Each configuration takes seconds at N = 2,000 and about a minute at N = 10,000 on a desktop. `r7c_barrier.py` imports `r7d_replay.py` from the sibling folder for the world, the trace and the reference, so keep the two folders together.
