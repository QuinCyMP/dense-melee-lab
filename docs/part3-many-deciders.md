# Part 3: several deciders for one fight

Part 2 moves the sending off the deciding machine. The deciding itself is still one process, and Part 1 says it grows linearly with the crowd. The question here is the one the literature calls impossible: when many players interact tightly in one area, can the *deciding* be split across machines without massive communication, latency or inconsistency?

Several deciders per *world* exist everywhere (regions per server, migrating entity authority, worker overlap zones). Several deciders per *fight at a point* did not, as far as we could find. These fixtures ask three things: which split of responsibility keeps the cross-decider share small (the partition study), whether D deciders give the same outcome as one (the equivalence study), and what the exchange costs in real processes (the barrier-cost study).

## 1. The fixture

A trace-driven simplification of the Part 1 harness, in numpy: a disc at about one body per square metre, 30 Hz, two teams in a front layout or mixed. Each tick: input from a recorded trace, move, crowd push (4 Jacobi iterations on 0.7 m discs), swing resolution, damage (120 per hit, 4,000 HP), death. A swing is 15 ticks with 4 active. Every number is a float sent as raw bits; quantising breaks bit-exactness.

Deciders are **bands** across the front line. A body belongs to the band it stands in at the start of the tick, and every decider holds read copies of bodies within a halo beyond its band.

## 2. Partition study: which split keeps the cross share small

Seven responsibility rules were compared at 2,000, 10,000 and 20,000 bodies with 8 to 64 deciders: square cells, overlapping circles, sticky per-entity owners, per-interaction pairs, and a spine-and-limbs role split. Cells cut across the front line put most hits across deciders. Bands cut *along* the front line (one band per stretch of the line, every decider holding both teams in its stretch) keep the cross share in the single digits at moderate D and grow it slowly with D. Pairs and roles have no cross writes by construction but pay for it elsewhere. The band rule is what the next two studies use.

## 3. Equivalence study: does D deciders equal one decider?

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

## 4. Barrier-cost study: what the barrier costs in real processes

Each decider is its own OS process (Python multiprocessing, spawn). They exchange through shared-memory mailboxes and synchronise with a spin barrier on shared counters, two barriers per tick (after the move, after the crowd). Four cost classes are timed per tick: compute, exchange, barrier wait, sequencer. Every tick is checked against the equivalence study's single-decider reference. 275 verified runs, D = 2, 4, 8, N = 500 to 20,000, 1,200 ticks each.

1. **Equivalence is exact in real processes**: 0 on hits, HP, positions and death ticks in all 275 runs, billions of body-ticks compared bit for bit. A 1 m halo breaks it at tick 1, so the check bites. The construction needs two barriers per tick, the halo, forward-to-owner attacker snapshots, and a hand-over record when a body changes band. No third round trip is needed for hits.
2. **The barrier mechanism is cheap; the barrier wait is not.** An empty spin barrier costs microseconds to tens of microseconds at D = 8. In the real run the wait is 4 to 25 times that: it is straggler time, from owned-body imbalance and per-tick compute noise, not the mechanism.
3. **Exchange** (packing and unpacking mailboxes) is a fraction of a millisecond; payload grows to about 100 kB per tick over all deciders at N = 20,000, D = 8, which is a few Mbit/s per decider. Most of the measured exchange bucket is Python overhead, not data movement.
4. **The sequencer is not a bottleneck.** With forward-to-owner, ordering is a local sort at the target's owner: p99 well under a millisecond. A central sequencer variant costs the same within noise and keeps equivalence at 0.
5. **Crossover.** Two deciders beat one from about N = 1,100, four from about 1,200, eight from about 2,700. At N = 20,000 the tick goes from about 20 ms (one decider) to 11.5 (two), 7.5 (four) and 6.4 ms (eight): 3.2× faster, 40% of the ideal split. Tail latency falls too: p99 from about 30 ms to under 10 ms at 20,000, because the slowest decider has fewer bodies. Below about 1,000 bodies one decider is as fast or faster.
6. **What limits the gain** at N = 20,000, D = 8: compute above the ideal (halo bodies are simulated twice, plus a per-tick numpy floor), then barrier wait, then exchange, then the slowest-decider effect, and the sequencer last.
7. **Against the native serial bound.** A native single decider is estimated to need tens of deciders at 10,000 to 20,000 bodies to fit the budget with zero overhead. The measured Python overhead at D = 8 is 3 to 9% of the budget, negligible against that, and it grows by roughly 0.16 ms per decider, so beyond about 30 deciders it stops being small (an extrapolation).

The chart in the README plots result 5.

## 4b. Beyond D = 8, and the band re-cut under load (issue #7)

Two questions were left open: how the overhead grows at D = 16 and 32, and what it costs to re-cut the bands when owned-body imbalance passes a threshold. Pass lines, written before the runs: equivalence 0 at D = 16 and 32 for N = 10,000 and 20,000, in one process and in real processes; equivalence 0 through every re-cut; and two negative controls that must fail (a 1 m halo at D = 32, and a re-cut that drops the dedupe keys).

Two machines. **Machine A**: Intel Core, Tiger Lake (CPUID family 6 model 140), 4 cores / 8 threads, laptop, Windows 11, Python 3.14.7, numpy 2.5.4. **Machine B**: a 2-core Linux VM (Intel Xeon 2.1 GHz), Python 3.13, numpy 2.5.3. Each machine records its own fight and compares against its own single-decider reference, so the fights differ slightly between machines (machine A at N = 10,000: 225,375 hits; machine B: 224,772) and only the equivalence results are compared across them.

### The re-cut protocol

Every decider writes its owned-alive count and a 1,024-bin histogram of its bodies' y positions into a census slot at the end of each tick. The slots alternate by tick parity, so a fast decider cannot overwrite a slot a slow one is still reading. After barrier 1 of the next tick every decider reads the same census. If owned max/mean exceeds the threshold, each decider sums the histograms and computes equal-population cuts by itself. The inputs are the same integers and the arithmetic is the same, so the cuts are identical bit for bit and nothing has to be broadcast. All deciders also assert, after the run, that they took the same decisions. The new cuts take effect one tick later. On that tick every decider exchanges with every other, not only its neighbours, because a body can land in a band that does not border the band it left. Hand-over records carry the body's state and its dedupe keys. A cooldown (150 ticks) separates two re-cuts. The census needs the global barrier.

### Results

**1. Equivalence is exact at D = 16 and 32** (measured).

| N | D | in one process (B) | real processes (A) | real processes (B) | cross-decider hits (B) | exchange bytes per tick, all deciders (A) |
|---|---|---|---|---|---|---|
| 10,000 | 16 | 0 | 0 | 0 | 13.0 % | 136,502 |
| 10,000 | 32 | 0 | 0 | 0 | 26.7 % | 277,055 |
| 20,000 | 16 | 0 | 0 | 0 | 8.6 % | 213,294 |
| 20,000 | 32 | 0 | 0 | 0 | 17.9 % | 431,081 |

"0" means 0 missing, extra and late hits, 0 bodies with an HP or position difference, and 0 differing death ticks, for 1,200 ticks. The in-process runs used both the stale-pose and the rewind variant. The negative control bites: a 1 m halo at D = 32 gives 189,251 missing and 150,785 extra hits at N = 10,000, and 284,942 / 222,815 at N = 20,000 (machine B, in one process). Runs: `n{N}_D{16,32}_t1200_h3_merge_w1` (A), `n{N}_D{16,32}_t1200_h3_merge_s1` (B), `n{N}_front_D{16,32}_dh0_dp0_owner-*` (B, in one process).

**2. The overhead slope beyond D = 8 is still unmeasured.** Every D = 16 and 32 run on both machines had more deciders than hardware threads, so their barrier wait (13 to 55 ms) measures processes queueing for a core, not the protocol. These runs carry `"oversubscribed": true`. Three parts of them do not depend on free cores and are reported (measured, machine A):

- Exchange bytes grow linearly in D: about 8.6 kB per decider per tick at N = 10,000 and 13.5 kB at N = 20,000. Machine B gives the same within 2 %.
- The exchange bucket (packing and unpacking) per decider grows from 0.65 to 0.76 to 0.98 ms for D = 8, 16, 32 at N = 10,000, roughly 0.014 ms per added decider. The 0.16 ms per decider of section 4 is therefore mostly barrier wait and straggler time, not data handling (model: arithmetic on these measurements).
- Compute per decider stops falling: 4.2, 3.3, 3.0 ms at N = 10,000 for D = 8, 16, 32, against an ideal split of 0.44 ms at D = 32. Halo bodies simulated twice and the per-tick numpy floor set it. Preemption can inflate a compute bucket on an oversubscribed machine, so read this as an upper bound.

**3. On machine A the split stops gaining at D = 4** (measured, gated suite, 3 clean runs each, median [min-max] of tick p50, ms):

| N | D = 1 | D = 2 | D = 4 | D = 8 | best speedup |
|---|---|---|---|---|---|
| 10,000 | 12.06 [11.8-15.0] | 7.64 [6.5-8.3] | 7.23 [6.4-7.5] | 7.68 [5.9-8.3] | 1.67x at D = 4 |
| 20,000 | 26.47 [26.2-26.9] | 16.90 [16.2-20.5] | 12.08 [11.4-16.7] | 13.55 [13.4-13.8] | 2.19x at D = 4 |

Machine A has 4 physical cores. At D = 8 two deciders share each core through hyperthreading, and compute per decider stays near 9 ms at N = 20,000 against an ideal of 3.3 ms. Physical cores, not hardware threads, set where the split stops paying. The `oversubscribed` flag compares D with threads and does not catch this. The neighbour barrier was not faster than the global barrier within the run-to-run spread. Runs: `n{N}_D{1,2,4,8}_t1200_h3_merge[_nb]_g{1,2,3}`.

**4. A band re-cut costs about one and a half ticks and keeps equivalence 0** (measured, machine A, threshold 1.15 on owned max/mean, cooldown 150 ticks). The one run with fewer deciders than threads, N = 10,000 and D = 8:

| | re-cut tick 838 | the 30 ticks before the decision |
|---|---|---|
| owned imbalance max/mean | 1.152 before, 1.003 after | rising from 1.00 at tick 1 |
| bodies handed over | 669 | 14 per tick (p50) |
| dedupe keys handed over | 663 | |
| exchange bytes | 67,765 | 55,584 (p50), +22 % |
| tick time, slowest decider | 8.5 ms; next tick 9.4 ms | 5.0 ms (p50), 12.4 ms (max) |

The re-cut tick is about 1.7 times a median tick for two ticks and stays below the worst tick of the 30 before it. Run: `n10000_D8_t1200_h3_merge_rc1.15_w1`.

Every other re-cut run also kept equivalence 0 (Table 7 of `barrier_cost.py --summarize`, 13 re-cuts across D = 8, 16, 32 and N = 10,000, 20,000 on machine A, 15 on machine B). Their timings are oversubscribed and are not quoted. Their counts are:

- A re-cut hands over 505 to 4,131 bodies and 333 to 2,300 dedupe keys, 5 to 66 times the usual per-tick hand-over.
- The bytes on a re-cut tick are 0 to 32 % above the median tick, and the relative extra shrinks as D grows, because the halo traffic grows with D and the hand-over does not.
- Imbalance grows faster with narrower bands. At N = 10,000 the first re-cut came at tick 838 for D = 8, 552 for D = 16 and 161 for D = 32. At D = 32 the 150-tick cooldown, not the threshold, decides how often re-cuts happen (6 re-cuts in 1,200 ticks; one fired at 1.185). At N = 20,000, D = 8 machine A never re-cut (maximum imbalance 1.147).

The negative control bites: the same re-cut with the dedupe keys dropped breaks equivalence (`n10000_D16_t1200_h3_merge_rc1.15_w1_neg`, and `n2000_D4_t300_h3_merge_rc1.03_check_neg`).

## 5. The protocol that follows (design)

A tick schedule for D deciders behind one replica, written against the measured rules: the target's owner decides every interaction, boundary positions exchanged inside every tick with a 3 m halo (6 m for area effects), hits on one target ordered by (swing tick, attacker id), back-writes to an attacker (a parry cutting a swing) forwarded to its owner and applied before any later phase of the same tick that reads it, a hand-over record with state and dedupe keys when a body changes band, and dual command delivery near band edges. Message shapes are fixed-size little-endian records with floats as raw bits. Lockstep has no slack: a lost or late exchange message stalls the tick and cannot be skipped. Ownership migration inside one tick is possible when the receiving decider already holds a read copy, which the halo guarantees.

## 6. Limits, read before quoting

- Python and numpy: absolute times are an upper bound and the exchange cost is mostly interpreter overhead. A native port would move every number except the equivalence.
- One machine. Deciders exchange through shared memory with a spin barrier; no network hop, no jitter, no loss between them. The equivalence study says any delay on positions breaks equivalence, so a real hop needs the uniform-latency construction or a lockstep transport, and that cost is unmeasured.
- One recorded fight per population and layout, one seed.
- The fixture is a simplification of Part 1's harness: no guard or parry in the barrier-cost fixture, no area effect, simplified crowd.
- Equivalence measured up to D = 32. Timing measured up to D = 8 only: the overhead slope beyond D = 8 needs a machine with at least 32 physical cores, and was not measured.
- Band re-cut: one timing measurement on one fight (N = 10,000, D = 8, machine A, a laptop whose clock varies with temperature). The census uses the global barrier; a neighbour-only barrier would need another way to agree on the decision. Three deciders meeting at a corner and migration cost in an engine remain open.

## 7. Reproduce

```bash
python part3/equivalence/replay_deciders.py --n 2000 --deciders 1,2,4,8 --delay 0,1,2
python part3/equivalence/replay_deciders.py --n 2000 --deciders 1 --delay 0 --variants perturb --perturb-k 20   # the noise floor
python part3/equivalence/replay_deciders.py --summarize

python part3/barrier-cost/barrier_cost.py --n 500,1000,2000,5000,10000,20000 --deciders 1,2,4,8
python part3/barrier-cost/barrier_cost.py --summarize

python part3/equivalence/replay_deciders.py --n 10000 --deciders 1,16,32 --delay 0 --variants owner-stale-seq,owner-rewind-seq
python part3/equivalence/replay_deciders.py --n 10000 --deciders 32 --delay 0 --variants owner-stale-seq-vm1   # must fail

python part3/barrier-cost/barrier_cost.py --n 10000 --deciders 1,2,4,8,16,32 --ticks 1200
python part3/barrier-cost/barrier_cost.py --n 10000 --deciders 8,16,32 --ticks 1200 --recut-thr 1.15
python part3/barrier-cost/barrier_cost.py --n 10000 --deciders 16 --ticks 1200 --recut-thr 1.15 --recut-neg   # must fail
```

Both scripts write one JSON per configuration under `raw/` beside themselves, refuse to overwrite, and print the tables from those files. Each configuration takes seconds at N = 2,000 and about a minute at N = 10,000 on a desktop. `barrier_cost.py` imports `replay_deciders.py` from the sibling folder for the world, the trace and the reference, so keep the two folders together.
