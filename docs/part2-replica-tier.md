# Part 2: a replica tier on unmodified Netcode for Entities

Part 1 ends at the send side: one machine runs out of datagrams before it runs out of collision work. The usual answer is a replication tier between the authority and the clients. Star Citizen's replication layer and Unreal's multi-server replication proxy are the known shapes, but neither publishes numbers, and nothing like it existed for Unity's Netcode for Entities (NfE). This part builds one from stock NfE worlds and measures it.

## 1. The construction

A **replica** is one process holding two ordinary NfE worlds:

- a **client world** that joins the authority as if it were a player and receives the authority's snapshots, which become the replica's state feed;
- a **server world** that re-hosts the same ghosts and serves its own clients.

Nothing in the package is forked. The replica binds one tick offset between its two worlds, forwards its clients' commands up to the authority, and measures its own hop to the authority so it can lead commands by that amount. A native relay process sits on each edge so that round-trip time, jitter and loss can be injected per hop.

Every crowd ghost carries a per-entity seal stamped by the authority each tick (id, tick, quantised position) and checked again at the replica and at the end client.

## 2. What was measured

Thin clients (no rendering) plus a few scripted bots that predict their own character. One desktop first, then one rented Linux machine to confirm the numbers travel.

| Claim | Evidence, in headline form |
|---|---|
| A single NfE server world's sending wall is a few hundred full-fidelity clients at 30 Hz | at 800 clients the run survives only with a raised send queue, a p99 tick in the tens of milliseconds and well under half the snapshots per second per client |
| 10,000 moving ghosts at full fidelity cost about 12 Mbit/s per client | the price is the player's line, not the replica's CPU |
| The replica works and it is cheap for the authority | the authority's tick p99 falls from about 9 ms with 100 direct clients to about 2 ms with one replica, and stays around 4 ms with four replicas serving the same clients |
| The seal holds end to end | 0 mismatches over several hundred million checks at the replica and at the clients |
| A client behind a replica predicts its own character as well as a direct client | corrections equal to direct (a few millimetres) when three conditions hold, listed below; equal under 100 ms round trip, 20 ms jitter and 1% loss on the client hop |
| Burst in one tick is fine for state, not for spawns | 10,000 ghosts changing a field in the same tick fit the budget; 10,000 projectile ghosts spawned in one tick stall the server for several ticks. **Projectiles must be events, not ghosts.** |
| Replica memory is linear in connections | about 0.4 GB plus roughly 38 MB per connection at 10,000 ghosts with the default 32-entry snapshot history |
| The numbers travel | the same build on a rented Linux machine gave the same figures within noise, at a cost of a few cents |

The three conditions for prediction through a proxy:

1. player bodies outrank the crowd in snapshot priority;
2. the replica binds a single tick offset between its two worlds and leads commands by the measured hop;
3. the hop is measured per replica, live, not assumed.

Missing any one of them shows up as corrections many times larger than direct.

## 3. The hidden lead

Through a replica, a command reaches the authority later than when the client dials the authority directly, so the client predicts further ahead. The measured lead settles at a handful of ticks per replica plus a command slack, roughly 170 to 230 ms of extra confirmation time. Prediction still shows the player's own input within a frame on both paths, so the question is not "does my body react later" but "when is a misprediction corrected and when do others see my action". A bot-side metric (input to first displayed response, input to authoritative confirmation) is defined so the human test can be designed around it. No human has run that test yet.

## 4. Snapshot-history memory (`part2/memory-model`)

NfE keeps, for every connection and ghost chunk, a ring of past snapshots so the next snapshot can be delta-compressed against whichever one the client acknowledged. From the package source the per-connection cost is

```
per connection ≈ ghosts × history × (8 + snapshot bytes per ghost)
```

With the harness's crowd ghost that predicts about 39 MB per connection at 10,000 ghosts and 32 history entries. The measured slope across 20 to 100 connections was within a few percent of that. The consequence is the formula itself: 400 connections at 10,000 ghosts is over 15 GB of history alone.

Cutting the history to 6 entries divides the memory by five but covers only about 200 ms at 30 Hz, and a client whose whole history is unacknowledged stops receiving those chunks. The define trades memory for a cliff in round-trip time, confirmed with a control run at 250 ms.

The design that follows: when every connection receives the same values for the same ghost on the same tick, the history can be stored **once**, indexed by tick, plus one acknowledged tick per connection and ghost. Memory then goes from O(connections × history) to O(history): for 100 connections at 10,000 ghosts, from gigabytes to tens of megabytes, with no change in delta size. The model is in `memory_model.py`; the send-path change it needs is not in the package and is future work.

## 5. Replica tree (design, arithmetic on measured inputs)

A replica joins its parent exactly as it joins the authority, so every edge of a tree carries the same feed.

- Authority egress depends only on the number of first-tier replicas, not on the number of end clients.
- A replica's limit is CPU and memory per connection, around a hundred children each.
- Two tiers serve 10,000 to 40,000 end clients from one authority; a third tier is needed only above that.
- The lead at the deepest tier is the sum of the per-tier leads, which is the quantity most likely to break feel. Freshness ages by one tick per tier, and snapshot loss compounds per tier because a lost fragment loses the whole snapshot.
- Every tier can verify the same authority-stamped seal at no extra cost.

## 6. A keyed seal for relays outside the trust boundary (design)

The seal above is an unkeyed hash: it catches bugs and corruption, not an adversary. If replicas or relays ever run on machines the operator does not control, a plain MAC is not enough either, because every verifying client would hold the key and could forge for the others. The design uses delayed key disclosure (TESLA, RFC 4082): the authority seals each record with a key it reveals a few ticks later, clients buffer that many ticks of sealed records, render at once and flag on failure. The tag fits the 32-bit field the hash uses today, so no extra bandwidth. The estimated cost is well under a millisecond per 10,000 records per tick before vectorisation. This is a design with cost estimates from published cycle counts; the experiment has not been run.

## 7. Limits

Thin clients and scripted bots; no rendered client and no human through the replica. One authority per run. Relays inject impairment on loopback, so real last-mile behaviour is approximated. The Unity harness, builds and raw run files are not part of this repository; the memory model is, and it is standard-library Python.
