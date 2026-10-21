# Related work

What exists, what it reached, and what this repository adds. Entries are limited to public sources; vendor claims are marked as such.

## Multi-stage replication (Part 2's shape)

| System | Shape | What is public | What is not |
|---|---|---|---|
| Star Citizen replication layer | game servers decide; a replication layer caches entity state; clients connect through a gateway | the shape, shard caps of a few hundred to a thousand (vendor) | per-node client counts, tick rate, bytes per client |
| Unreal Engine Iris | one process, pre-send quantisation shared by all connections, per-connection filter, priority and stream | the design; "up to 100 players per server" guidance | larger figures |
| Unreal MultiServerReplication | a proxy between backends and clients | the plugin | measurements |
| Improbable ScavLab / Otherside | custom; about 4,000 live players, 2 Hz updates for most avatars (press) | event sizes | internals |
| Hadean Aether Wars | space split into boxes assigned to machines; thousands of players with simplified gameplay | demo scale | cross-cell hit resolution, barrier cost |
| EVE Online (2025 cosmetic fan-out) | one serialised update fanned out over a message bus | message rates | physics stays at 1 Hz with time dilation |
| Source TV, Donnybrook (SIGCOMM 2008), Colyseus (NSDI 2006), the 2002 concentrator tier | relay chains, per-frame forwarding trees, primary and replica objects | papers and docs | none at full fidelity to thousands |

**Added here:** the same shape on an unmodified commercial netcode (Unity NfE) with numbers; the three conditions under which prediction through a proxy equals direct; a per-entity seal verified end to end; a measured per-connection memory figure and its formula; and a shared time-indexed baseline history that per-connection history can collapse into when everyone receives everything.

## Deciding across machines (Part 3's shape)

| Technique | What it offers | Why it is not this |
|---|---|---|
| LEARS (Raaen et al., 2012) | a lockless, relaxed-atomicity server that parallelises one spatial partition across cores | states that a tightly interacting area cannot be subdivided without massive communication, latency or inconsistency; this is the claim Part 3 tests |
| Deterministic lockstep RTS (Age of Empires, Supreme Commander) | peers run identical simulations on one ordered input log | replicates the whole simulation; it does not partition it |
| HLA conservative time management, DIS | ordered delivery with lookahead across federates | causal order, not bit equality; no single outcome by design in DIS |
| SpatialOS, Photon overlap zones, seamless-world papers (Assiotis and Tzanov 2006 and others) | one writer per entity component, async interaction across region borders | several deciders per world; interactions across a border are resolved case by case |
| Calvin (deterministic database) | replicas and partitions execute one global order in epochs | needs declared read and write sets; transactions that depend on reads need a reconnaissance pass |
| GROMACS domain decomposition, Anton | sub-millisecond steps across hundreds of nodes; bit repeats only with fixed hardware and rank count (GROMACS) or fixed-point arithmetic (Anton) | the discipline for partition-invariant reductions is borrowed from here |
| Box2D v3 | cross-platform determinism with thread-local results merged in a fixed order | an in-process technique, borrowed for the ordering rule |

**Added here:** a fixture in which several deciders resolve one fight at a point bit-identically to one decider, the three necessary conditions and the cost of each missing one, the noise-floor yardstick for "how different is different", a measured barrier and sequencer cost in separate OS processes, the crossover where D deciders beat one, and a construction (uniform hit latency with rewind) that keeps exactness under a bounded delay.

## Single-server dense combat (Part 1's shape)

Public figures from shipped games are listed in Part 1, section 7. None discloses server-decided melee hitboxes for 200 players in a confined arena at 30 Hz. The closest public artefacts are Photon's BR200 sample (200 at 60 Hz, a shooter) and Bannerlord's 600-player servers (tick rate undisclosed).

## Sources

- Raaen, Espeland, Stensland, Petlund, Halvorsen, Griwodz. *LEARS: A Lockless, Relaxed-Atomicity State Model for Parallel Execution of a Game Server Partition*. ICPP Workshops 2012.
- Bettner, Terrano. *1500 Archers on a 28.8: Network Programming in Age of Empires and Beyond*. GDC 2001.
- Fujimoto, Weatherly. *HLA Time Management and DIS*.
- Thomson, Diamond, Weng, Ren, Shao, Abadi. *Calvin: Fast Distributed Transactions for Partitioned Database Systems*. SIGMOD 2012.
- Bharambe et al. *Donnybrook: Enabling Large-Scale, High-Speed, Peer-to-Peer Games*. SIGCOMM 2008.
- Bharambe, Pang, Seshan. *Colyseus: A Distributed Architecture for Online Multiplayer Games*. NSDI 2006.
- Catto. *Determinism* (Box2D v3 blog post), 2024.
- Perrig, Canetti, Tygar, Song. *The TESLA Broadcast Authentication Protocol*; RFC 4082.
- Clark et al. *Live Migration of Virtual Machines*. NSDI 2005.
- Unity, *Netcode for Entities* manual: client and server worlds, prediction, serialization costs, snapshot history defines.
- Epic Games, *Introduction to Iris*, *Replication Graph*, *MultiServerReplication* documentation.
- Cloud Imperium Games, replication layer and server meshing communications.
