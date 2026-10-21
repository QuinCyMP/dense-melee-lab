"""Snapshot-history memory model: per-connection snapshot-history memory of Netcode for Entities, predicted from
the package source and compared with the Part 2 measurements; plus the shared-baseline alternative.

Source of the formula (com.unity.netcode@96783eeffb32, the package cache of the Unity test harness):
  Runtime/Snapshot/GhostChunkSerializationState.cs  CalculateSize(serializerDataSize, chunkCapacity) =
      DataPerChunkSize + SnapshotHistorySize * (align16(8 * capacity) + align16(serializerDataSize * capacity))
      DataPerChunkSize = align16(4 * (9 + H + ceil(H / 32)))         -- one such state per (connection, ghost chunk)
  Runtime/Snapshot/GhostCollectionSystem.cs  ghost snapshot size = align16(4 + change-mask bytes + enabled bytes)
      + sum over replicated components of align16(component snapshot size)
  Runtime/Variants/GhostTransformVariants.cs  LocalTransform default variant = Position float3 + Scale + Rotation (8 x 4 B)

Crowd ghost (the Unity test harness's crowd ghost, not published): LocalTransform, CellCoord (2 int),
NetPosition (float3), CrowdAction (byte, uint, int, uint, int3, uint), Health (2 int) only when CrowdHealth > 0.

    python -I part2/memory-model/memory_model.py
"""
import math


def a16(n):
    return (n + 15) & ~15


COMPONENTS = {                     # replicated bytes per component snapshot (every field stored as 4-byte ints)
    'header (tick + change mask)': 4 + 4,
    'LocalTransform (pos 3 + scale 1 + rot 4)': 8 * 4,
    'CellCoord (X, Z)': 2 * 4,
    'NetPosition (float3)': 3 * 4,
    'CrowdAction (Acting, ActionTick, Id, Tick, QPos 3, PosHash)': 8 * 4,
}
HEALTH = 2 * 4
GHOSTS = 10_000
ENTITY_B = 8


def snapshot_size(health=False):
    return sum(a16(v) for v in COMPONENTS.values()) + (a16(HEALTH) if health else 0)


def per_connection_mb(H, ghosts=GHOSTS, health=False, chunk_capacity=128):
    S = snapshot_size(health)
    chunks = math.ceil(ghosts / chunk_capacity)
    per_chunk = a16(4 * (9 + H + (H + 31) // 32)) + H * (a16(ENTITY_B * chunk_capacity) + a16(S * chunk_capacity))
    return chunks * per_chunk / 1e6


# measured (Part 2, authority and replica at 100 connections): process memory, GB
MEASURED = {
    'authority, 100 direct clients': {32: 4.17, 6: 1.08, 'conns': 100},
    'replica, 100 clients': {32: 4.15, 6: 1.16, 'conns': 100},
    'replica, 20 clients (4 x 20 run)': {32: 1.13, 6: 0.74, 'conns': 20},
}


def shared_history_mb(H, conns, ghosts=GHOSTS, health=False, chunk_capacity=128):
    """One time-indexed history per ghost chunk for all connections + per connection a last-acked tick per ghost."""
    S = snapshot_size(health)
    history = math.ceil(ghosts / chunk_capacity) * H * a16(S * chunk_capacity) / 1e6
    per_conn_acks = ghosts * 4 / 1e6                         # uint last acked tick per (connection, ghost)
    return history + conns * per_conn_acks


def main():
    S = snapshot_size()
    print(f'crowd ghost snapshot size {S} B (+16 with Health) · per ghost per history slot {S + ENTITY_B} B')
    for H in (32, 16, 6):
        print(f'H={H:>2}: predicted per connection {per_connection_mb(H):6.2f} MB (10,000 ghosts, capacity 128)')
    print()
    for name, m in MEASURED.items():
        slope_mb = (m[32] - m[6]) * 1000 / m['conns'] / 26            # MB per connection per history slot
        print(f'{name}: (mem@32 - mem@6) / conns / 26 slots = {slope_mb:.3f} MB per slot per connection '
              f'= {slope_mb * 1e6 / GHOSTS:.0f} B per ghost-slot (predicted {S + ENTITY_B})')
    r100, r20 = MEASURED['replica, 100 clients'], MEASURED['replica, 20 clients (4 x 20 run)']
    for H in (32, 6):
        slope = (r100[H] - r20[H]) * 1000 / 80
        print(f'replica slope between 20 and 100 connections at H={H}: {slope:.1f} MB per connection '
              f'(predicted {per_connection_mb(H):.1f}); intercept {r20[H] - 20 * slope / 1000:.2f} GB')
    print()
    for conns in (100, 400, 1000):
        for H in (32, 64):
            print(f'{conns:>5} connections H={H}: per-connection history {conns * per_connection_mb(H) / 1000:6.2f} GB · '
                  f'shared history + acked ticks {shared_history_mb(H, conns) / 1000:6.3f} GB')


if __name__ == '__main__':
    main()