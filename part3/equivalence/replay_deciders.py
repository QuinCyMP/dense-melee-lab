"""Equivalence study: does a fight decided by D deciders give the same outcome as one decider on the same input trace?

Question (docs/part3-many-deciders.md, equivalence study): replay one recorded input trace through
1 and D deciders, diff every tick; pass line: divergence 0 on hits, HP, positions (or an accepted bound).

What is simulated (a self-contained, trace-driven simplification of part1/numpy-harness -- see the
"Simplifications" list in docs/part3-many-deciders.md):
  world      the fixture's disc at ~1 person/m2, team = id & 1, 'front' layout (team 0 on x < 0, team 1 on x > 0) or the
             'mixed' layout; same seed and generator as part1/numpy-harness/dense_melee_bench.py
  tick       30 Hz. input -> move (4 m/s idle, 1.5 m/s while swinging) -> crowd push (4 Jacobi iterations, 0.7 m
             discs, disc walls: the fixture's stage_move) -> swing hit resolution -> damage (120 per hit, HP 4000)
  swing      15 ticks (6 windup, 4 active, 5 recover) as the fixture; during the 4 active ticks every living enemy
             within 2.5 m and inside the 120 deg arc in front of the attacker is hit, once per (attacker, target,
             swing) -- the fixture's per-pair dedupe (hit_matrix); death at HP <= 0 at the end of the tick (the body
             freezes and leaves the crowd). Several hits on one target in one batch: the first ceil(HP/120) count,
             the rest are dropped (overkill).
  trace      open loop per tick and actor: move vector (int16), yaw (uint8), swing intent (+ the target id the
             recorder chose, unused by the engine). Movement is a seeded bot script (run, reverse, stop, circle,
             zigzag segments of 20-90 ticks); the swing intent is recorded from the single-decider run (nearest
             enemy within 2.5 m and inside the 120 deg arc of the current facing, then snap the yaw to it and hold
             it for the swing). Replays apply the same trace whatever the state: the targets are never named.
  deciders   the best the partition study rule (docs/part3-many-deciders.md): circle_S = bands across the front line (cuts on y at equal
             population of the start positions), a body is owned by the band it stands in at the start of the
             tick, each decider is a circle around its band (half-diagonal + 3 m) that holds read copies.
             The target's decider resolves an interaction, else the attacker's, else any circle holding both
             (the circle margin means the first case always holds -- counted as `not_visible`).

Boundary exchange (relay delay d, in ticks; d = 0 is a barrier inside the tick):
  positions  after the move and after the crowd every decider publishes its owned bodies; a neighbour sees
             them d ticks old. The crowd of owned bodies runs on owned bodies + the read copies of the bodies within 5 m
             of the band (the 4 crowd iterations reach 4 x 0.7 m = 2.8 m) -- at d = 0 this reproduces the single
             decider bit for bit, which is checked, not assumed.
  hits       cross-decider swing = attacker and target owned by different deciders.
             owner-stale   forward-to-owner (forward-to-owner): the attacker's state travels d ticks; the target's decider tests
                           it against the target's pose at arrival (no rewind) and applies the hit on arrival.
             owner-rewind  forward-to-owner + lag compensation: the target's decider tests the attacker's state
                           against the target's pose of the swing tick (its own history) and applies on arrival.
             naive         the attacker's decider tests its own exact pose against the target copy that is d ticks
                           old, then sends the damage, which arrives d ticks later.
  order      seq = the cross events of one target are applied sorted by (swing tick, attacker id) -- the single
             decider's order; arr = arrival order (local hits first, then by source decider id, then attacker id).
             It matters only when a target takes more hits in a batch than its HP can bear (overkill: the hits
             after the killing one are dropped).
  latency L  uniform hit latency: every hit, local ones too, takes effect L ticks after the swing tick. The
             comparison reference then is the single decider with the same latency L (a different game, but
             one decider and D deciders agree).

Run from anywhere:
    python replay_deciders.py --n 2000 --deciders 1,2,4,8 --delay 0,1,2
    python replay_deciders.py --n 10000 --deciders 1,2,4 --delay 0,1
    python replay_deciders.py --n 2000 --deciders 2,4,8 --delay 0,1,2 --variants owner-stale-seq-p0,owner-stale-seq-h0
    python replay_deciders.py --summarize
Variant name = <mode>-<pose>-<order>[-uni][-p0][-h0]; p0: position exchange exact (only the hit relay delayed),
h0: hit relay exact (only position copies delayed), uni: latency L = delay. mode owner|naive, pose stale|rewind
(naive ignores it), order seq|arr. Reference arrays and traces are cached in --cache (default <tmp>/equivalence_cache).
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parent
RAW = Path(os.environ.get('EQUIV_RAW', OUT / 'raw'))

DT = 1.0 / 30.0
WINDUP, ACTIVE, RECOVER = 6, 4, 5
FRAMES = WINDUP + ACTIVE + RECOVER
BODY_R = 0.35
SKIN = 0.30
MIN_D = 2 * BODY_R
REACH = 2 * BODY_R + SKIN
CROWD_ITERS = 4
V_IDLE, V_SWING = 4.0, 1.5
HP0, DMG = 4000, 120
SWING_REACH = 2.5
ARC_COS = math.cos(math.radians(60.0))     # 120 degree arc = +-60 degrees
YAW_LOCK = 10                              # ticks the recorded bot holds the yaw after starting a swing
CIRCLE_MARGIN = 3.0
SEAM_M = 4.5                              # an attacker closer than this to a band boundary forwards its state
VIEW_MARGIN = 5.0                          # computing margin for the crowd view (>= 4 x 0.7 m + slack)
HRING = 8
SEED = 20261010
TICKS = 1200
KEY_SW = 4096                              # swing id range for hit keys


# ----------------------------------------------------------------------------- world
class World:
    def __init__(self, N, layout, per_m2=1.0, seed=SEED):
        self.N, self.layout = N, layout
        self.R = math.sqrt(N / (math.pi * per_m2))
        R = self.R
        self.team = (np.arange(N) & 1).astype(np.int8)
        if layout == 'front':
            r = np.random.default_rng(seed)
            side = np.where(self.team == 1, 1.0, -1.0)
            ang = r.uniform(-math.pi / 2, math.pi / 2, N)
            rad = R * np.sqrt(r.uniform(0, 1, N)) * 0.95
            self.px0 = side * np.abs(rad * np.cos(ang))
            self.py0 = rad * np.sin(ang)
        else:
            r = np.random.default_rng(seed + 1)
            ang = r.uniform(0, 2 * math.pi, N)
            rad = R * np.sqrt(r.uniform(0, 1, N)) * 0.95
            self.px0 = rad * np.cos(ang)
            self.py0 = rad * np.sin(ang)

    def bands(self, D):
        """circle_S: cuts on y at equal population of the start positions; returns cuts, y-range, circles."""
        cuts = np.quantile(self.py0, np.arange(1, D) / D) if D > 1 else np.empty(0)
        ylo = np.r_[-np.inf, cuts]
        yhi = np.r_[cuts, np.inf]
        R = self.R
        blo, bhi = np.maximum(ylo, -R), np.minimum(yhi, R)
        cen = np.column_stack([np.zeros(D), (blo + bhi) / 2])
        rad = np.hypot(R, (bhi - blo) / 2) + CIRCLE_MARGIN
        return cuts, ylo, yhi, cen, rad


def quant_yaw(ang):
    return ((((ang + math.pi) / (2 * math.pi) * 256).astype(np.int64)) & 255).astype(np.uint8)


def yaw_vec(yq):
    ang = yq.astype(np.float64) / 256 * (2 * math.pi) - math.pi
    return np.cos(ang), np.sin(ang)


# ----------------------------------------------------------------------------- crowd (the fixture's stage_move, float64)
def crowd(px, py, R):
    """Jacobi crowd push on a view; local index order == global id order, so the pair order is canonical."""
    n = len(px)
    px, py = px.copy(), py.copy()
    i = j = np.empty(0, np.int64)
    if n > 1:
        pr = cKDTree(np.column_stack([px, py])).query_pairs(REACH, output_type='ndarray')
        if len(pr):
            i = np.concatenate([pr[:, 0], pr[:, 1]]).astype(np.int64)
            j = np.concatenate([pr[:, 1], pr[:, 0]]).astype(np.int64)
            o = np.lexsort((j, i))
            i, j = i[o], j[o]
            dx, dy = px[i] - px[j], py[i] - py[j]
            keep = dx * dx + dy * dy < REACH * REACH
            i, j = i[keep], j[keep]
    lim = R - BODY_R
    for _ in range(CROWD_ITERS):
        if len(i):
            dx, dy = px[i] - px[j], py[i] - py[j]
            d2 = dx * dx + dy * dy
            ov = d2 < MIN_D * MIN_D
            ii, jj, dxx, dyy = i[ov], j[ov], dx[ov], dy[ov]
            d = np.sqrt(d2[ov])
            tiny = d < 1e-4
            if tiny.any():
                dxx = np.where(tiny, np.where(ii < jj, 1e-2, -1e-2), dxx)
                dyy = np.where(tiny, 0.0, dyy)
                d = np.where(tiny, 1e-2, d)
            pen = (MIN_D - d) * 0.5 / d
            cx = np.bincount(ii, weights=dxx * pen, minlength=n)
            cy = np.bincount(ii, weights=dyy * pen, minlength=n)
            w = 1.0 / np.sqrt(np.maximum(np.bincount(ii, minlength=n), 1))
            px += cx * w
            py += cy * w
        r = np.sqrt(px * px + py * py)
        k = np.where(r > lim, lim / np.maximum(r, 1e-6), 1.0)
        px *= k
        py *= k
    return px, py


def sweep(team, ax, ay, afc, afs, ateam, tids, tx, ty):
    """(attacker row, target id) pairs: enemy within SWING_REACH and inside the arc. tids/tx/ty = candidate targets."""
    if len(ax) == 0 or len(tids) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    m = cKDTree(np.column_stack([ax, ay])).sparse_distance_matrix(
        cKDTree(np.column_stack([tx, ty])), SWING_REACH + 1e-6, output_type='coo_matrix')
    r, c = m.row.astype(np.int64), m.col.astype(np.int64)
    tg = tids[c]
    dx, dy = tx[c] - ax[r], ty[c] - ay[r]
    d = np.sqrt(dx * dx + dy * dy)
    ok = (team[tg] != ateam[r]) & (d <= SWING_REACH) & ((afc[r] * dx + afs[r] * dy) >= ARC_COS * d)
    return r[ok], tg[ok]


# ----------------------------------------------------------------------------- input providers
class Recorder:
    """Bot scripts + swing intents, run against the single decider while it records the trace."""

    def __init__(self, world, T):
        N = world.N
        self.w, self.T = world, T
        self.rng = np.random.default_rng(SEED + 99)
        self.mx = np.zeros((T + 1, N), np.int16)
        self.my = np.zeros((T + 1, N), np.int16)
        self.yq = np.zeros((T + 1, N), np.uint8)
        self.sw = []
        self.seg_end = np.zeros(N, np.int64)
        self.start = np.zeros(N, np.int64)
        self.mode = np.zeros(N, np.int64)
        self.base = np.zeros(N)
        self.omega = np.zeros(N)
        self.mag = np.ones(N)
        self.yq_prev = np.zeros(N, np.uint8)
        self.lock_until = np.zeros(N, np.int64)
        self.lock_yq = np.zeros(N, np.uint8)

    def __call__(self, t, px, py, alive, act):
        w, r = self.w, self.rng
        N = w.N
        e = np.nonzero(self.seg_end <= t)[0]
        if e.size:
            n = e.size
            self.mode[e] = r.choice(5, size=n, p=[0.30, 0.15, 0.20, 0.15, 0.20])   # run reverse stop circle zigzag
            self.seg_end[e] = t + r.integers(20, 91, n)
            self.start[e] = t
            if w.layout == 'front':
                self.base[e] = np.where(w.team[e] == 0, 0.0, math.pi) + r.normal(0, 0.5, n)
            else:
                self.base[e] = r.uniform(0, 2 * math.pi, n)
            self.omega[e] = r.choice([-1.0, 1.0], n) * r.uniform(0.05, 0.2, n)
            self.mag[e] = r.uniform(0.5, 1.0, n)
        el = t - self.start
        zig = np.where((el // 8) % 2 == 0, 0.8, -0.8)
        ang = np.where(self.mode == 0, self.base,
              np.where(self.mode == 1, self.base + math.pi,
              np.where(self.mode == 3, self.base + self.omega * el, self.base + zig)))
        mag = np.where(self.mode == 2, 0.0, self.mag)
        mx = np.rint(np.cos(ang) * 32767 * mag).astype(np.int16)
        my = np.rint(np.sin(ang) * 32767 * mag).astype(np.int16)
        yq = np.where(mag > 0, quant_yaw(ang), self.yq_prev).astype(np.uint8)
        yq = np.where(self.lock_until > t, self.lock_yq, yq).astype(np.uint8)
        fcx, fsx = yaw_vec(yq)
        idle = alive & (t - act >= FRAMES)
        sa, st = [], []
        for c in (0, 1):
            cand = np.nonzero(idle & (w.team == c))[0]
            en = np.nonzero(alive & (w.team != c))[0]
            if cand.size == 0 or en.size == 0:
                continue
            dist, k = cKDTree(np.column_stack([px[en], py[en]])).query(
                np.column_stack([px[cand], py[cand]]), distance_upper_bound=SWING_REACH)
            ok = np.isfinite(dist)
            a, tg, d = cand[ok], en[k[ok]], dist[ok]
            dx, dy = px[tg] - px[a], py[tg] - py[a]
            arc = (fcx[a] * dx + fsx[a] * dy) >= ARC_COS * d
            a, tg, dx, dy = a[arc], tg[arc], dx[arc], dy[arc]
            if a.size:
                q = quant_yaw(np.arctan2(dy, dx))
                yq[a] = q
                self.lock_yq[a] = q
                self.lock_until[a] = t + YAW_LOCK
                sa.append(a)
                st.append(tg)
        sa = np.concatenate(sa) if sa else np.empty(0, np.int64)
        st = np.concatenate(st) if st else np.empty(0, np.int64)
        o = np.argsort(sa, kind='stable')
        sa, st = sa[o], st[o]
        self.yq_prev = yq.copy()
        self.mx[t], self.my[t], self.yq[t] = mx, my, yq
        self.sw.append(np.column_stack([np.full(sa.size, t), sa, st]).astype(np.int64))
        return mx, my, yq, sa, st

    def arrays(self):
        sw = np.concatenate(self.sw) if self.sw else np.empty((0, 3), np.int64)
        return dict(mx=self.mx, my=self.my, yq=self.yq, sw=sw)


class Replay:
    def __init__(self, tr):
        self.mx, self.my, self.yq, sw = tr['mx'], tr['my'], tr['yq'], tr['sw']
        self.sw = sw
        self.off = np.searchsorted(sw[:, 0], np.arange(0, len(self.mx) + 2))

    def __call__(self, t, px, py, alive, act):
        s = self.sw[self.off[t]:self.off[t + 1]]
        return self.mx[t], self.my[t], self.yq[t], s[:, 1], s[:, 2]


# ----------------------------------------------------------------------------- the engine
def simulate(world, T, inputs, cfg, ref=None, record=False, hp0=HP0):
    """One tick-by-tick run of D deciders over a trace. cfg keys: D dh dp mode pose order L."""
    N, R, team = world.N, world.R, world.team
    D, dh, dp, mode, pose, order, L = (cfg[k] for k in ('D', 'dh', 'dp', 'mode', 'pose', 'order', 'L'))
    H = HRING
    vm = cfg.get('vm', VIEW_MARGIN)
    px, py = world.px0.copy(), world.py0.copy()
    hp = np.full(N, hp0, np.int64)
    alive = np.ones(N, bool)
    death = np.full(N, -1, np.int64)
    act = np.full(N, -1000, np.int64)
    pd = np.empty(0, np.int64)       # (attacker, target, swing) pairs already hit
    own_hist = np.zeros((HRING, N), np.int64)
    yawi = np.zeros(N, np.uint8)
    pos_hist = np.empty((H, 2, N))
    mv_hist = np.empty((H, 2, N))
    alive_hist = np.ones((H, N), bool)
    pos_hist[:, 0, :], pos_hist[:, 1, :] = px, py
    mv_hist[:, 0, :], mv_hist[:, 1, :] = px, py
    if D > 1:
        cuts, ylo, yhi, cen, crad = world.bands(D)
    queue = {}
    hits = []                       # arrays (T_apply, tau, a, tgt, sw, cross)
    stats = dict(events=0, dropped=0, not_visible=0, hits=0, cross_hits=0)
    series = None
    if ref is not None:
        series = {k: np.zeros(T + 1) for k in ('pos_max', 'pos_n1cm', 'pos_nexact', 'hp_max', 'hp_n', 'alive0', 'alive1')}
    rec_pos = rec_hp = None
    if record:
        rec_pos = np.empty((T + 1, N, 2))
        rec_hp = np.empty((T + 1, N), np.int32)
        rec_pos[0, :, 0], rec_pos[0, :, 1], rec_hp[0] = px, py, hp0
    t0 = time.perf_counter()
    for t in range(1, T + 1):
        s = t % H
        own = np.searchsorted(cuts, py) if D > 1 else np.zeros(N, np.int64)
        alive_start = alive.copy()
        alive_hist[s] = alive_start
        # ---- input
        mxq, myq, yq, swa, swt = inputs(t, px, py, alive, act)
        yawi = np.where(alive, yq, yawi).astype(np.uint8)
        desx = np.where(alive, mxq.astype(np.float64) / 32767, 0.0)
        desy = np.where(alive, myq.astype(np.float64) / 32767, 0.0)
        if swa.size:
            can = alive[swa] & (t - act[swa] >= FRAMES)
            act[swa[can]] = t
        fc, fs = yaw_vec(yawi)
        # ---- move
        spd = np.where(t - act < FRAMES, V_SWING, V_IDLE)
        mvx = px + desx * spd * DT
        mvy = py + desy * spd * DT
        mv_hist[s, 0], mv_hist[s, 1] = mvx, mvy
        # ---- crowd, one view per decider
        if D == 1:
            idx = np.nonzero(alive)[0]
            nx, ny = crowd(mvx[idx], mvy[idx], R)
            npx, npy = mvx.copy(), mvy.copy()
            npx[idx], npy[idx] = nx, ny
        else:
            npx, npy = mvx.copy(), mvy.copy()
            sl = (t - dp) % H
            fx, fy, fal = mv_hist[sl, 0], mv_hist[sl, 1], alive_hist[sl]
            for d in range(D):
                owned = alive & (own == d)
                if not owned.any():
                    continue
                vis = fal & (own != d) & (fy >= ylo[d] - vm) & (fy <= yhi[d] + vm)
                vid = np.nonzero(owned | vis)[0]
                vx = np.where(owned, mvx, fx)[vid]
                vy = np.where(owned, mvy, fy)[vid]
                nx, ny = crowd(vx, vy, R)
                sel = owned[vid]
                npx[vid[sel]], npy[vid[sel]] = nx[sel], ny[sel]
        px, py = npx, npy
        pos_hist[s, 0], pos_hist[s, 1] = px, py
        # ---- hits: a sector sweep, every enemy in reach and arc once per swing (the fixture's per-pair dedupe)
        batch = []
        if pd.size:
            pd = pd[(pd % KEY_SW) >= t - 3 * FRAMES]
        own_hist[s] = own

        def sched(tau, a, tg, sw, src, cr, when):
            if when <= t:
                batch.append((tau, a, tg, sw, src, cr))
            else:
                queue.setdefault(when, []).append((tau, a, tg, sw, src, cr))

        def mark(a, tg, sw):
            nonlocal pd
            pd = np.concatenate([pd, (a * N + tg) * KEY_SW + sw])

        def arrive(ev):
            """owner-stale: an attacker snapshot (swing tick tau) reaches the deciders that hold targets it may hit;
            tested against the targets' pose now (no rewind)."""
            tau, a, sw, src, ax, ay, afc, afs = ev
            tid = np.nonzero(alive_start)[0]
            r, tg = sweep(team, ax, ay, afc, afs, team[a], tid, px[tid], py[tid])
            ok = own_hist[tau[r] % H, tg] != src[r]         # targets that were not on the attacker's decider then
            r, tg = r[ok], tg[ok]
            aa, ss = a[r], sw[r]
            keep = ~np.isin((aa * N + tg) * KEY_SW + ss, pd)
            r, tg, aa, ss = r[keep], tg[keep], aa[keep], ss[keep]
            if aa.size:
                o = own[tg]
                vis = np.hypot(ax[r] - cen[o, 0], ay[r] - cen[o, 1]) <= crad[o]
                stats['not_visible'] += int((~vis).sum())
                mark(aa, tg, ss)
                batch.append((tau[r], aa, tg, ss, src[r], np.ones(aa.size, bool)))

        for item in queue.pop(t, []):
            if len(item) == 6:
                batch.append(item)
            else:
                arrive(item)

        frame = t - act
        A = np.nonzero(alive_start & (frame >= WINDUP) & (frame < WINDUP + ACTIVE))[0]
        if A.size:
            stats['events'] += int(A.size)
            ax, ay, afc, afs, aswing = px[A], py[A], fc[A], fs[A], act[A]
            if mode == 'naive' and D > 1:                   # each attacker's decider searches its own view of the targets
                rr, tt = [], []
                sl = (t - dh) % H
                for d in range(D):
                    m = own[A] == d
                    if not m.any():
                        continue
                    mine = own == d
                    tal = np.where(mine, alive_start, alive_hist[sl])
                    tx = np.where(mine, px, pos_hist[sl, 0])
                    ty = np.where(mine, py, pos_hist[sl, 1])
                    tid = np.nonzero(tal)[0]
                    r, tg = sweep(team, ax[m], ay[m], afc[m], afs[m], team[A][m], tid, tx[tid], ty[tid])
                    rr.append(np.nonzero(m)[0][r])
                    tt.append(tg)
                r, tg = np.concatenate(rr), np.concatenate(tt)
            else:
                tid = np.nonzero(alive_start)[0]
                r, tg = sweep(team, ax, ay, afc, afs, team[A], tid, px[tid], py[tid])
            a, swv = A[r], aswing[r]
            keep = ~np.isin((a * N + tg) * KEY_SW + swv, pd)
            r, tg, a, swv = r[keep], tg[keep], a[keep], swv[keep]
            src = own[a]
            cr = own[a] != own[tg]
            tau = np.full(a.size, t, np.int64)
            loc = ~cr
            if loc.any():                                   # same decider: exact state, no relay
                mark(a[loc], tg[loc], swv[loc])
                sched(tau[loc], a[loc], tg[loc], swv[loc], src[loc], cr[loc], t + L)
            if cr.any() and mode == 'naive':                # attacker decides on the target copy, damage message follows
                c = cr
                o = own[a[c]]
                sl = (t - dh) % H
                vis = np.hypot(pos_hist[sl, 0][tg[c]] - cen[o, 0], pos_hist[sl, 1][tg[c]] - cen[o, 1]) <= crad[o]
                stats['not_visible'] += int((~vis).sum())
                mark(a[c], tg[c], swv[c])
                sched(tau[c], a[c], tg[c], swv[c], src[c], cr[c], t + max(L, dh))
            elif cr.any() and mode == 'owner' and pose == 'rewind':
                c = cr                                      # target pose of the swing tick, from its own history
                o = own[tg[c]]
                vis = np.hypot(px[a[c]] - cen[o, 0], py[a[c]] - cen[o, 1]) <= crad[o]
                stats['not_visible'] += int((~vis).sum())
                mark(a[c], tg[c], swv[c])
                sched(tau[c], a[c], tg[c], swv[c], src[c], cr[c], t + max(L, dh))
            if mode == 'owner' and pose == 'stale' and D > 1:
                near = np.abs(py[A][:, None] - cuts[None, :]).min(axis=1) <= SEAM_M
                if near.any():                              # snapshots go to the neighbours only when a seam is near
                    ev = (np.full(int(near.sum()), t, np.int64), A[near], aswing[near], own[A][near],
                          ax[near], ay[near], afc[near], afs[near])
                    if dh == 0:
                        arrive(ev)
                    else:
                        queue.setdefault(t + dh, []).append(ev)
        # ---- sequence and apply
        if batch:
            tau, a, tg, sw, src, cr = (np.concatenate([b[i] for b in batch]) for i in range(6))
            if order == 'seq':
                o = np.lexsort((tg, a, tau))
            else:
                o = np.lexsort((tg, a, tau, np.where(cr, src, -1), cr.astype(np.int64)))
            tau, a, tg, sw, src, cr = tau[o], a[o], tg[o], sw[o], src[o], cr[o]
            s2 = np.argsort(tg, kind='stable')
            tgs = tg[s2]
            first = np.r_[True, tgs[1:] != tgs[:-1]]
            ix = np.arange(len(tgs))
            rank = ix - np.maximum.accumulate(np.where(first, ix, 0))
            need = (np.maximum(hp[tgs], 0) + DMG - 1) // DMG
            sel = s2[rank < need]
            sel.sort()
            stats['dropped'] += int(len(tg) - len(sel))
            if sel.size:
                hp -= DMG * np.bincount(tg[sel], minlength=N)
                hits.append(np.column_stack([np.full(sel.size, t), tau[sel], a[sel], tg[sel], sw[sel], cr[sel]]).astype(np.int64))
                stats['hits'] += int(sel.size)
                stats['cross_hits'] += int(cr[sel].sum())
        newly = alive & (hp <= 0)
        if newly.any():
            alive[newly] = False
            death[newly] = t
        # ---- record / diff
        hpc = np.maximum(hp, 0)
        if record:
            rec_pos[t, :, 0], rec_pos[t, :, 1], rec_hp[t] = px, py, hpc
        if ref is not None:
            e = np.hypot(px - ref['pos'][t, :, 0], py - ref['pos'][t, :, 1])
            dh_ = np.abs(hpc - ref['hp'][t].astype(np.int64))
            series['pos_max'][t] = e.max()
            series['pos_n1cm'][t] = int((e > 0.01).sum())
            series['pos_nexact'][t] = int((e > 0).sum())
            series['hp_max'][t] = dh_.max()
            series['hp_n'][t] = int((dh_ > 0).sum())
            series['alive0'][t] = int((alive & (team == 0)).sum())
            series['alive1'][t] = int((alive & (team == 1)).sum())
    H_ = np.concatenate(hits) if hits else np.empty((0, 6), np.int64)
    res = dict(hits=dict(T=H_[:, 0], tau=H_[:, 1], a=H_[:, 2], tgt=H_[:, 3], sw=H_[:, 4], cross=H_[:, 5].astype(bool)),
               death=death, hp=np.maximum(hp, 0), stats=stats, series=series, wall_s=time.perf_counter() - t0,
               final_pos=np.column_stack([px, py]))
    if record:
        res['rec_pos'], res['rec_hp'] = rec_pos, rec_hp
    return res


# ----------------------------------------------------------------------------- cache of trace and reference
def case_key(N, layout, T, hp0):
    return f'v2_n{N}_{layout}_t{T}_hp{hp0}_s{SEED}'


def save_ref(path, res):
    np.save(str(path) + '.pos.npy', res['rec_pos'])
    np.save(str(path) + '.hp.npy', res['rec_hp'])
    h = res['hits']
    np.savez(str(path) + '.hits.npz', death=res['death'], **h)
    (Path(str(path) + '.stats.json')).write_text(json.dumps(res['stats']))


def load_ref(path):
    z = np.load(str(path) + '.hits.npz')
    return dict(pos=np.load(str(path) + '.pos.npy', mmap_mode='r'), hp=np.load(str(path) + '.hp.npy', mmap_mode='r'),
                death=z['death'], hits={k: z[k] for k in ('T', 'tau', 'a', 'tgt', 'sw', 'cross')},
                stats=json.loads(Path(str(path) + '.stats.json').read_text()))


def ref_cfg(L):
    return dict(D=1, dh=0, dp=0, mode='owner', pose='rewind', order='seq', L=L)


def prepare(N, layout, T, hp0, cache, Ls):
    cache.mkdir(parents=True, exist_ok=True)
    key = case_key(N, layout, T, hp0)
    tpath = cache / (key + '.trace.npz')
    world = World(N, layout)
    log = {}
    if not tpath.exists() or not (cache / (key + '_L0.hits.npz')).exists():
        t0 = time.time()
        rec = Recorder(world, T)
        res = simulate(world, T, rec, ref_cfg(0), record=True, hp0=hp0)
        np.savez(tpath, **rec.arrays())
        save_ref(cache / (key + '_L0'), res)
        # the replay of the trace through the same single decider must reproduce the recording bit for bit
        tr = dict(np.load(tpath))
        chk = simulate(world, T, Replay(tr), ref_cfg(0), ref=load_ref(cache / (key + '_L0')), hp0=hp0)
        ser = chk['series']
        assert ser['pos_nexact'].max() == 0 and ser['hp_n'].max() == 0, 'replay of the recording is not deterministic'
        log['record_wall_s'] = round(time.time() - t0, 1)
        print(f'[prepare] {key}: recorded + replay check ok ({log["record_wall_s"]} s), hits {res["stats"]["hits"]}, '
              f'deaths {(res["death"] >= 0).sum()}', flush=True)
    tr = None
    for L in Ls:
        if L and not (cache / (key + f'_L{L}.hits.npz')).exists():
            tr = tr or dict(np.load(tpath))
            res = simulate(world, T, Replay(tr), ref_cfg(L), record=True, hp0=hp0)
            save_ref(cache / (key + f'_L{L}'), res)
            print(f'[prepare] {key} reference with hit latency {L}: hits {res["stats"]["hits"]}', flush=True)
    return key


# ----------------------------------------------------------------------------- diff
def pctl(v, q):
    v = np.asarray(v)
    return float(np.percentile(v, q)) if len(v) else 0.0


def summarize_series(s):
    return dict(p50=pctl(s, 50), p99=pctl(s, 99), max=float(np.max(s)) if len(s) else 0.0)


def first_nonzero(s):
    nz = np.nonzero(np.asarray(s)[1:] > 0)[0]
    return int(nz[0] + 1) if nz.size else None


def diff_hits(rh, uh, N, T):
    rk = (rh['a'] * N + rh['tgt']) * KEY_SW + rh['sw']
    uk = (uh['a'] * N + uh['tgt']) * KEY_SW + uh['sw']
    rs, us = np.argsort(rk), np.argsort(uk)
    rks, uks = rk[rs], uk[us]
    if len(rks):
        p = np.minimum(np.searchsorted(rks, uks), len(rks) - 1)
        in_ref = rks[p] == uks
    else:
        p = np.zeros(len(uks), np.int64)
        in_ref = np.zeros(len(uks), bool)
    missing = len(rks) - int(in_ref.sum())
    extra = int((~in_ref).sum())
    rT = rh['T'][rs][p][in_ref]
    uT = uh['T'][us][in_ref]
    late, early = int((uT > rT).sum()), int((uT < rT).sum())
    lateness = (uT - rT)[uT > rT]
    # per tick symmetric difference of (hit, apply tick)
    ck_r = rk * 2048 + rh['T']
    ck_u = uk * 2048 + uh['T']
    x = np.setxor1d(ck_r, ck_u)
    per_tick = np.bincount((x % 2048).astype(np.int64), minlength=T + 1)[:T + 1]
    # which side the mismatches fall on
    miss_mask = np.ones(len(rks), bool)
    if len(rks):
        miss_mask[np.searchsorted(rks, uks[in_ref])] = False
    miss_cross = int(rh['cross'][rs][miss_mask].sum())
    extra_cross = int(uh['cross'][us][~in_ref].sum())
    late_cross = int(uh['cross'][us][in_ref][uT > rT].sum())
    denied = np.bincount(rh['tgt'][rs][miss_mask], minlength=N)       # hits a target should have taken and did not
    unearned = np.bincount(uh['tgt'][us][~in_ref], minlength=N)       # hits a target took that one decider never gave
    return dict(missing=missing, extra=extra, late=late, early=early, late_ticks_max=int(lateness.max()) if lateness.size else 0,
                late_ticks_mean=float(lateness.mean()) if lateness.size else 0.0,
                missing_cross=miss_cross, extra_cross=extra_cross, late_cross=late_cross,
                matched=int(in_ref.sum()) - late - early, ref_hits=len(rks), run_hits=len(uks),
                players_denied_a_hit=int((denied > 0).sum()), players_took_unearned_hit=int((unearned > 0).sum()),
                players_any_mismatch=int(((denied + unearned) > 0).sum())), per_tick, denied + unearned


def dump_run(path, summary, config, series):
    """summary and config readable, the per-tick series compact (one array per line)"""
    path.write_text('{\n"summary": ' + json.dumps(summary, indent=1) + ',\n"config": ' + json.dumps(config) +
                    ',\n"series": {\n' + ',\n'.join(f'"{k}": ' + json.dumps(v, separators=(',', ':')) for k, v in series.items()) +
                    '\n}\n}\n')


def t_half(death, N):
    """tick at which half of all players are dead (None when fewer than half die)"""
    d = np.sort(death[death >= 0])
    return int(d[N // 2 - 1]) if len(d) >= N // 2 else None


def run_config(task):
    N, layout, T, hp0, cache, cfg, variant, force = task
    name = f'n{N}_{layout}_D{cfg["D"]}_dh{cfg["dh"]}_dp{cfg["dp"]}_{variant}' + ('' if hp0 == HP0 else f'_hp{hp0}')
    path = RAW / (name + '.json')
    if path.exists() and not force:
        return json.loads(path.read_text())['summary']
    key = case_key(N, layout, T, hp0)
    world = World(N, layout)
    world.px0[cfg.get('pbody', 0)] += cfg.get('perturb', 0.0)          # noise-floor control: one body starts 1 micrometre away
    tr = dict(np.load(cache / (key + '.trace.npz')))
    ref = load_ref(cache / (key + f'_L{cfg["L"]}'))
    res = simulate(world, T, Replay(tr), cfg, ref=ref, hp0=hp0)
    ser = res['series']
    hd, hit_tick, per_target = diff_hits(ref['hits'], res['hits'], N, T)
    fin = res['final_pos']
    ref_fin = np.asarray(ref['pos'][T])
    end_err = np.hypot(*(fin - ref_fin).T)
    death_diff = int((res['death'] != ref['death']).sum())
    hp_final_diff = int((res['hp'] != ref['hp'][T].astype(np.int64)).sum())
    st = res['stats']
    hit_t = hit_tick[1:]
    summary = dict(
        name=name, N=N, layout=layout, hp0=hp0, D=cfg['D'], dh=cfg['dh'], dp=cfg['dp'], variant=variant, latency=cfg['L'],
        hits=hd, hits_sym_per_tick=summarize_series(hit_t), hit_tick_diff_total=int(hit_t.sum()),
        hp_max_diff=summarize_series(ser['hp_max'][1:]), hp_bodies_differ=summarize_series(ser['hp_n'][1:]),
        pos_max_diff_m=summarize_series(ser['pos_max'][1:]), pos_bodies_gt_1cm=summarize_series(ser['pos_n1cm'][1:]),
        pos_bodies_nonexact=summarize_series(ser['pos_nexact'][1:]),
        first_divergence_tick=dict(hits=first_nonzero(hit_tick), hp=first_nonzero(ser['hp_n']),
                                   pos=first_nonzero(ser['pos_nexact']), pos_gt_1cm=first_nonzero(ser['pos_n1cm'])),
        end_pos_max_m=float(end_err.max()), end_pos_bodies_gt_1cm=int((end_err > 0.01).sum()),
        players_death_tick_differs=death_diff, players_final_hp_differs=hp_final_diff,
        players_with_hit_mismatch=int((per_target > 0).sum()),
        t_half=t_half(res['death'], N), ref_t_half=t_half(ref['death'], N),
        deaths=int((res['death'] >= 0).sum()), ref_deaths=int((ref['death'] >= 0).sum()),
        survivors_by_team=[int(((res['death'] < 0) & (world.team == c)).sum()) for c in (0, 1)],
        ref_survivors_by_team=[int(((ref['death'] < 0) & (world.team == c)).sum()) for c in (0, 1)],
        cross_share_hits=st['cross_hits'] / max(st['hits'], 1), 
        hits_total=st['hits'], dropped_overkill=st['dropped'], ref_dropped_overkill=ref['stats']['dropped'],
        not_visible=st['not_visible'], wall_s=round(res['wall_s'], 1))
    dump_run(path, summary, dict(N=N, layout=layout, T=T, hp0=hp0, **cfg),
             {k: [round(float(x), 6) for x in v[1:]] for k, v in dict(hit_sym=hit_tick, **ser).items()})
    return summary


def variant_cfg(variant, D, delay, dh_override=None):
    parts = variant.split('-')
    if parts[0] == 'naive':
        parts.insert(1, 'copy')
    mode, pose, order = parts[0], parts[1], parts[2]
    flags = set(parts[3:])
    dh = 0 if 'h0' in flags else delay
    dp = 0 if 'p0' in flags else delay
    L = delay if 'uni' in flags else 0
    cfg = dict(D=D, dh=dh, dp=dp, mode=mode, pose=pose, order=order, L=L)
    for f in flags:
        if f.startswith('vm'):
            cfg['vm'] = float(f[2:])                 # crowd view margin override (metres)
    return cfg


DEFAULT_VARIANTS = 'owner-stale-seq,owner-stale-arr,owner-rewind-seq,naive-seq,naive-arr'


def set_raw(path):
    global RAW
    RAW = Path(path)
    os.environ['EQUIV_RAW'] = str(RAW)


def harness_running():
    try:
        out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq Harness*'], capture_output=True, text=True, timeout=20).stdout
        return any(l.strip().lower().startswith('harness') for l in out.splitlines())
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--layout', default='front', choices=('front', 'mixed'))
    ap.add_argument('--ticks', type=int, default=TICKS)
    ap.add_argument('--hp', type=int, default=HP0)
    ap.add_argument('--deciders', default='1,2,4,8')
    ap.add_argument('--delay', default='0,1,2')
    ap.add_argument('--variants', default=DEFAULT_VARIANTS)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--cache', default=str(Path(tempfile.gettempdir()) / 'equivalence_cache'))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--perturb-k', type=int, default=1, help='number of 1-micrometre perturbation controls when the variant list has `perturb`')
    ap.add_argument('--raw', default=str(RAW), help='directory for the per-configuration JSON')
    ap.add_argument('--summarize', action='store_true')
    args = ap.parse_args()
    set_raw(args.raw)
    if args.summarize:
        return summarize()
    if harness_running():
        args.procs = min(args.procs, 2)
        print('Harness process running: limiting to 2 processes', flush=True)
    RAW.mkdir(exist_ok=True)
    cache = Path(args.cache)
    Ds = [int(x) for x in args.deciders.split(',')]
    delays = [int(x) for x in args.delay.split(',')]
    variants = args.variants.split(',')
    tasks = []
    Ls = {0}
    if 'perturb' in variants:
        variants.remove('perturb')
        for k in range(args.perturb_k):
            tasks.append((args.n, args.layout, args.ticks, args.hp, cache,
                          dict(ref_cfg(0), perturb=1e-6, pbody=(k * 97 + 13) % args.n), f'perturb-1um-k{k}', args.force))
    for D in Ds:
        if D == 1:
            continue
        for v in variants:
            for d in delays:
                cfg = variant_cfg(v, D, d)
                Ls.add(cfg['L'])
                tasks.append((args.n, args.layout, args.ticks, args.hp, cache, cfg, v, args.force))
    prepare(args.n, args.layout, args.ticks, args.hp, cache, sorted(Ls))
    # the single-decider replay is the D = 1 row: it must equal the recording exactly
    if 1 in Ds:
        tasks.insert(0, (args.n, args.layout, args.ticks, args.hp, cache, ref_cfg(0), 'ref-replay', args.force))
    t0 = time.time()
    print(f'{len(tasks)} configurations, {args.procs} processes', flush=True)
    with mp.Pool(args.procs) as pool:
        for s in pool.imap_unordered(run_config, tasks):
            print(f'{s["name"]:62s} hits miss/extra/late {s["hits"]["missing"]}/{s["hits"]["extra"]}/{s["hits"]["late"]}  '
                  f'hp max {s["hp_max_diff"]["max"]:.0f}  pos max {s["pos_max_diff_m"]["max"]:.3g}  '
                  f'cross {100 * s["cross_share_hits"]:.1f}%  {s["wall_s"]} s', flush=True)
    print(f'done in {time.time() - t0:.0f} s')


def load_all():
    out = []
    for p in sorted(RAW.glob('*.json')):
        j = json.loads(p.read_text())
        out.append((j['summary'], j['series']))
    return out


def fmt_first(fd):
    f = lambda v: '-' if v is None else str(v)
    return f'{f(fd["hits"])}/{f(fd["hp"])}/{f(fd["pos"])}'


def row_md(label, r):
    h = r['hits']
    return (f'| {label} | {100 * r["cross_share_hits"]:.1f} | {h["ref_hits"]} | {h["missing"]} | {h["extra"]} | {h["late"]} | {h["early"]} | '
            f'{r["hp_max_diff"]["max"]:.0f} | {r["pos_max_diff_m"]["max"]:.3g} | {fmt_first(r["first_divergence_tick"])} | '
            f'{r["players_death_tick_differs"]} | {r["players_final_hp_differs"]} |')


HEAD = ('| config | cross hits % | ref hits | missing | extra | late | early | HP diff max | pos diff max (m) | '
        'first diff tick hits/HP/pos | players death tick differs | players final HP differs |\n'
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|')


def summarize():
    rows = load_all()
    groups = {}
    for r, ser in rows:
        groups.setdefault((r['N'], r['layout'], r.get('hp0', HP0)), []).append((r, ser))
    for (N, layout, hp0), items in sorted(groups.items()):
        print(f'\n## N = {N} / {layout} / HP {hp0}\n')
        by_var = {}
        for r, ser in items:
            if r['variant'].startswith('perturb'):
                continue
            by_var.setdefault(r['variant'], []).append((r, ser))
        for v, lst in sorted(by_var.items()):
            print(f'\n### {v}\n')
            print(HEAD)
            for r, ser in sorted(lst, key=lambda x: (x[0]['D'], x[0]['dh'], x[0]['dp'])):
                print(row_md(f'D={r["D"]} dh={r["dh"]} dp={r["dp"]}', r))
        per = [(r, ser) for r, ser in items if r['variant'].startswith('perturb')]
        if per:
            print(f'\n### noise floor: {len(per)} runs of the single decider with one body moved 1 micrometre\n')
            print(HEAD)
            for r, ser in per[:3]:
                print(row_md(r['variant'], r))
            q = lambda f: (min(f(r) for r, _ in per), float(np.median([f(r) for r, _ in per])), max(f(r) for r, _ in per))
            print('\n| outcome | ref | noise floor min / median / max |\n|---|---:|---|')
            rr = per[0][0]
            for name, ref, f in (('hits total', rr['hits']['ref_hits'], lambda r: r['hits_total']),
                                 ('deaths', rr['ref_deaths'], lambda r: r['deaths']),
                                 ('survivors team 0', rr['ref_survivors_by_team'][0], lambda r: r['survivors_by_team'][0]),
                                 ('survivors team 1', rr['ref_survivors_by_team'][1], lambda r: r['survivors_by_team'][1]),
                                 ('tick of half the players dead', rr['ref_t_half'], lambda r: r['t_half'] or -1),
                                 ('missing hits', 0, lambda r: r['hits']['missing']),
                                 ('extra hits', 0, lambda r: r['hits']['extra']),
                                 ('first diff tick (hits)', None, lambda r: r['first_divergence_tick']['hits'] or -1),
                                 ('first diff tick (pos > 1 cm)', None, lambda r: r['first_divergence_tick']['pos_gt_1cm'] or -1)):
                lo, med, hi = q(f)
                print(f'| {name} | {ref} | {lo} / {med:.0f} / {hi} |')
        print('\n### outcome and horizon\n')
        print('| config | deaths | survivors t0/t1 | tick half dead | hit diffs ticks 1-60 | hit diffs 1-300 | bodies >1 cm at tick 60 | at 300 | at 1200 |\n|---|---:|---|---:|---:|---:|---:|---:|---:|')
        for r, ser in sorted(items, key=lambda x: (x[0]['variant'], x[0]['D'], x[0]['dh'], x[0]['dp'])):
            if r['variant'].startswith('perturb'):
                continue
            hs = ser['hit_sym']
            pn = ser['pos_n1cm']
            print(f'| {r["variant"]} D={r["D"]} dh={r["dh"]} dp={r["dp"]} | {r["deaths"]} (ref {r["ref_deaths"]}) | '
                  f'{r["survivors_by_team"][0]}/{r["survivors_by_team"][1]} (ref {r["ref_survivors_by_team"][0]}/{r["ref_survivors_by_team"][1]}) | '
                  f'{r["t_half"]} (ref {r["ref_t_half"]}) | {sum(hs[:60]):.0f} | {sum(hs[:300]):.0f} | {pn[59]:.0f} | {pn[299]:.0f} | {pn[-1]:.0f} |')
    bad = [(r['name'], r['not_visible']) for r, _ in rows if r['not_visible']]
    print('\nconfigurations with interactions outside the deciding circle:', bad or 'none')


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
