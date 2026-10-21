"""Dense-battle R7c + R7f: what a per-tick barrier exchange between D deciders costs in real processes.

Question (docs/part3-many-deciders.md, R7c and R7f): R7d showed that D deciders equal one
decider only with a boundary exchange inside every tick, a >= 3 m crowd halo and canonical ordering of the hits on one
target -- but ran the deciders in lock-step inside ONE process, so the barrier cost was not measured. Here every decider is
its own OS process (multiprocessing, spawn), the exchange goes through shared memory mailboxes, the barrier is a spin
barrier on shared-memory counters, and the per-tick clock is split into compute / exchange / barrier wait / sequencer.

The simulation is r7d_replay.py's, imported unchanged (World, crowd, sweep, the recorded trace, the single-decider
reference and diff_hits): same disc at 1 person/m2, 30 Hz, same bands (circle_S: cuts on y at equal start population),
crowd = 4 Jacobi iterations of 0.7 m discs, sector hit test, 120 damage, HP 4000, overkill drop. Only the exchange is new.

Tick of decider d (two barriers; three/four with --seq central):
  1  input, swing intents from the trace, MOVE of every body it holds the state of
  2  PUBLISH 1  moved positions of its bodies that lie within the halo (default 3 m) of a neighbour's band (id, x, y), and the
     full state (position, swing start, yaw, hp, dedupe keys) of bodies that crossed into the neighbour's band last tick
  ---- barrier 1 ----
  3  READ 1, CROWD on owned + halo copies (the r7d rule), PUBLISH 2  snapshots of the active attackers that stand within
     5.5 m of a neighbour's band (forward-to-owner, R7b at delay 0)
  ---- barrier 2 ----
  4  READ 2, sweep: every attacker (own + snapshots) against the bodies this decider owns; per-pair dedupe; the hits on one
     target are ordered by (swing tick, attacker id) and the first ceil(HP/120) count (the SEQUENCER step, timed on its own);
     deaths; bodies that ended the tick in another band are handed over in the next PUBLISH 1.
  --seq central: after the sweep every decider sends all hits on targets that took a cross hit to decider 0, which ranks
     them (the central sequencer), barrier 3 / barrier 4 around it.

Ownership: a body is owned by the band it stands in at the start of the tick (r7d). The decider that owned it last tick
still holds its state at the start of this tick, so it does the (state-only) move, then hands it over in PUBLISH 1.

Measured per tick and decider (ms): compute, exchange (pack + unpack of mailboxes), barrier wait, sequencer; bytes written
(payload only). Equivalence is checked every tick against the single-decider reference of r7d (positions and HP of every body bit
for bit, every applied hit, death ticks); the check is NOT inside the clocks.

    python r7c_barrier.py --n 2000 --deciders 1,2 --ticks 1200
    python r7c_barrier.py --empty            # latency of exchange + barrier with an empty simulation
    python r7c_barrier.py --summarize
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import tempfile
import time
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
R7D = OUT.parent / 'r7d'
sys.path.insert(0, str(R7D))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np                      # noqa: E402
import r7d_replay as b                  # noqa: E402

RAW = OUT / 'raw'
CACHE = Path(tempfile.gettempdir()) / 'r7c_cache'
pc = time.perf_counter_ns

H1 = np.dtype([('id', '<i4'), ('x', '<f8'), ('y', '<f8')])                                            # 20 B halo row
HAND = np.dtype([('id', '<i4'), ('x', '<f8'), ('y', '<f8'), ('act', '<i4'), ('yaw', 'u1'), ('hp', '<i4')])   # 29 B
SNAP = np.dtype([('id', '<i4'), ('x', '<f8'), ('y', '<f8'), ('yaw', 'u1'), ('act', '<i4')])           # 25 B attacker snapshot
CAND = np.dtype([('a', '<i4'), ('tg', '<i4'), ('sw', '<i4'), ('hp', '<i4')])                          # 16 B hit candidate
PDK = np.dtype('<i8')                                                                                 # 8 B dedupe key
HM_ATK = b.SWING_REACH + 3.0           # attacker snapshots go to a neighbour whose band is closer than this (reach + drift)
BUCKETS = ['in', 'pack1', 'w1', 'unpack1', 'crowd', 'pack2', 'w2', 'unpack2', 'sweep', 'seq', 'seqc', 'w3', 'w4', 'apply']
BI = {k: i for i, k in enumerate(BUCKETS)}
NB = len(BUCKETS)
# which buckets form the four reported classes
CLASS = dict(compute=['in', 'crowd', 'sweep', 'apply'], exchange=['pack1', 'unpack1', 'pack2', 'unpack2'],
             barrier=['w1', 'w2', 'w3', 'w4'], sequencer=['seq', 'seqc'])
CAPS = dict(hand=8192, pd=1 << 16, cand=1 << 16)


# ----------------------------------------------------------------------------- shared memory layout
def layout(N, D):
    """byte offsets: 4 barrier counters, then for every ordered pair (src, dst) one mailbox per phase"""
    cap_h = cap_s = N
    off = {}
    pos = 0
    off['flags'] = pos
    pos += 4 * D * 64
    box = 64 + cap_h * H1.itemsize + CAPS['hand'] * HAND.itemsize + CAPS['pd'] * 8
    box = (box + 63) // 64 * 64
    box2 = 64 + cap_s * SNAP.itemsize
    box2 = (box2 + 63) // 64 * 64
    box3 = 64 + CAPS['cand'] * CAND.itemsize + CAPS['cand']       # candidates to the sequencer + accept bytes back
    box3 = (box3 + 63) // 64 * 64
    off['p1'], off['p2'], off['p3'] = pos, pos + D * D * box, pos + D * D * (box + box2)
    off['box'], off['box2'], off['box3'] = box, box2, box3
    off['size'] = off['p3'] + D * D * box3
    off['cap_h'], off['cap_s'] = cap_h, cap_s
    return off


class Mail:
    """numpy views of the mailboxes of one (src, dst) pair"""

    def __init__(self, buf, off, D, s, d):
        i = s * D + d
        o1 = off['p1'] + i * off['box']
        self.hdr1 = np.ndarray((4,), '<i8', buf, o1)
        p = o1 + 64
        self.h1 = np.ndarray((off['cap_h'],), H1, buf, p)
        p += off['cap_h'] * H1.itemsize
        self.hand = np.ndarray((CAPS['hand'],), HAND, buf, p)
        p += CAPS['hand'] * HAND.itemsize
        self.pd = np.ndarray((CAPS['pd'],), PDK, buf, p)
        o2 = off['p2'] + i * off['box2']
        self.hdr2 = np.ndarray((4,), '<i8', buf, o2)
        self.snap = np.ndarray((off['cap_s'],), SNAP, buf, o2 + 64)
        o3 = off['p3'] + i * off['box3']
        self.hdr3 = np.ndarray((4,), '<i8', buf, o3)
        self.cand = np.ndarray((CAPS['cand'],), CAND, buf, o3 + 64)
        self.acc = np.ndarray((CAPS['cand'],), 'u1', buf, o3 + 64 + CAPS['cand'] * CAND.itemsize)


class Spin:
    """sense-free spin barrier: every participant owns one 64-byte slot of a shared counter array"""

    def __init__(self, buf, base, D, me, peers=None):
        self.f = buf[base:base + D * 64].cast('q')
        self.me = me
        self.others = [i * 8 for i in (peers if peers is not None else range(D)) if i != me]

    def wait(self, k):
        f = self.f
        f[self.me * 8] = k
        n = 0
        for i in self.others:
            while f[i] < k:
                n += 1
                if n > 4000:
                    time.sleep(0)
                    n = 0


def rank_ok(a_, tg_, hp_):
    """canonical order of the hits on one target: (swing tick, attacker id); the first ceil(hp/DMG) count. Aligned with the input."""
    n = len(tg_)
    ok = np.zeros(n, bool)
    if n == 0:
        return ok
    o = np.lexsort((tg_, a_))                      # swing tick is equal for all rows of one tick
    s2 = o[np.argsort(tg_[o], kind='stable')]
    tgs = tg_[s2]
    first = np.r_[True, tgs[1:] != tgs[:-1]]
    ix = np.arange(n)
    rank = ix - np.maximum.accumulate(np.where(first, ix, 0))
    need = (np.maximum(hp_[s2], 0) + b.DMG - 1) // b.DMG
    ok[s2] = rank < need
    return ok


# ----------------------------------------------------------------------------- one decider process
def decider(me, a, shm_name, conn, start_bar):
    from multiprocessing import shared_memory
    N, T, D = a['N'], a['T'], a['D']
    vm, hm, seqmode, verify = a['halo'], HM_ATK, a['seq'], a['verify']
    cache = Path(a['cache'])
    key = b.case_key(N, 'front', T, b.HP0)
    world = b.World(N, 'front')
    team, R = world.team, world.R
    tr = np.load(cache / (key + '.trace.npz'))
    mx, my, yqa, sw = tr['mx'], tr['my'], tr['yq'], tr['sw']
    swoff = np.searchsorted(sw[:, 0], np.arange(0, T + 3))
    ref = b.load_ref(cache / (key + '_L0'))
    shm = shared_memory.SharedMemory(name=shm_name)
    off = layout(N, D)
    buf = shm.buf
    others = [e for e in range(D) if e != me]
    if D > 1:
        cuts, ylo, yhi, cen, crad = world.bands(D)
        # neighbours: deciders whose band is closer than the farthest reach of an exchange (halo or attacker snapshot)
        nbrs = [e for e in others if max(0.0, ylo[e] - yhi[me], ylo[me] - yhi[e]) < max(vm, hm)]
    else:
        nbrs = []
    peers = nbrs if a['barrier'] == 'neighbor' else None
    bars = [Spin(buf, off['flags'] + i * D * 64, D, me, peers if i < 2 else None) for i in range(4)]
    mail = {(s, d): Mail(buf, off, D, s, d) for s in range(D) for d in range(D) if s != d}
    px, py = world.px0.copy(), world.py0.copy()
    hp = np.full(N, b.HP0, np.int64)
    act = np.full(N, -1000, np.int64)
    yawi = np.zeros(N, np.uint8)
    pd = np.empty(0, np.int64)
    H = np.nonzero((np.searchsorted(cuts, py) if D > 1 else np.zeros(N, np.int64)) == me)[0].astype(np.int64)
    ser = np.zeros((T + 1, NB), np.float32)         # ms per bucket
    tot = np.zeros(T + 1, np.float32)               # whole tick, ms (verify excluded)
    cnt = np.zeros((T + 1, 10), np.int64)           # held, owned, halo_in, hand_in, snap_in, hits, cross_hits, msgs_out, bytes_out, pairs
    ver = dict(pos_bad=0, hp_bad=0, pos_max=0.0, checked=0, first_bad=-1)
    hits_acc, deaths = [], []
    DMG, FR, WU, AC = b.DMG, b.FRAMES, b.WINDUP, b.ACTIVE
    start_bar.wait()
    t_wall0 = time.perf_counter()
    for t in range(1, T + 1):
        T0 = pc()
        tk = T0
        sec = np.zeros(NB)
        c = cnt[t]
        c[0] = len(H)

        def lap(name):
            nonlocal tk
            n = pc()
            sec[BI[name]] += (n - tk) / 1e6
            tk = n

        # ---- 1. input, swing intents, move of every held body
        mxq, myq, yq = mx[t][H], my[t][H], yqa[t][H]
        yawi[H] = yq
        desx = mxq.astype(np.float64) / 32767
        desy = myq.astype(np.float64) / 32767
        sa = sw[swoff[t]:swoff[t + 1], 1]
        if sa.size and len(H):
            ps = np.minimum(np.searchsorted(H, sa), len(H) - 1)
            am = sa[H[ps] == sa]
            if am.size:
                can = (t - act[am]) >= FR
                act[am[can]] = t
        spd = np.where(t - act[H] < FR, b.V_SWING, b.V_IDLE)
        mvx = px[H] + desx * spd * b.DT
        mvy = py[H] + desy * spd * b.DT
        if D > 1:
            ob = np.searchsorted(cuts, py[H])           # owner band at the start of the tick
        else:
            ob = np.zeros(len(H), np.int64)
        lap('in')
        if D == 1:
            nx, ny = b.crowd(mvx, mvy, R)
            px[H], py[H] = nx, ny
            O = H
            ex_ids = np.empty(0, np.int64)
            lap('crowd')
            Ox = Oy = None
        else:
            # ---- 2. publish 1
            mine = ob == me
            nbytes = 0
            nmsg = 0
            pdk_tg = (pd // b.KEY_SW) % N if pd.size else pd
            drop_pd = np.zeros(pd.size, bool)
            assert np.isin(ob, [me] + nbrs).all(), 'a body moved past the neighbour band'
            for e in nbrs:
                mb = mail[(me, e)]
                mh = ob == e
                mo = (~mh) & (mvy >= ylo[e] - vm) & (mvy <= yhi[e] + vm)
                n1, n2 = int(mo.sum()), int(mh.sum())
                assert n1 <= off['cap_h'] and n2 <= CAPS['hand'], 'mailbox overflow'
                v = mb.h1[:n1]
                v['id'], v['x'], v['y'] = H[mo], mvx[mo], mvy[mo]
                hh = mb.hand[:n2]
                ids2 = H[mh]
                hh['id'], hh['x'], hh['y'] = ids2, mvx[mh], mvy[mh]
                hh['act'], hh['yaw'], hh['hp'] = act[ids2], yawi[ids2], hp[ids2]
                if n2 and pd.size:
                    mp_ = np.isin(pdk_tg, ids2)
                    n3 = int(mp_.sum())
                    mb.pd[:n3] = pd[mp_]
                    drop_pd |= mp_
                else:
                    n3 = 0
                mb.hdr1[0], mb.hdr1[1], mb.hdr1[2] = n1, n2, n3
                nbytes += n1 * H1.itemsize + n2 * HAND.itemsize + n3 * 8
                nmsg += int(n1 > 0 or n2 > 0 or n3 > 0)
            if drop_pd.any():
                pd = pd[~drop_pd]
            lap('pack1')
            bars[0].wait(t)
            lap('w1')
            # ---- 3. read 1
            own_ids = [H[mine]]
            own_x, own_y = [mvx[mine]], [mvy[mine]]
            mo_me = (~mine) & (mvy >= ylo[me] - vm) & (mvy <= yhi[me] + vm)       # my departed bodies are foreign for me now
            f_ids, f_x, f_y = [H[mo_me]], [mvx[mo_me]], [mvy[mo_me]]
            new_pd = []
            nh = nhd = 0
            for s in nbrs:
                mb = mail[(s, me)]
                n1, n2, n3 = int(mb.hdr1[0]), int(mb.hdr1[1]), int(mb.hdr1[2])
                if n1:
                    v = mb.h1[:n1]
                    f_ids.append(v['id'].astype(np.int64)), f_x.append(v['x'].copy()), f_y.append(v['y'].copy())
                    nh += n1
                if n2:
                    hh = mb.hand[:n2]
                    ids2 = hh['id'].astype(np.int64)
                    own_ids.append(ids2), own_x.append(hh['x'].copy()), own_y.append(hh['y'].copy())
                    act[ids2], yawi[ids2], hp[ids2] = hh['act'], hh['yaw'], hh['hp']
                    nhd += n2
                if n3:
                    new_pd.append(mb.pd[:n3].copy())
            if new_pd:
                pd = np.concatenate([pd] + new_pd)
            oi, ox, oy = np.concatenate(own_ids), np.concatenate(own_x), np.concatenate(own_y)
            fi, fx, fy = np.concatenate(f_ids), np.concatenate(f_x), np.concatenate(f_y)
            vid = np.concatenate([oi, fi])
            order = np.argsort(vid, kind='stable')
            vid = vid[order]
            vx = np.concatenate([ox, fx])[order]
            vy = np.concatenate([oy, fy])[order]
            isown = (order < len(oi))
            c[2], c[3] = nh, nhd
            lap('unpack1')
            # ---- crowd
            nx, ny = b.crowd(vx, vy, R)
            O = vid[isown]
            px[O], py[O] = nx[isown], ny[isown]
            lap('crowd')
        # ---- attackers
        fr = t - act[O]
        atk = (fr >= WU) & (fr < WU + AC)
        Aloc = O[atk]
        if D > 1:
            # ---- publish 2
            ay_ = py[Aloc]
            for e in nbrs:
                mb = mail[(me, e)]
                m = (ay_ >= ylo[e] - hm) & (ay_ <= yhi[e] + hm)
                n1 = int(m.sum())
                assert n1 <= off['cap_s'], 'snapshot overflow'
                ids2 = Aloc[m]
                v = mb.snap[:n1]
                v['id'], v['x'], v['y'], v['yaw'], v['act'] = ids2, px[ids2], py[ids2], yawi[ids2], act[ids2]
                mb.hdr2[0] = n1
                nbytes += n1 * SNAP.itemsize
                nmsg += int(n1 > 0)
            lap('pack2')
            bars[1].wait(t)
            lap('w2')
            s_ids, s_x, s_y, s_yaw, s_act = [Aloc], [px[Aloc]], [py[Aloc]], [yawi[Aloc]], [act[Aloc]]
            ns = 0
            for s in nbrs:
                mb = mail[(s, me)]
                n1 = int(mb.hdr2[0])
                if n1:
                    v = mb.snap[:n1]
                    s_ids.append(v['id'].astype(np.int64)), s_x.append(v['x'].copy()), s_y.append(v['y'].copy())
                    s_yaw.append(v['yaw'].copy()), s_act.append(v['act'].astype(np.int64))
                    ns += n1
            aid, axx, ayy = np.concatenate(s_ids), np.concatenate(s_x), np.concatenate(s_y)
            ayw, aact = np.concatenate(s_yaw), np.concatenate(s_act)
            aloc = np.concatenate([np.zeros(len(Aloc), bool), np.ones(ns, bool)])       # True = snapshot (cross)
            c[4] = ns
            lap('unpack2')
        else:
            aid, axx, ayy, ayw, aact = Aloc, px[Aloc], py[Aloc], yawi[Aloc], act[Aloc]
            aloc = np.zeros(len(Aloc), bool)
        # ---- sweep: every attacker against the bodies this decider owns
        afc, afs = b.yaw_vec(ayw)
        r, tg = b.sweep(team, axx, ayy, afc, afs, team[aid], O, px[O], py[O])
        if pd.size:
            pd = pd[(pd % b.KEY_SW) >= t - 3 * FR]
        aa, swv, cr = aid[r], aact[r], aloc[r]
        kk = (aa * N + tg) * b.KEY_SW + swv
        keep = ~np.isin(kk, pd)
        aa, tg, swv, cr, kk = aa[keep], tg[keep], swv[keep], cr[keep], kk[keep]
        if kk.size:
            pd = np.concatenate([pd, kk])
        lap('sweep')
        # ---- sequencing: the hits on one target, in (swing tick, attacker id) order, first ceil(HP/120) count
        ok = np.zeros(len(tg), bool)
        if seqmode == 'central' and D > 1:
            ct = np.unique(tg[cr]) if cr.any() else np.empty(0, np.int64)
            inrows = np.isin(tg, ct) if ct.size else np.zeros(len(tg), bool)
            rows = np.nonzero(inrows)[0]
            nr = len(rows)
            assert nr <= CAPS['cand']
            if me != 0:
                mb0 = mail[(me, 0)]
                cv = mb0.cand[:nr]
                cv['a'], cv['tg'], cv['sw'], cv['hp'] = aa[rows], tg[rows], swv[rows], hp[tg[rows]]
                mb0.hdr3[0] = nr
                nbytes += nr * CAND.itemsize
                nmsg += int(nr > 0)
            lap('pack2')
            bars[2].wait(t)
            lap('w3')
            acc_me = None
            if me == 0:
                parts = [(aa[rows].astype(np.int64), tg[rows], hp[tg[rows]])]
                for s in others:
                    mbs = mail[(s, 0)]
                    n1 = int(mbs.hdr3[0])
                    cv = mbs.cand[:n1]
                    parts.append((cv['a'].astype(np.int64), cv['tg'].astype(np.int64), cv['hp'].astype(np.int64)))
                lap('unpack2')
                ca, ctg, chp = (np.concatenate([p[i] for p in parts]) for i in range(3))
                accepted = rank_ok(ca, ctg, chp)
                p0 = 0
                for si, s in enumerate([0] + others):
                    n1 = len(parts[si][0])
                    seg = accepted[p0:p0 + n1]
                    p0 += n1
                    if s == 0:
                        acc_me = seg
                    else:
                        mail[(0, s)].acc[:n1] = seg.astype(np.uint8)
                nbytes += len(ca)
                lap('seqc')
            bars[3].wait(t)
            lap('w4')
            if me != 0:
                acc_me = mail[(0, me)].acc[:nr].astype(bool)
                lap('unpack2')
            ok[rows] = acc_me
            loc = ~inrows
            ok[loc] = rank_ok(aa[loc], tg[loc], hp[tg[loc]])
        else:
            ok = rank_ok(aa, tg, hp[tg])
        lap('seq')
        sel = np.nonzero(ok)[0]
        if len(sel):
            u, k = np.unique(tg[sel], return_counts=True)
            hp[u] -= DMG * k
            hits_acc.append(np.column_stack([np.full(len(sel), t), np.full(len(sel), t), aa[sel], tg[sel], swv[sel], cr[sel]]).astype(np.int64))
            c[5], c[6] = len(sel), int(cr[sel].sum())
        dead = O[hp[O] <= 0]
        if dead.size:
            deaths.append(np.column_stack([dead, np.full(len(dead), t)]))
        lap('apply')
        H_next = O[hp[O] > 0]
        c[1] = len(O)
        c[7], c[8] = nmsg if D > 1 else 0, nbytes if D > 1 else 0
        ser[t] = sec
        tot[t] = (pc() - T0) / 1e6
        # ---- equivalence check against the single-decider reference (outside the clocks)
        if verify:
            rp = ref['pos'][t]
            ex_, ey_ = px[O] - rp[O, 0], py[O] - rp[O, 1]
            pbad = int(((ex_ != 0) | (ey_ != 0)).sum())
            hbad = int((np.maximum(hp[O], 0) != np.asarray(ref['hp'][t])[O].astype(np.int64)).sum())
            ver['pos_bad'] += pbad
            ver['hp_bad'] += hbad
            ver['checked'] += len(O)
            if len(O):
                ver['pos_max'] = max(ver['pos_max'], float(np.hypot(ex_, ey_).max()))
            if (pbad or hbad) and ver['first_bad'] < 0:
                ver['first_bad'] = t
        H = H_next
    wall = time.perf_counter() - t_wall0
    hits = np.concatenate(hits_acc) if hits_acc else np.empty((0, 6), np.int64)
    dth = np.concatenate(deaths) if deaths else np.empty((0, 2), np.int64)
    conn.send(dict(me=me, ser=ser, tot=tot, cnt=cnt, ver=ver, hits=hits, deaths=dth, wall=wall, final_hp=hp, N=N))
    conn.close()
    try:
        del bars, mail
        shm.close()
    except BufferError:
        pass


# ----------------------------------------------------------------------------- orchestration
def harness_running():
    try:
        out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq Harness*'], capture_output=True, text=True, timeout=20).stdout
        return any(l.strip().lower().startswith('harness') for l in out.splitlines())
    except Exception:
        return False


def sys_times():
    """(idle, kernel, user) 100 ns ticks of the whole machine (kernel includes idle)"""
    import ctypes
    from ctypes import wintypes
    i, k, u = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
    f = lambda x: (x.dwHighDateTime << 32) | x.dwLowDateTime
    return f(i), f(k), f(u)


def busy_pct(a, b_):
    di, dk, du = b_[0] - a[0], b_[1] - a[1], b_[2] - a[2]
    tot = dk + du
    return round(100.0 * (tot - di) / tot, 1) if tot else 0.0


def pct(v, q):
    v = np.asarray(v, float)
    return round(float(np.percentile(v, q)), 4) if len(v) else 0.0


def run_config(N, D, T, halo=3.0, seq='merge', verify=True, tag='', force=False, barrier='global'):
    name = f'n{N}_D{D}_t{T}_h{halo:g}_{seq}' + ('' if verify else '_nov') + ('' if barrier == 'global' else '_nb') + (f'_{tag}' if tag else '')
    path = RAW / (name + '.json')
    if path.exists() and not force:
        return json.loads(path.read_text())['summary']
    hr = harness_running()
    if hr and D > 2:
        print(f'[skip] {name}: Harness is running, D={D} > 2 workers not allowed', flush=True)
        return dict(name=name, skipped='Harness running')
    from multiprocessing import shared_memory
    ctx = mp.get_context('spawn')
    off = layout(N, D)
    shm = shared_memory.SharedMemory(create=True, size=off['size'])
    shm.buf[:off['flags'] + 4 * D * 64] = bytes(4 * D * 64 + off['flags'])
    start_bar = ctx.Barrier(D)
    st0 = sys_times()
    a = dict(N=N, T=T, D=D, halo=halo, seq=seq, verify=verify, cache=str(CACHE), barrier=barrier)
    procs, conns = [], []
    for me in range(D):
        pr, ch = ctx.Pipe(duplex=False)
        p = ctx.Process(target=decider, args=(me, a, shm.name, ch, start_bar), daemon=True)
        p.start()
        procs.append(p)
        conns.append(pr)
    res = [None] * D
    t0 = time.time()
    while any(r is None for r in res):
        for i, cn in enumerate(conns):
            if res[i] is None and cn.poll(0.2):
                res[i] = cn.recv()
        if any(r is None for r in res) and any((not p.is_alive()) and p.exitcode not in (0, None) for p in procs):
            for p in procs:
                p.terminate()
            shm.close()
            shm.unlink()
            raise RuntimeError(f'{name}: a decider process died: ' + str([p.exitcode for p in procs]))
    for p in procs:
        p.join(timeout=60)
    shm.close()
    shm.unlink()
    wall = time.time() - t0
    st1 = sys_times()
    hr2 = harness_running()
    # --- equivalence
    key = b.case_key(N, 'front', T, b.HP0)
    ref = b.load_ref(CACHE / (key + '_L0'))
    hits = np.concatenate([r['hits'] for r in res])
    mh = dict(T=hits[:, 0], tau=hits[:, 1], a=hits[:, 2], tgt=hits[:, 3], sw=hits[:, 4], cross=hits[:, 5].astype(bool))
    hd, _, _ = b.diff_hits(ref['hits'], mh, N, T)
    dth = np.concatenate([r['deaths'] for r in res])
    death = np.full(N, -1, np.int64)
    death[dth[:, 0]] = dth[:, 1]
    rd = ref['death']
    ref_death = np.where((rd >= 0) & (rd <= T), rd, -1)
    equiv = dict(hits_missing=hd['missing'], hits_extra=hd['extra'], hits_late=hd['late'], ref_hits=hd['ref_hits'],
                 run_hits=hd['run_hits'], pos_bodies_bad=sum(r['ver']['pos_bad'] for r in res),
                 hp_bodies_bad=sum(r['ver']['hp_bad'] for r in res), pos_max_m=max(r['ver']['pos_max'] for r in res),
                 body_ticks_checked=sum(r['ver']['checked'] for r in res), death_tick_differs=int((death != ref_death).sum()),
                 deaths=int((death >= 0).sum()), ref_deaths=int((ref_death >= 0).sum()), verified=verify,
                 first_bad_tick=[r['ver']['first_bad'] for r in res])
    equiv['zero'] = bool(verify and equiv['hits_missing'] == 0 and equiv['hits_extra'] == 0 and equiv['hits_late'] == 0 and
                         equiv['pos_bodies_bad'] == 0 and equiv['hp_bodies_bad'] == 0 and equiv['death_tick_differs'] == 0)
    # --- timing
    W = 5                                              # warm-up ticks dropped from the statistics
    ser = np.stack([r['ser'][W + 1:] for r in res])    # (D, ticks, NB)
    tot = np.stack([r['tot'][W + 1:] for r in res])    # (D, ticks)
    cnt = np.stack([r['cnt'][W + 1:] for r in res])    # (D, ticks, 10)
    cls = {k: ser[:, :, [BI[x] for x in v]].sum(axis=2) for k, v in CLASS.items()}      # (D, ticks)
    out = dict(name=name, N=N, D=D, ticks=T, halo_m=halo, seq=seq, barrier=barrier,
               throughput_tick_ms=round(max(r['wall'] for r in res) / T * 1000, 4), harness_before=hr, harness_after=hr2, wall_s=round(wall, 1),
               machine_cpu_busy_pct_during_run=busy_pct(st0, st1), own_workers=D)
    for k, v in cls.items():
        mean_d = v.mean(axis=0)
        out[k + '_ms'] = dict(p50=pct(mean_d, 50), p99=pct(mean_d, 99), max=round(float(mean_d.max()), 4),
                              worst_decider_p50=pct(v.max(axis=0), 50), worst_decider_p99=pct(v.max(axis=0), 99))
    tick_sys = tot.max(axis=0)
    out['tick_ms'] = dict(p50=pct(tick_sys, 50), p99=pct(tick_sys, 99), max=round(float(tick_sys.max()), 4), mean=round(float(tick_sys.mean()), 4))
    out['tick_mean_decider_ms'] = dict(p50=pct(tot.mean(axis=0), 50), p99=pct(tot.mean(axis=0), 99))
    span = (cls['compute'] + cls['exchange'] + cls['sequencer']).max(axis=0)       # slowest decider's own work, no waiting
    out['span_own_work_ms'] = dict(p50=pct(span, 50), p99=pct(span, 99))
    out['buckets_p50_ms'] = {k: pct(ser[:, :, i].mean(axis=0), 50) for k, i in BI.items()}
    out['buckets_p99_ms'] = {k: pct(ser[:, :, i].mean(axis=0), 99) for k, i in BI.items()}
    by = cnt[:, :, 8].sum(axis=0)                    # bytes written per tick by all deciders
    out['bytes_per_tick'] = dict(p50=pct(by, 50), p99=pct(by, 99), max=int(by.max()))
    out['bytes_per_decider_tick'] = dict(p50=pct(cnt[:, :, 8].max(axis=0), 50), p99=pct(cnt[:, :, 8].max(axis=0), 99))
    out['messages_per_tick_p50'] = pct(cnt[:, :, 7].sum(axis=0), 50)
    out['rows_p50'] = dict(held_total=pct(cnt[:, :, 0].sum(axis=0), 50), halo_in_per_decider=pct(cnt[:, :, 2].mean(axis=0), 50),
                           handover_in_total=pct(cnt[:, :, 3].sum(axis=0), 50), snapshots_in_per_decider=pct(cnt[:, :, 4].mean(axis=0), 50),
                           hits_per_tick=pct(cnt[:, :, 5].sum(axis=0), 50), cross_hits_per_tick=pct(cnt[:, :, 6].sum(axis=0), 50))
    imb = cnt[:, :, 1].max(axis=0) / np.maximum(cnt[:, :, 1].mean(axis=0), 1)
    out['owned_imbalance_max_over_mean_p50'] = pct(imb, 50)
    out['per_decider'] = [dict(d=i, tick_p50=pct(tot[i], 50), tick_p99=pct(tot[i], 99),
                               **{k + '_p50': pct(cls[k][i], 50) for k in cls}, **{k + '_p99': pct(cls[k][i], 99) for k in cls},
                               owned_p50=pct(cnt[i, :, 1], 50)) for i in range(D)]
    out['alive_start_p50_total'] = pct(cnt[:, :, 0].sum(axis=0), 50)
    out['equivalence'] = equiv
    out['host'] = dict(python=platform.python_version(), numpy=np.__version__, machine=platform.platform(), cpu_count=os.cpu_count(),
                       date=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    series = dict(tick_sys_ms=np.round(tot.max(axis=0), 3).tolist(), bytes=by.tolist(),
                  compute_mean_ms=np.round(cls['compute'].mean(axis=0), 3).tolist(), barrier_mean_ms=np.round(cls['barrier'].mean(axis=0), 3).tolist(),
                  owned_total=cnt[:, :, 0].sum(axis=0).tolist())
    path.write_text('{\n"summary": ' + json.dumps(out, indent=1) + ',\n"series": {\n' +
                    ',\n'.join(f'"{k}": ' + json.dumps(v, separators=(',', ':')) for k, v in series.items()) + '\n}\n}\n')
    tag_ok = 'EQUIV 0' if equiv['zero'] else ('equiv NOT 0' if verify else 'unverified')
    print(f'[{name}] tick p50 {out["tick_ms"]["p50"]} p99 {out["tick_ms"]["p99"]} ms | compute {out["compute_ms"]["p50"]} '
          f'exch {out["exchange_ms"]["p50"]} barrier {out["barrier_ms"]["p50"]} seq {out["sequencer_ms"]["p50"]} | '
          f'{out["bytes_per_tick"]["p50"]:.0f} B/tick | {tag_ok} | wall {out["wall_s"]} s', flush=True)
    return out


# ----------------------------------------------------------------------------- empty-simulation latency
def empty_worker(me, D, kind, rows_h, rows_s, ticks, shm_name, off, bar_mp, pipes, conn):
    from multiprocessing import shared_memory
    if kind == 'pipe':
        send = {e: pipes[(me, e)][1] for e in range(D) if e != me}
        recv = {s: pipes[(s, me)][0] for s in range(D) if s != me}
        payload = bytes(rows_h * H1.itemsize + rows_s * SNAP.itemsize)
        bar_mp.wait()
        t_ = np.zeros(ticks)
        for t in range(ticks):
            t0 = pc()
            for _rep in range(2):          # two exchange points per tick (moved positions, then attacker snapshots)
                for e in sorted(send):     # ordered pairwise exchange: no deadlock when the payload exceeds the pipe buffer
                    if me < e:
                        send[e].send_bytes(payload)
                        recv[e].recv_bytes()
                    else:
                        recv[e].recv_bytes()
                        send[e].send_bytes(payload)
            t_[t] = (pc() - t0) / 1e6
        conn.send(t_)
        return
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = shm.buf
    bars = [Spin(buf, off['flags'] + i * D * 64, D, me) for i in range(2)]
    mail = {(s, d): Mail(buf, off, D, s, d) for s in range(D) for d in range(D) if s != d}
    others = [e for e in range(D) if e != me]
    bar_mp.wait()
    t_ = np.zeros(ticks)
    w_ = np.zeros(ticks)
    for t in range(1, ticks + 1):
        t0 = pc()
        for e in others:
            mb = mail[(me, e)]
            v = mb.h1[:rows_h]
            v['id'] = 1
            v['x'] = 2.0
            v['y'] = 3.0
            mb.hdr1[0] = rows_h
        wt = pc()
        if kind == 'spin':
            bars[0].wait(t)
        else:
            bar_mp.wait()
        w_[t - 1] += (pc() - wt) / 1e6
        for s in others:
            mb = mail[(s, me)]
            n = int(mb.hdr1[0])
            _ = mb.h1[:n]['id'].astype(np.int64), mb.h1[:n]['x'].copy(), mb.h1[:n]['y'].copy()
        for e in others:
            mb = mail[(me, e)]
            v = mb.snap[:rows_s]
            v['id'] = 1
            v['x'] = 2.0
            mb.hdr2[0] = rows_s
        wt = pc()
        if kind == 'spin':
            bars[1].wait(t)
        else:
            bar_mp.wait()
        w_[t - 1] += (pc() - wt) / 1e6
        for s in others:
            mb = mail[(s, me)]
            n = int(mb.hdr2[0])
            _ = mb.snap[:n]['id'].astype(np.int64), mb.snap[:n]['x'].copy()
        t_[t - 1] = (pc() - t0) / 1e6
    conn.send(np.stack([t_, w_]))
    shm.close()


def empty_bench(D, kind, rows_h, rows_s, ticks=3000):
    from multiprocessing import shared_memory
    ctx = mp.get_context('spawn')
    N = max(rows_h, rows_s, 64)
    off = layout(N, D)
    shm = shared_memory.SharedMemory(create=True, size=off['size'])
    shm.buf[:off['flags'] + 4 * D * 64] = bytes(4 * D * 64 + off['flags'])
    bar = ctx.Barrier(D)
    pipes = {}
    if kind == 'pipe':
        for s in range(D):
            for e in range(D):
                if s != e:
                    pipes[(s, e)] = ctx.Pipe(duplex=False)
    procs, conns = [], []
    for me in range(D):
        pr, ch = ctx.Pipe(duplex=False)
        p = ctx.Process(target=empty_worker, args=(me, D, kind, rows_h, rows_s, ticks, shm.name, off, bar, pipes, ch), daemon=True)
        p.start()
        procs.append(p)
        conns.append(pr)
    res = [c.recv() for c in conns]
    for p in procs:
        p.join(timeout=30)
    shm.close()
    shm.unlink()
    if kind == 'pipe':
        tt = np.stack(res)[:, 20:]
        wt = None
    else:
        tt = np.stack([r[0] for r in res])[:, 20:]
        wt = np.stack([r[1] for r in res])[:, 20:]
    tick = tt.max(axis=0)
    o = dict(D=D, kind=kind, rows_halo=rows_h, rows_snap=rows_s, ticks=ticks, exchange_points=2,
             payload_bytes_per_pair_per_tick=rows_h * H1.itemsize + rows_s * SNAP.itemsize,
             tick_ms=dict(p50=pct(tick, 50), p99=pct(tick, 99), max=round(float(tick.max()), 4)),
             per_exchange_point_ms_p50=round(pct(tick, 50) / 2, 4))
    if wt is not None:
        o['wait_ms_both_barriers'] = dict(p50=pct(wt.mean(axis=0), 50), p99=pct(wt.mean(axis=0), 99))
    return o


def run_empty(force=False):
    """latency of exchange + barrier with an empty simulation; resumable (rows already measured are kept)"""
    path = RAW / 'empty_latency.json'
    old = json.loads(path.read_text()) if (path.exists() and not force) else dict(rows=[])
    done = {(x['D'], x['kind'], x['rows_halo'], x['rows_snap']) for x in old['rows'] if not x.get('skipped')}
    hr = harness_running()
    out = dict(harness_running=hr, date=datetime.now(timezone.utc).isoformat(timespec='seconds'),
               rows=[x for x in old['rows'] if not x.get('skipped')])
    # payload per pair per tick: zero, and rows typical for the real runs (halo ~ 2R x 3 m per side, snapshots ~ 1/4 of them)
    cases = [(0, 0), (150, 40), (500, 130), (1000, 260)]
    for D in (2, 4, 8):
        if hr and D > 2:
            out['rows'].append(dict(D=D, skipped='Harness running'))
            continue
        for kind in ('spin', 'mpbarrier', 'pipe'):
            for rh, rs in cases:
                if kind == 'pipe' and rh == 0:
                    rh, rs = 1, 1
                if (D, kind, rh, rs) in done:
                    continue
                for attempt in range(6):
                    wait_clear()
                    st0 = sys_times()
                    o = empty_bench(D, kind, rh, rs)
                    o['machine_cpu_busy_pct'] = busy_pct(st0, sys_times())
                    o['harness_after'] = harness_running()
                    if not o['harness_after'] and o['machine_cpu_busy_pct'] <= D * 100.0 / (os.cpu_count() or 16) + 30.0:
                        break
                out['rows'].append(o)
                print(f'[empty] D={D} {kind} halo={rh} snap={rs}: tick p50 {o["tick_ms"]["p50"]} p99 {o["tick_ms"]["p99"]} ms', flush=True)
                path.write_text(json.dumps(out, indent=1))
    path.write_text(json.dumps(out, indent=1))
    return out


def wait_clear(maxwait=1800, busy_thr=25.0):
    """block until no Harness process runs and the machine is quiet (below busy_thr % for a 2 s sample)"""
    t0 = time.time()
    while time.time() - t0 < maxwait:
        if not harness_running():
            a = sys_times()
            time.sleep(2)
            if busy_pct(a, sys_times()) < busy_thr:
                return True
        time.sleep(3)
    return False


def clean(s, D):
    """a gated run is valid when Harness did not run before or after it and the machine was not busy beyond the workers themselves"""
    return (not s.get('skipped')) and not s['harness_before'] and not s['harness_after'] and         s.get('machine_cpu_busy_pct_during_run', 100) <= D * 100.0 / (os.cpu_count() or 16) + 30.0


def run_suite(Ns, Ds, specs, reps, T=1200, prefix='g'):
    """every configuration in a clear window (no Harness, machine quiet); an invalid run is deleted and repeated.
    specs: (seq, halo, verify, barrier) tuples"""
    for rep in range(1, reps + 1):
        for N in Ns:
            for spec in specs:
                seq, halo, verify, barrier = spec
                for D in Ds:
                    if (barrier == 'neighbor' and D < 4) or (D == 1 and (seq != 'merge' or halo != 3.0 or barrier != 'global')):
                        continue
                    tag = f'{prefix}{rep}'
                    for attempt in range(8):
                        if not wait_clear():
                            print(f'[suite] no clear window for N={N} D={D}', flush=True)
                            break
                        s_ = run_config(N, D, T, halo, seq, verify, tag, True, barrier)
                        if clean(s_, D):
                            break
                        nm = s_.get('name')
                        if nm and (RAW / (nm + '.json')).exists():
                            (RAW / (nm + '.json')).unlink()
                        print(f'[suite] run not clean (attempt {attempt + 1}), repeating', flush=True)


MAIN_SPECS = [('merge', 3.0, True, 'global'), ('merge', 3.0, True, 'neighbor')]
VARIANT_SPECS = [('merge', 3.0, True, 'global'), ('central', 3.0, True, 'global'), ('merge', 5.0, True, 'global'), ('merge', 3.0, False, 'global')]


def feed_bench(N, D, iters=3000):
    """R7c second half: merging the D owned arrays into one state feed (24 B per body) is a scatter; timed in one process"""
    rec = np.dtype([('id', '<i4'), ('x', '<f8'), ('y', '<f8'), ('hp', '<i4')])
    rng = np.random.default_rng(7)
    world = b.World(N, 'front')
    cuts = np.quantile(world.py0, np.arange(1, D) / D) if D > 1 else np.empty(0)
    own = np.searchsorted(cuts, world.py0)
    allrec = np.zeros(N, rec)
    allrec['id'], allrec['x'], allrec['y'], allrec['hp'] = np.arange(N), world.px0, world.py0, rng.integers(0, 4000, N)
    parts = [allrec[own == d] for d in range(D)]
    feed = np.zeros(N, rec)
    t_ = np.zeros(iters)
    for i in range(iters):
        t0 = pc()
        for pt in parts:
            feed[pt['id']] = pt
        t_[i] = (pc() - t0) / 1e6
    return dict(N=N, D=D, bytes=N * rec.itemsize, merge_ms=dict(p50=pct(t_, 50), p99=pct(t_, 99), max=round(float(t_.max()), 4)),
                identical_to_single_array=bool(np.array_equal(feed, allrec)))


# ----------------------------------------------------------------------------- summary
def is_gated(name):
    tag = name.rsplit('_', 1)[-1]
    return len(tag) >= 2 and tag[0] in 'gv' and tag[1:].isdigit()


def load_runs(gated=True):
    """group the raw runs by configuration (N, D, seq, halo, verify, barrier); the rep tag is not part of the key"""
    g = {}
    for p in sorted(RAW.glob('n*.json')):
        s = json.loads(p.read_text())['summary']
        if s.get('skipped') or s['name'].endswith('_neg'):
            continue
        if gated and not is_gated(s['name']):
            continue
        k = (s['N'], s['D'], s['seq'], s['halo_m'], bool(s['equivalence']['verified']), s.get('barrier', 'global'))
        g.setdefault(k, []).append(s)
    return g


def med(v):
    return float(np.median(v))


def own_work(s):
    return [d['compute_p50'] + d['exchange_p50'] + d['sequencer_p50'] for d in s['per_decider']]


def imbalance_est(s):
    w = own_work(s)
    return max(w) - float(np.mean(w))


MAIN = lambda k: k[2] == 'merge' and k[3] == 3.0 and k[4]


def summarize():
    g = load_runs(True)
    allg = load_runs(False)
    keys = sorted(g)
    for bar in ('global', 'neighbor'):
        print(f'## Table 1{"a" if bar == "global" else "b"} - cost per tick (ms), barrier = {bar}; median over clean runs of the per-run p50 / p99; mean over deciders; seq merge, halo 3 m, equivalence checked')
        print('| N | D | clean runs | compute p50/p99 | exchange p50/p99 | barrier wait p50/p99 | sequencer p50/p99 | tick p50 [min-max] | tick p99 | throughput ms/tick | own work of slowest p50 | imbalance est. | bytes/tick (all) | bytes per decider | msgs/tick | equivalence 0 | machine CPU % |')
        print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for k in keys:
            if not MAIN(k) or k[5] != bar:
                continue
            r = g[k]
            f = lambda key, q: med([x[key][q] for x in r])
            t50 = [x['tick_ms']['p50'] for x in r]
            cpu = [x.get('machine_cpu_busy_pct_during_run') for x in r if x.get('machine_cpu_busy_pct_during_run') is not None]
            print(f'| {k[0]} | {k[1]} | {len(r)} | {f("compute_ms","p50"):.2f}/{f("compute_ms","p99"):.2f} | {f("exchange_ms","p50"):.2f}/{f("exchange_ms","p99"):.2f} | '
                  f'{f("barrier_ms","p50"):.2f}/{f("barrier_ms","p99"):.2f} | {f("sequencer_ms","p50"):.3f}/{f("sequencer_ms","p99"):.3f} | '
                  f'{med(t50):.2f} [{min(t50):.2f}-{max(t50):.2f}] | {f("tick_ms","p99"):.2f} | {med([x["throughput_tick_ms"] for x in r]):.2f} | {med([x["span_own_work_ms"]["p50"] for x in r]):.2f} | '
                  f'{med([imbalance_est(x) for x in r]):.2f} | {med([x["bytes_per_tick"]["p50"] for x in r]):.0f} | {med([x["bytes_per_decider_tick"]["p50"] for x in r]):.0f} | '
                  f'{med([x["messages_per_tick_p50"] for x in r]):.0f} | {"yes" if all(x["equivalence"]["zero"] for x in r) else "NO"} | {(f"{med(cpu):.0f}" if cpu else "-")} |')
        print()
    print('## Table 2 - D deciders against one decider and against the ideal split (tick p50, median over clean runs, barrier global)')
    print('| N | T1 (D=1) ms | D | tick ms | speedup T1/T(D) | ideal T1/D ms | efficiency (ideal / tick) | exchange + barrier + sequencer ms | overhead share of tick | beats one decider? | tick p99: D vs 1 |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for N in sorted({k[0] for k in keys}):
        base = g.get((N, 1, 'merge', 3.0, True, 'global'))
        if not base:
            continue
        t1, t1p = med([x['tick_ms']['p50'] for x in base]), med([x['tick_ms']['p99'] for x in base])
        for D in (2, 4, 8):
            r = g.get((N, D, 'merge', 3.0, True, 'global'))
            if not r:
                continue
            tD, tDp = med([x['tick_ms']['p50'] for x in r]), med([x['tick_ms']['p99'] for x in r])
            ov = med([x['exchange_ms']['p50'] + x['barrier_ms']['p50'] + x['sequencer_ms']['p50'] for x in r])
            print(f'| {N} | {t1:.2f} | {D} | {tD:.2f} | {t1 / tD:.2f} | {t1 / D:.2f} | {100 * t1 / D / tD:.0f} % | {ov:.2f} | {100 * ov / tD:.0f} % | {"yes" if tD < t1 else "NO"} | {tDp:.1f} vs {t1p:.1f} |')
    print()
    print('## Table 3 - variants (clean runs; baseline = the main configuration of the same N, D)')
    print('| N | D | variant | runs | tick p50/p99 | compute | exchange | barrier | sequencer mean | decider 0 sequencer p50/p99 | bytes/tick | equivalence 0 | baseline tick p50/p99 |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k in keys:
        if MAIN(k) and k[5] == 'global':
            continue
        if k[5] != 'global' and MAIN(k):
            continue
        r = g[k]
        f = lambda key, q: med([y[key][q] for y in r])
        d0 = [y['per_decider'][0] for y in r]
        bs = g.get((k[0], k[1], 'merge', 3.0, True, 'global'))
        bl = f'{med([y["tick_ms"]["p50"] for y in bs]):.2f}/{med([y["tick_ms"]["p99"] for y in bs]):.2f}' if bs else '-'
        print(f'| {k[0]} | {k[1]} | {k[2]} halo {k[3]:g}{"" if k[4] else " no-verify"}{"" if k[5] == "global" else " nb"} | {len(r)} | {f("tick_ms","p50"):.2f}/{f("tick_ms","p99"):.2f} | {f("compute_ms","p50"):.2f} | {f("exchange_ms","p50"):.2f} | {f("barrier_ms","p50"):.2f} | '
              f'{f("sequencer_ms","p50"):.3f} | {med([d["sequencer_p50"] for d in d0]):.3f}/{med([d["sequencer_p99"] for d in d0]):.3f} | {med([y["bytes_per_tick"]["p50"] for y in r]):.0f} | '
              f'{"yes" if all(y["equivalence"]["zero"] for y in r) else ("NO" if k[4] else "unverified")} | {bl} |')
    print()
    print('## Table 5 - spread of all runs, including ungated ones (tick p50 ms; min / median / max over runs)')
    print('| N | D | runs | min | median | max |')
    print('|---|---|---|---|---|---|')
    for k in sorted(allg):
        if not MAIN(k) or k[5] != 'global':
            continue
        t = [x['tick_ms']['p50'] for x in allg[k]]
        print(f'| {k[0]} | {k[1]} | {len(t)} | {min(t):.2f} | {med(t):.2f} | {max(t):.2f} |')
    print()
    e = RAW / 'empty_latency.json'
    if e.exists():
        d = json.loads(e.read_text())
        print('## Table 4 - exchange + barrier latency with an empty simulation (2 exchange points per tick)')
        print('| D | transport | rows halo / snap per pair | payload B per pair per tick | tick p50 ms | tick p99 ms | per exchange point p50 ms | barrier wait p50 (both) |')
        print('|---|---|---|---|---|---|---|---|')
        for x in d['rows']:
            if x.get('skipped'):
                print(f'| {x["D"]} | skipped ({x["skipped"]}) | | | | | | |')
                continue
            print(f'| {x["D"]} | {x["kind"]} | {x["rows_halo"]} / {x["rows_snap"]} | {x["payload_bytes_per_pair_per_tick"]} | {x["tick_ms"]["p50"]:.3f} | {x["tick_ms"]["p99"]:.3f} | '
                  f'{x["per_exchange_point_ms_p50"]:.3f} | {x.get("wait_ms_both_barriers", {}).get("p50", "-")} |')
    f6 = RAW / 'feed_merge.json'
    if f6.exists():
        print()
        print('## Table 6 - merging the D owned arrays into one state feed (24 B per body: id, x, y, hp), one process')
        print('| N | D | feed bytes | merge ms p50 | p99 | identical to single array |')
        print('|---|---|---|---|---|---|')
        for x in json.loads(f6.read_text()):
            print(f'| {x["N"]} | {x["D"]} | {x["bytes"]} | {x["merge_ms"]["p50"]:.3f} | {x["merge_ms"]["p99"]:.3f} | {x["identical_to_single_array"]} |')


def trim_raw():
    """keep the raw directory small: gated runs keep two per-tick series (slowest-decider tick ms, bytes), ungated runs keep the summary only"""
    for p in sorted(RAW.glob('n*.json')):
        d = json.loads(p.read_text())
        s = d['summary']
        if s.get('skipped'):
            continue
        ser = d.get('series', {})
        if is_gated(s['name']):
            ser = {k: v for k, v in ser.items() if k in ('tick_sys_ms', 'bytes')}
        elif not s['name'].endswith('_neg'):
            ser = {}
        body = json.dumps(s, indent=1)
        rows = ',\n'.join('"%s": %s' % (kk, json.dumps(v, separators=(',', ':'))) for kk, v in ser.items())
        p.write_text('{\n"summary": ' + body + ',\n"series": {\n' + rows + '\n}\n}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--deciders', default='1,2')
    ap.add_argument('--ticks', type=int, default=300)
    ap.add_argument('--halo', type=float, default=3.0)
    ap.add_argument('--seq', default='merge', choices=['merge', 'central'])
    ap.add_argument('--no-verify', action='store_true')
    ap.add_argument('--barrier', default='global', choices=['global', 'neighbor'])
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--empty', action='store_true')
    ap.add_argument('--suite', default='', help='clear-window suite: comma list of N')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--variants', action='store_true')
    ap.add_argument('--feed', action='store_true')
    ap.add_argument('--trim', action='store_true')
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--summarize', action='store_true')
    args = ap.parse_args()
    RAW.mkdir(exist_ok=True)
    if args.summarize:
        summarize()
        return
    if args.empty:
        run_empty(args.force)
        return
    if args.trim:
        trim_raw()
        return
    if args.feed:
        out = [feed_bench(N, D) for N in (2000, 10000, 20000) for D in (1, 2, 4, 8)]
        (RAW / 'feed_merge.json').write_text(json.dumps(out, indent=1))
        for o in out:
            print(o, flush=True)
        return
    if args.suite:
        Ns = [int(x) for x in args.suite.split(',')]
        Ds = [int(x) for x in args.deciders.split(',')]
        for N in Ns:
            b.prepare(N, 'front', args.ticks, b.HP0, CACHE, [0])
        if args.variants:
            run_suite(Ns, Ds, VARIANT_SPECS, args.reps, args.ticks, 'v')
        else:
            run_suite(Ns, Ds, MAIN_SPECS, args.reps, args.ticks)
        return
    if args.prepare:
        b.prepare(args.n, 'front', args.ticks, b.HP0, CACHE, [0])
        return
    b.prepare(args.n, 'front', args.ticks, b.HP0, CACHE, [0])
    for D in [int(x) for x in args.deciders.split(',')]:
        run_config(args.n, D, args.ticks, args.halo, args.seq, not args.no_verify, args.tag, args.force, args.barrier)


if __name__ == '__main__':
    mp.freeze_support()
    main()
