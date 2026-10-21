"""Dense melee numpy harness (vectorized numpy).

Measures the SERVER-SIDE work of a dense 100v100 melee brawl, per tick, end to end:
  input recv (loopback UDP) -> movement + crowd contacts -> history ring ->
  lag-compensated weapon sweeps vs multi-capsule hurtboxes (3D, rotating blades) ->
  guard/parry/damage + AoE fanout + status ticking -> snapshot quantize + delta encode ->
  UDP send (loopback).

What the numbers mean
  * They are an UPPER BOUND for a data-oriented implementation: every stage runs as
    batched numpy calls driven by a Python interpreter (single thread). A compiled SoA/SIMD
    implementation does the same arithmetic without interpreter and temporary-array overhead.
  * They are NOT an engine, a real 200-machine harness, a client/render benchmark or a playtest.
  * Correctness is checked against a slow exhaustive double-precision oracle every tick
    (suite `verify`), including historical (rewound) broad phase and snapshot decode.

Run from the repository root:
  python part1/numpy-harness/dense_melee_bench.py verify
  python part1/numpy-harness/dense_melee_bench.py all
Writes <suite>_results.md / .json beside this file. Fixed seeds; standard library + numpy only.
"""
import ctypes
import gc
import json
import math
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

DT = 1.0 / 30.0
HIST = 32
WINDUP, ACTIVE, RECOVER = 6, 4, 5
FRAMES = WINDUP + ACTIVE + RECOVER          # 15 ticks per swing = 2 attacks/s at 30 Hz
NUM_ACTIONS = 32
BODY_R = 0.35                               # crowd disc radius (m)
SKIN = 0.30                                 # neighbour-list skin (m)
ROOT_Z = 1.0
MAX_STATUS = 32
F32 = np.float32

# body capsules in root space: x forward, y left, z up  (x0,y0,z0,x1,y1,z1,r)
BODY8 = np.array([
    [0.00, 0.00, 1.62, 0.00, 0.00, 1.70, 0.12],    # head
    [0.00, 0.00, 1.05, 0.00, 0.00, 1.45, 0.20],    # torso
    [0.00, 0.00, 0.85, 0.00, 0.00, 0.95, 0.18],    # pelvis
    [0.00, 0.24, 1.42, 0.10, 0.30, 1.00, 0.07],    # left arm
    [0.00, -0.24, 1.42, 0.20, -0.28, 1.08, 0.07],  # right arm
    [0.00, 0.10, 0.85, 0.02, 0.12, 0.08, 0.09],    # left leg
    [0.00, -0.10, 0.85, 0.02, -0.12, 0.08, 0.09],  # right leg
    [0.05, -0.25, 1.05, 0.25, -0.30, 0.95, 0.06],  # right forearm/hand
], dtype=np.float64)

# length, arc degrees, blade radius, thrust
WEAPONS = [(0.95, 140.0, 0.04, False), (1.50, 160.0, 0.05, False), (2.30, 0.0, 0.04, True), (0.85, 120.0, 0.06, False)]


# ----------------------------------------------------------------------------- baked poses
def blade_at(action, u):
    length, arc, _, thrust = WEAPONS[action % 4]
    variant = action // 4
    u = min(max(u, 0.0), 1.0)
    height = 1.15 + 0.1 * (variant % 3)
    if thrust:
        x0 = 0.10 + 0.45 * u
        return np.array([x0, -0.18, 1.30, x0 + length * (0.55 + 0.45 * u), -0.18 + 0.05 * (variant % 2), height])
    half = math.radians(arc) * 0.5
    th0 = half if variant & 1 else -half
    th = th0 + (-th0 - th0) * u
    c, s = math.cos(th), math.sin(th)
    x0, y0 = 0.05 + 0.45 * c, -0.20 + 0.45 * s
    return np.array([x0, y0, 1.40, x0 + length * c, y0 + length * s, height - 0.2 * (u - 0.5)])


def point_seg_dist(p, seg):
    a, b = seg[:3], seg[3:]
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(a + ab * t - p))


def bake(H, SUB):
    hurt = np.zeros((NUM_ACTIONS, FRAMES, H, 7))
    sweep = np.zeros((NUM_ACTIONS, FRAMES, SUB, 6))
    sweep_r = np.zeros((NUM_ACTIONS, FRAMES))
    sc = np.zeros((NUM_ACTIONS, FRAMES, 3))
    sbr = np.zeros((NUM_ACTIONS, FRAMES))
    root_r = 0.0
    max_infl = 0.0
    max_reach = 0.0
    for a in range(NUM_ACTIONS):
        for f in range(FRAMES):
            ph = f / FRAMES
            lean = 0.08 * math.sin(math.pi * ph) * (1 + 0.2 * (a % 3))
            swing = 0.15 * math.sin(2 * math.pi * ph + a)
            for h in range(H):
                b = BODY8[h // 2 if H == 16 else h].copy()
                if H == 16:
                    mid = (b[0:3] + b[3:6]) / 2
                    if h % 2 == 0:
                        b[3:6] = mid
                    else:
                        b[0:3] = mid
                if b[2] > 0.9:
                    b[0] += lean
                if b[5] > 0.9:
                    b[3] += lean
                if abs(b[1]) > 0.2:
                    b[3] += swing
                hurt[a, f, h] = b
                for e in (0, 3):
                    root_r = max(root_r, math.dist(b[e:e + 3], (0, 0, ROOT_Z)) + b[6])
            if not (WINDUP <= f < WINDUP + ACTIVE):
                continue
            k = f - WINDUP
            us = [(k + s / (SUB - 1)) / ACTIVE for s in range(SUB)]
            segs = [blade_at(a, u) for u in us]
            # conservative sampling inflation: max distance from any intermediate blade point
            # (dense in time and along the blade) to the nearer of the two bracketing sample segments
            infl = 0.0
            for s in range(SUB - 1):
                for m in range(1, 24):
                    seg = blade_at(a, us[s] + (us[s + 1] - us[s]) * m / 24)
                    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
                        p = seg[:3] * (1 - t) + seg[3:] * t
                        infl = max(infl, min(point_seg_dist(p, segs[s]), point_seg_dist(p, segs[s + 1])))
            pts = np.array([sg[:3] for sg in segs] + [sg[3:] for sg in segs])
            centre = pts.mean(axis=0)
            r = WEAPONS[a % 4][2] + infl
            sweep[a, f] = np.array(segs)
            sweep_r[a, f] = r
            sc[a, f] = centre
            sbr[a, f] = np.max(np.linalg.norm(pts - centre, axis=1)) + r
            max_infl = max(max_infl, infl)
            max_reach = max(max_reach, max(math.hypot(sg[3], sg[4]) for sg in segs))
    return dict(hurt=hurt.astype(F32), sweep=sweep.astype(F32), sweep_r=sweep_r.astype(F32), sc=sc.astype(F32),
                sbr=sbr.astype(F32), sbr_max=float(sbr.max()), root_r=float(root_r), infl=max_infl, reach=max_reach, H=H, SUB=SUB)


# ----------------------------------------------------------------------------- geometry kernels
def seg_seg_d2_fast(Px, Py, Pz, Dx, Dy, Dz, A, Qx, Qy, Qz, Ex, Ey, Ez):
    """Branchless closest-distance^2 between segments P+Ds and Q+Et (broadcasting float32)."""
    rx, ry, rz = Px - Qx, Py - Qy, Pz - Qz
    e = Ex * Ex + Ey * Ey + Ez * Ez + F32(1e-8)
    f = Ex * rx + Ey * ry + Ez * rz
    c = Dx * rx + Dy * ry + Dz * rz
    b = Dx * Ex + Dy * Ey + Dz * Ez
    denom = A * e - b * b
    s = np.clip((b * f - c * e) / np.maximum(denom, F32(1e-7)), 0, 1)
    s = np.where(denom > F32(1e-7), s, F32(0))
    t = np.clip((b * s + f) / e, 0, 1)
    s = np.clip((b * t - c) / A, 0, 1)
    cx = rx + Dx * s - Ex * t
    cy = ry + Dy * s - Ey * t
    cz = rz + Dz * s - Ez * t
    return cx * cx + cy * cy + cz * cz


def seg_seg_d2_ref(P, D, Q, E):
    """Exact Ericson RTCD 5.1.9 in float64 with explicit branches (oracle). Arrays (...,3)."""
    r = P - Q
    a = np.sum(D * D, -1)
    e = np.sum(E * E, -1)
    f = np.sum(E * r, -1)
    c = np.sum(D * r, -1)
    b = np.sum(D * E, -1)
    eps = 1e-12
    denom = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(denom != 0, np.clip((b * f - c * e) / denom, 0, 1), 0.0)
        t = (b * s + f) / e
        s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s))
        t = np.clip(t, 0, 1)
        # degenerate branches
        s = np.where(a <= eps, 0.0, s)
        t = np.where(a <= eps, np.clip(f / e, 0, 1), t)
        t = np.where(e <= eps, 0.0, t)
        s = np.where((e <= eps) & (a > eps), np.clip(-c / a, 0, 1), s)
    d = r + D * s[..., None] - E * t[..., None]
    return np.sum(d * d, -1)


def grid_pairs(qx, qy, tx, ty, cell):
    """All (query, target) index pairs whose cells are within one cell. cell must be >= query radius."""
    ox = min(float(qx.min()), float(tx.min())) - cell
    oy = min(float(qy.min()), float(ty.min())) - cell
    tcx = np.floor((tx - ox) / cell).astype(np.int64)
    tcy = np.floor((ty - oy) / cell).astype(np.int64)
    tkey = (tcx << 21) + tcy
    order = np.argsort(tkey, kind="stable")
    sk = tkey[order]
    qcx = np.floor((qx - ox) / cell).astype(np.int64)
    qcy = np.floor((qy - oy) / cell).astype(np.int64)
    keys = ((qcx[:, None] + OFF_X) << 21) + (qcy[:, None] + OFF_Y)
    keys = keys.ravel()
    lo = np.searchsorted(sk, keys, "left")
    hi = np.searchsorted(sk, keys, "right")
    cnt = hi - lo
    tot = int(cnt.sum())
    if tot == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    qi = np.repeat(np.repeat(np.arange(len(qx)), 9), cnt)
    start = np.repeat(lo, cnt)
    base = np.repeat(np.cumsum(cnt) - cnt, cnt)
    tj = order[start + (np.arange(tot) - base)]
    return qi, tj


OFF_X = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int64)
OFF_Y = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int64)

# ----------------------------------------------------------------------------- snapshot codec
# fields: x, y, yaw, act, frame, guard, hp, stam, flags
FIELD_BITS = [(2, 8), (3, 10), (4, 4), (5, 3), (6, 12), (7, 8), (8, 8)]
SHIFTS = np.arange(18, dtype=np.uint64)
SHIFTS_I = np.arange(18, dtype=np.int32)


def encode_many(cur, bases):
    """cur (N,9) int64, bases (U,N,9) int64 -> list of U byte strings (LSB-first bitstream)."""
    U, N, _ = bases.shape
    same = np.all(cur[None] == bases, axis=2)
    vals = np.zeros((U, N, 10), np.uint64)
    nb = np.zeros((U, N, 10), np.int32)
    vals[:, :, 0] = (~same).astype(np.uint64)
    nb[:, :, 0] = 1
    for k, fi in enumerate((0, 1)):
        d = cur[None, :, fi] - bases[:, :, fi]
        zero = d == 0
        small = (~zero) & (d >= -64) & (d < 64)
        v = np.where(zero, 0, np.where(small, 1 | ((d + 64) << 2), 3 | (cur[None, :, fi] << 2)))
        vals[:, :, 1 + k] = v.astype(np.uint64)
        nb[:, :, 1 + k] = np.where(zero, 1, np.where(small, 9, 18))
    for k, (fi, bits) in enumerate(FIELD_BITS):
        eq = cur[None, :, fi] == bases[:, :, fi]
        vals[:, :, 3 + k] = np.where(eq, 0, 1 | (cur[None, :, fi] << 1)).astype(np.uint64)
        nb[:, :, 3 + k] = np.where(eq, 1, bits + 1)
    nb[:, :, 1:] *= (~same)[:, :, None]
    bitsarr = ((vals[..., None] >> SHIFTS) & np.uint64(1)).astype(np.uint8)
    mask = SHIFTS_I < nb[..., None]
    return [np.packbits(bitsarr[u][mask[u]], bitorder="little").tobytes() for u in range(U)]


def decode_all(data, base, n):
    """Pure-Python reference decoder: base (n,9) ints -> decoded (n,9)."""
    acc = int.from_bytes(data, "little")
    pos = 0

    def get(k):
        nonlocal pos
        v = (acc >> pos) & ((1 << k) - 1)
        pos += k
        return v

    out = [list(map(int, base[i])) for i in range(n)]
    for i in range(n):
        if not get(1):
            continue
        for fi in (0, 1):
            if get(1):
                if get(1):
                    out[i][fi] = get(16)
                else:
                    out[i][fi] = int(base[i][fi]) + get(7) - 64
        for fi, bits in FIELD_BITS:
            if get(1):
                out[i][fi] = get(bits)
    return out


# ----------------------------------------------------------------------------- simulation
class Sim:
    def __init__(self, cfg):
        self.cfg = cfg
        self.N = N = cfg["N"]
        self.rng = np.random.default_rng(cfg["seed"])
        self.T = bake(cfg["H"], cfg["SUB"])
        r = self.rng
        self.team = (np.arange(N) & 1).astype(np.int8)
        side = np.where(self.team == 1, 1.0, -1.0)
        if cfg["arena"] == "disc":
            R = cfg["disc_r"]
            ang = r.uniform(-math.pi / 2, math.pi / 2, N)
            rad = R * np.sqrt(r.uniform(0, 1, N)) * 0.95
            self.px = (side * np.abs(rad * np.cos(ang))).astype(F32)
            self.py = (rad * np.sin(ang)).astype(F32)
        else:
            L, W = cfg["corr_l"], cfg["corr_w"]
            self.px = (side * (1.0 + r.uniform(0, 1, N) * (L / 2 - 2))).astype(F32)
            self.py = ((r.uniform(0, 1, N) - 0.5) * (W - 1)).astype(F32)
        self.fc = np.ones(N, F32)
        self.fs = np.zeros(N, F32)
        self.desx = np.zeros(N, F32)
        self.desy = np.zeros(N, F32)
        self.lag = (1 + r.integers(0, cfg["max_lag"], N)).astype(np.int64)
        self.rtt = np.maximum(1, 2 * self.lag - 2)
        self.act = r.integers(0, NUM_ACTIONS, N).astype(np.int64)
        self.act_start = np.zeros(N, np.int64) if cfg["burst"] else -r.integers(0, FRAMES, N).astype(np.int64)
        self.guard = np.zeros(N, bool)
        self.guard_start = np.full(N, -1000, np.int64)
        self.hp = np.full(N, 4000, np.int64)
        self.stam = np.full(N, 255, np.int64)
        self.nearest = (np.arange(N) + 1) % N
        # history ring [HIST, N]
        self.hx = np.zeros((HIST, N), F32)
        self.hy = np.zeros((HIST, N), F32)
        self.hc = np.ones((HIST, N), F32)
        self.hs = np.zeros((HIST, N), F32)
        self.hact = np.zeros((HIST, N), np.int64)
        self.hframe = np.zeros((HIST, N), np.int64)
        self.hguard = np.zeros((HIST, N), bool)
        self.hguard_start = np.full((HIST, N), -1000, np.int64)
        self.hist_rad = np.zeros(N, F32)
        # per-swing dedupe
        self.hit_matrix = np.zeros((N, N), bool)
        self.swing_of = np.full(N, -99999, np.int64)
        # statuses
        self.st_rem = np.zeros((N, MAX_STATUS), np.int32)
        self.st_type = np.zeros((N, MAX_STATUS), np.int32)
        self.st_count = np.zeros(N, np.int64)
        self.stat_sum = np.zeros((N, 4), F32)
        # snapshot ring [HIST, N, 9]
        self.qring = np.zeros((HIST, N, 9), np.int64)
        self.last_ack = np.full(N, -10 ** 9, np.int64)
        if cfg["arena"] == "disc":
            self.min_x = self.min_y = -cfg["disc_r"] - 2
        else:
            self.min_x, self.min_y = -cfg["corr_l"] / 2 - 2, -cfg["corr_w"] / 2 - 2
        self.payloads = [b""] * N
        self.distinct_baselines = []
        self.net = cfg["net"]
        if self.net:
            self.srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.cli = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            for s in (self.srv, self.sink, self.cli):
                s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 * 1024 * 1024)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32 * 1024 * 1024)
                s.bind(("127.0.0.1", 0))
                s.setblocking(False)
            self.srv_addr = self.srv.getsockname()
            self.sink_addr = self.sink.getsockname()
            self.rx_buf = bytearray(256)
        self.pkt_dtype = np.dtype([("id", "<i4"), ("tick", "<i4"), ("mx", "<i2"), ("my", "<i2"), ("yaw", "u1"),
                                   ("a", "<u2"), ("g", "u1"), ("pad", "V8")])

    def close(self):
        if self.net:
            for s in (self.srv, self.sink, self.cli):
                s.close()

    def frame_of(self, tick):
        return np.clip(tick - self.act_start, 0, FRAMES - 1)

    # ------------------------------------------------------------ untimed client/bot side
    def bot_inputs(self, tick):
        N, cfg = self.N, self.cfg
        if tick % 3 == 1:
            enemy = np.nonzero(self.team == 1)[0], np.nonzero(self.team == 0)[0]
            for t in (0, 1):
                me = np.nonzero(self.team == t)[0]
                en = enemy[t]
                dx = self.px[en][None, :] - self.px[me][:, None]
                dy = self.py[en][None, :] - self.py[me][:, None]
                self.nearest[me] = en[np.argmin(dx * dx + dy * dy, axis=1)]
        j = self.nearest
        dx = self.px[j] - self.px
        dy = self.py[j] - self.py
        d = np.sqrt(dx * dx + dy * dy) + 1e-6
        mv = (d > 1.1).astype(F32)
        pk = np.zeros(N, self.pkt_dtype)
        pk["id"] = np.arange(N)
        pk["tick"] = tick
        pk["mx"] = (dx / d * 32767 * mv).astype(np.int16)
        pk["my"] = (dy / d * 32767 * mv).astype(np.int16)
        pk["yaw"] = (((np.arctan2(dy, dx) + math.pi) / (2 * math.pi) * 256).astype(np.int64) & 255).astype(np.uint8)
        start = (tick % FRAMES == 0) if cfg["burst"] else (tick - self.act_start >= FRAMES)
        start = np.broadcast_to(start, (N,))
        pk["a"] = np.where(start, 1 + self.rng.integers(0, NUM_ACTIONS, N), 0)
        idle = tick - self.act_start >= FRAMES
        pk["g"] = ((~start) & idle & (self.rng.integers(0, 8, N) < 3)).astype(np.uint8)
        self.pk = pk
        if self.net:
            raw = pk.tobytes()
            for i in range(N):
                self.cli.sendto(raw[24 * i:24 * i + 24], self.srv_addr)

    # ------------------------------------------------------------ stage: input
    def stage_input(self, tick):
        if self.net:
            chunks = []
            recv_into, buf = self.srv.recv_into, self.rx_buf
            while True:
                try:
                    n = recv_into(buf)
                except (BlockingIOError, OSError):
                    break
                if n >= 24:
                    chunks.append(bytes(buf[:24]))
            if not chunks:
                return
            pk = np.frombuffer(b"".join(chunks), self.pkt_dtype)
        else:
            pk = self.pk
        i = pk["id"].astype(np.int64)
        yaw = pk["yaw"].astype(F32) / 256 * (2 * math.pi) - math.pi
        self.fc[i] = np.cos(yaw)
        self.fs[i] = np.sin(yaw)
        self.desx[i] = pk["mx"].astype(F32) / 32767
        self.desy[i] = pk["my"].astype(F32) / 32767
        a = pk["a"].astype(np.int64)
        can = (a > 0) & (tick - self.act_start[i] >= FRAMES)
        ii = i[can]
        self.act[ii] = a[can] - 1
        self.act_start[ii] = tick
        self.guard[ii] = False
        g = pk["g"] > 0
        newg = g & ~self.guard[i]
        self.guard_start[i[newg]] = tick
        self.guard[i[g]] = True
        drop = (~g) & (tick - self.act_start[i] >= FRAMES)
        self.guard[i[drop]] = False

    # ------------------------------------------------------------ stage: movement + crowd
    def walls(self):
        cfg = self.cfg
        if cfg["arena"] == "disc":
            r = np.sqrt(self.px * self.px + self.py * self.py)
            lim = cfg["disc_r"] - BODY_R
            k = np.where(r > lim, lim / np.maximum(r, 1e-6), 1.0).astype(F32)
            self.px *= k
            self.py *= k
        else:
            np.clip(self.px, -cfg["corr_l"] / 2 + BODY_R, cfg["corr_l"] / 2 - BODY_R, out=self.px)
            np.clip(self.py, -cfg["corr_w"] / 2 + BODY_R, cfg["corr_w"] / 2 - BODY_R, out=self.py)

    def stage_move(self, tick):
        N = self.N
        sp = np.where(tick - self.act_start < FRAMES, F32(1.5), F32(4.0))
        self.px += self.desx * sp * F32(DT)
        self.py += self.desy * sp * F32(DT)
        reach = 2 * BODY_R + SKIN
        qi, tj = grid_pairs(self.px, self.py, self.px, self.py, reach)
        m = qi != tj
        i, j = qi[m], tj[m]
        dx = self.px[i] - self.px[j]
        dy = self.py[i] - self.py[j]
        keep = dx * dx + dy * dy < reach * reach
        i, j = i[keep], j[keep]           # neighbour list reused by all iterations (Verlet skin)
        min_d = F32(2 * BODY_R)
        for _ in range(self.cfg["crowd_iters"]):
            dx = self.px[i] - self.px[j]
            dy = self.py[i] - self.py[j]
            d2 = dx * dx + dy * dy
            ov = d2 < min_d * min_d
            ii, jj, dxx, dyy = i[ov], j[ov], dx[ov], dy[ov]
            d = np.sqrt(d2[ov])
            tiny = d < 1e-4
            if tiny.any():
                dxx = np.where(tiny, np.where(ii < jj, F32(1e-2), F32(-1e-2)), dxx)
                dyy = np.where(tiny, F32(0), dyy)
                d = np.where(tiny, F32(1e-2), d)
            pen = (min_d - d) * F32(0.5) / d
            cx = np.bincount(ii, weights=dxx * pen, minlength=N)
            cy = np.bincount(ii, weights=dyy * pen, minlength=N)
            w = 1.0 / np.sqrt(np.maximum(np.bincount(ii, minlength=N), 1))
            self.px += (cx * w).astype(F32)
            self.py += (cy * w).astype(F32)
            self.walls()

    def penetration(self):
        qi, tj = grid_pairs(self.px, self.py, self.px, self.py, 2 * BODY_R)
        m = qi < tj
        dx = self.px[qi[m]] - self.px[tj[m]]
        dy = self.py[qi[m]] - self.py[tj[m]]
        d = np.sqrt(dx * dx + dy * dy)
        p = 2 * BODY_R - d[d < 2 * BODY_R]
        return (float(p.mean()) if p.size else 0.0), (float(p.max()) if p.size else 0.0)

    # ------------------------------------------------------------ stage: history
    def stage_history(self, tick):
        s = tick % HIST
        self.hx[s] = self.px
        self.hy[s] = self.py
        self.hc[s] = self.fc
        self.hs[s] = self.fs
        self.hact[s] = self.act
        self.hframe[s] = self.frame_of(tick)
        self.hguard[s] = self.guard
        self.hguard_start[s] = self.guard_start
        L = min(self.cfg["max_lag"], tick - 1)
        if L > 0:
            slots = (tick - np.arange(1, L + 1)) % HIST
            dx = self.hx[slots] - self.px
            dy = self.hy[slots] - self.py
            self.hist_rad = np.sqrt((dx * dx + dy * dy).max(axis=0))

    # ------------------------------------------------------------ stage: hit detection
    def active_attackers(self, tick):
        f = self.frame_of(tick)
        A = np.nonzero((f >= WINDUP) & (f < WINDUP + ACTIVE))[0]
        return A, f[A]

    def stage_hit(self, tick):
        T = self.T
        A, fa = self.active_attackers(tick)
        self.n_active = len(A)
        if len(A) == 0:
            self.n_cand = self.n_narrow = self.n_exact = 0
            return np.empty(0, np.int64), np.empty(0, np.int64)
        aa = self.act[A]
        c, s = self.fc[A], self.fs[A]
        lc = T["sc"][aa, fa]
        pxa, pya = self.px[A], self.py[A]
        scx = pxa + c * lc[:, 0] - s * lc[:, 1]
        scy = pya + s * lc[:, 0] + c * lc[:, 1]
        scz = lc[:, 2]
        sbr = T["sbr"][aa, fa]
        rb = T["sweep_r"][aa, fa]
        vslot = (tick - self.lag[A]) % HIST
        qr = T["sbr_max"] + T["root_r"] + float(self.hist_rad.max())
        # level 1: grid on CURRENT positions with query radius expanded by the max history displacement,
        # then an exact Combat Root sphere test at the attacker's REWOUND view tick (historical broad phase)
        ai, tj = grid_pairs(scx, scy, self.px, self.py, qr)
        keep = self.team[tj] != self.team[A[ai]]
        ai, tj = ai[keep], tj[keep]
        vs = vslot[ai]
        rx, ry = self.hx[vs, tj], self.hy[vs, tj]
        dx, dy, dz = rx - scx[ai], ry - scy[ai], ROOT_Z - scz[ai]
        rs = sbr[ai] + T["root_r"]
        keep = dx * dx + dy * dy + dz * dz <= rs * rs
        ai, tj, vs, rx, ry = ai[keep], tj[keep], vs[keep], rx[keep], ry[keep]
        P = len(ai)
        self.n_cand = P
        self.n_narrow = P * T["H"] * T["SUB"]
        self.n_exact = 0
        if P == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        # level 2: all candidate hurt capsules vs all sweep sub-segments, batched
        caps = T["hurt"][self.hact[vs, tj], self.hframe[vs, tj]]          # (P,H,7)
        tc, ts = self.hc[vs, tj][:, None], self.hs[vs, tj][:, None]
        rxx, ryy = rx[:, None], ry[:, None]
        X0 = rxx + tc * caps[:, :, 0] - ts * caps[:, :, 1]
        Y0 = ryy + ts * caps[:, :, 0] + tc * caps[:, :, 1]
        X1 = rxx + tc * caps[:, :, 3] - ts * caps[:, :, 4]
        Y1 = ryy + ts * caps[:, :, 3] + tc * caps[:, :, 4]
        sw = T["sweep"][aa, fa]                                             # (nA,SUB,6)
        cc, ss = c[:, None], s[:, None]
        SX0 = pxa[:, None] + cc * sw[:, :, 0] - ss * sw[:, :, 1]
        SY0 = pya[:, None] + ss * sw[:, :, 0] + cc * sw[:, :, 1]
        SX1 = pxa[:, None] + cc * sw[:, :, 3] - ss * sw[:, :, 4]
        SY1 = pya[:, None] + ss * sw[:, :, 3] + cc * sw[:, :, 4]
        Dx, Dy, Dz = SX1 - SX0, SY1 - SY0, sw[:, :, 5] - sw[:, :, 2]
        Aa = Dx * Dx + Dy * Dy + Dz * Dz + F32(1e-8)
        if not self.cfg["prefilter"]:
            self.n_exact = self.n_narrow
            g = lambda v: v[ai][:, :, None]
            d2 = seg_seg_d2_fast(g(SX0), g(SY0), g(sw[:, :, 2]), g(Dx), g(Dy), g(Dz), g(Aa),
                                 X0[:, None, :], Y0[:, None, :], caps[:, None, :, 2],
                                 (X1 - X0)[:, None, :], (Y1 - Y0)[:, None, :], (caps[:, :, 5] - caps[:, :, 2])[:, None, :])
            rad = caps[:, None, :, 6] + rb[ai][:, None, None]
            hit = (d2 <= rad * rad).reshape(P, -1).any(axis=1)
            return A[ai[hit]], tj[hit]
        # level 3 (conservative prefilter): bounding sphere of each sweep sub-segment vs bounding sphere of each
        # hurt capsule; the exact segment-segment kernel runs only on the surviving (pair, sub-segment, capsule) triples
        Sx0, Sy0, Sz0 = SX0[ai], SY0[ai], sw[:, :, 2][ai]
        Sdx, Sdy, Sdz, Sa = Dx[ai], Dy[ai], Dz[ai], Aa[ai]
        Srad = F32(0.5) * np.sqrt(Sa) + rb[ai][:, None]
        Scx, Scy, Scz = Sx0 + F32(0.5) * Sdx, Sy0 + F32(0.5) * Sdy, Sz0 + F32(0.5) * Sdz
        Ex, Ey, Ez = X1 - X0, Y1 - Y0, caps[:, :, 5] - caps[:, :, 2]
        Crad = F32(0.5) * np.sqrt(Ex * Ex + Ey * Ey + Ez * Ez) + caps[:, :, 6]
        Ccx, Ccy, Ccz = X0 + F32(0.5) * Ex, Y0 + F32(0.5) * Ey, caps[:, :, 2] + F32(0.5) * Ez
        ddx = Scx[:, :, None] - Ccx[:, None, :]
        ddy = Scy[:, :, None] - Ccy[:, None, :]
        ddz = Scz[:, :, None] - Ccz[:, None, :]
        rr = Srad[:, :, None] + Crad[:, None, :]
        pi, si, hi = np.nonzero(ddx * ddx + ddy * ddy + ddz * ddz <= rr * rr)
        self.n_exact = len(pi)
        if len(pi) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        d2 = seg_seg_d2_fast(Sx0[pi, si], Sy0[pi, si], Sz0[pi, si], Sdx[pi, si], Sdy[pi, si], Sdz[pi, si], Sa[pi, si],
                             X0[pi, hi], Y0[pi, hi], caps[pi, hi, 2], Ex[pi, hi], Ey[pi, hi], Ez[pi, hi])
        rad = caps[pi, hi, 6] + rb[ai][pi]
        hitp = np.zeros(P, bool)
        hitp[pi[d2 <= rad * rad]] = True
        return A[ai[hitp]], tj[hitp]

    def oracle_hits(self, tick):
        """Exhaustive: every active attacker vs every enemy, every capsule, every sweep segment, float64 RTCD."""
        T = self.T
        A, fa = self.active_attackers(tick)
        if len(A) == 0:
            return set()
        aa = self.act[A]
        out = set()
        for k in range(len(A)):
            a = A[k]
            en = np.nonzero(self.team != self.team[a])[0]
            vs = (tick - self.lag[a]) % HIST
            c, s = float(self.fc[a]), float(self.fs[a])
            sw = T["sweep"][aa[k], fa[k]].astype(np.float64)
            P0 = np.stack([self.px[a] + c * sw[:, 0] - s * sw[:, 1], self.py[a] + s * sw[:, 0] + c * sw[:, 1], sw[:, 2]], -1)
            P1 = np.stack([self.px[a] + c * sw[:, 3] - s * sw[:, 4], self.py[a] + s * sw[:, 3] + c * sw[:, 4], sw[:, 5]], -1)
            caps = T["hurt"][self.hact[vs, en], self.hframe[vs, en]].astype(np.float64)
            tc = self.hc[vs, en].astype(np.float64)[:, None]
            ts = self.hs[vs, en].astype(np.float64)[:, None]
            rx = self.hx[vs, en].astype(np.float64)[:, None]
            ry = self.hy[vs, en].astype(np.float64)[:, None]
            Q0 = np.stack([rx + tc * caps[:, :, 0] - ts * caps[:, :, 1], ry + ts * caps[:, :, 0] + tc * caps[:, :, 1], caps[:, :, 2]], -1)
            Q1 = np.stack([rx + tc * caps[:, :, 3] - ts * caps[:, :, 4], ry + ts * caps[:, :, 3] + tc * caps[:, :, 4], caps[:, :, 5]], -1)
            d2 = seg_seg_d2_ref(P0[None, :, None, :], (P1 - P0)[None, :, None, :], Q0[:, None, :, :], (Q1 - Q0)[:, None, :, :])
            rad = caps[:, None, :, 6] + float(T["sweep_r"][aa[k], fa[k]])
            hit = (d2 <= rad * rad).reshape(len(en), -1).any(axis=1)
            out.update((int(a), int(t)) for t in en[hit])
        return out

    # ------------------------------------------------------------ stage: resolve + AoE + statuses
    def stage_resolve(self, tick, ha, ht):
        N = self.N
        new_swing = self.swing_of != self.act_start
        rows = np.nonzero(new_swing)[0]
        if rows.size:
            self.hit_matrix[rows] = False
            self.swing_of[rows] = self.act_start[rows]
        if ha.size:
            fresh = ~self.hit_matrix[ha, ht]
            a, t = ha[fresh], ht[fresh]
            self.hit_matrix[a, t] = True
            vs = (tick - self.lag[a]) % HIST
            dx = self.px[a] - self.hx[vs, t]
            dy = self.py[a] - self.hy[vs, t]
            d = np.sqrt(dx * dx + dy * dy) + 1e-6
            facing = (self.hc[vs, t] * dx + self.hs[vs, t] * dy) / d > 0.5
            blocked = self.hguard[vs, t] & facing
            parry = blocked & ((tick - self.lag[a]) - self.hguard_start[vs, t] <= 6)
            self.act_start[a[parry]] = tick - WINDUP - ACTIVE
            np.subtract.at(self.stam, t[blocked & ~parry], 20)
            np.subtract.at(self.hp, t[~blocked], 120)
            np.maximum(self.stam, 0, out=self.stam)
        # AoE on first active frame for a deterministic subset of swings
        casters = np.nonzero((tick - self.act_start) == WINDUP)[0]
        sel = ((casters.astype(np.uint64) * np.uint64(2654435761) + self.act_start[casters].astype(np.uint64)) % np.uint64(1000)) < int(self.cfg["aoe_frac"] * 1000)
        casters = casters[sel]
        self.n_aoe = 0
        if casters.size:
            cx = self.px[casters] + self.fc[casters] * 2
            cy = self.py[casters] + self.fs[casters] * 2
            qi, tj = grid_pairs(cx, cy, self.px, self.py, 4.0)
            ok = (self.team[tj] != self.team[casters[qi]])
            qi, tj = qi[ok], tj[ok]
            ddx, ddy = self.px[tj] - cx[qi], self.py[tj] - cy[qi]
            ok = ddx * ddx + ddy * ddy <= 16.0
            qi, tj = qi[ok], tj[ok]
            self.n_aoe = int(tj.size)
            if tj.size:
                order = np.argsort(tj, kind="stable")
                tgt, src = tj[order], casters[qi[order]]
                idx = np.arange(tgt.size)
                first = np.r_[True, tgt[1:] != tgt[:-1]]
                rank = idx - np.maximum.accumulate(np.where(first, idx, 0))
                slot = self.st_count[tgt] + rank
                slot = np.where(slot < MAX_STATUS, slot, src % MAX_STATUS)
                self.st_rem[tgt, slot] = 150
                self.st_type[tgt, slot] = src & 7
                self.st_count = np.minimum(self.st_count + np.bincount(tgt, minlength=N), MAX_STATUS)
        valid = np.arange(MAX_STATUS)[None, :] < self.st_count[:, None]
        self.st_rem -= valid
        alive = valid & (self.st_rem > 0)
        dot = alive & (self.st_rem % 15 == 0)
        self.hp -= 5 * dot.sum(axis=1)
        ty = self.st_type
        self.stat_sum[:, 0] = (alive & ((ty & 1) > 0)).sum(1) * 0.05
        self.stat_sum[:, 1] = (alive & ((ty & 2) > 0)).sum(1) * 0.03
        self.stat_sum[:, 2] = (alive & ((ty & 4) > 0)).sum(1) * 0.02
        self.stat_sum[:, 3] = alive.sum(1) * 0.01
        order = np.argsort(~alive, axis=1, kind="stable")
        self.st_rem = np.take_along_axis(np.where(alive, self.st_rem, 0), order, 1)
        self.st_type = np.take_along_axis(self.st_type, order, 1)
        self.st_count = alive.sum(1)
        self.hp[self.hp <= 0] = 4000
        regen = (~self.guard) & (self.stam < 255)
        self.stam[regen] += 1

    # ------------------------------------------------------------ stage: snapshot
    def quantize(self, tick):
        q = self.qring[tick % HIST]
        q[:, 0] = np.clip(((self.px - self.min_x) / 0.02).astype(np.int64), 0, 65535)
        q[:, 1] = np.clip(((self.py - self.min_y) / 0.02).astype(np.int64), 0, 65535)
        q[:, 2] = ((np.arctan2(self.fs, self.fc) + math.pi) / (2 * math.pi) * 256).astype(np.int64) & 255
        q[:, 3] = self.act
        q[:, 4] = self.frame_of(tick)
        q[:, 5] = self.guard
        q[:, 6] = np.clip(self.hp, 0, 4095)
        q[:, 7] = self.stam
        q[:, 8] = self.st_count & 255
        return q

    def stage_snapshot(self, tick):
        N = self.N
        cur = self.quantize(tick)
        acked = tick - self.rtt
        got = (acked >= 0) & (self.rng.uniform(0, 1, N) >= self.cfg["loss"])
        self.last_ack = np.where(got, np.maximum(self.last_ack, acked), self.last_ack)
        age = tick - self.last_ack
        base = np.where((age > 0) & (age < HIST), self.last_ack, -1)
        self.base = base
        zero = np.zeros((N, 9), np.int64)

        def base_state(b):
            return self.qring[b % HIST] if b >= 0 else zero

        hdr = np.zeros((N, 32), np.uint8)
        hv = hdr.view(np.int32)
        hv[:, 0] = tick
        hv[:, 1] = base
        hv[:, 2] = np.clip(self.hp, 0, 4095)
        hdr_rows = hdr.tobytes()
        if self.cfg["snap"] == "shared":
            uniq, inv = np.unique(base, return_inverse=True)
            bases = np.stack([base_state(int(b)) for b in uniq])
            enc = encode_many(cur, bases)
            self.payloads = [hdr_rows[32 * c:32 * c + 32] + enc[inv[c]] for c in range(N)]
            self.distinct_baselines.append(len(uniq))
        else:
            bases = np.stack([base_state(int(b)) for b in base])
            enc = []
            for lo in range(0, N, 50):
                enc.extend(encode_many(cur, bases[lo:lo + 50]))
            self.payloads = [hdr_rows[32 * c:32 * c + 32] + enc[c] for c in range(N)]
            self.distinct_baselines.append(N)

    def verify_decode(self, tick):
        bad = 0
        cur = self.qring[tick % HIST]
        for c in range(0, self.N, max(1, self.N // 8)):
            p = self.payloads[c]
            b = int(np.frombuffer(p[4:8], "<i4")[0])
            base = self.qring[b % HIST] if b >= 0 else np.zeros((self.N, 9), np.int64)
            dec = decode_all(p[32:], base, self.N)
            if not np.array_equal(np.array(dec, np.int64), cur):
                bad += 1
        return bad

    # ------------------------------------------------------------ stage: send
    def stage_send(self):
        if not self.net:
            return
        send, addr = self.cli.sendto, self.sink_addr
        for p in self.payloads:
            for off in range(0, len(p), 1200):
                send(p[off:off + 1200], addr)

    def drain_sink(self):
        if not self.net:
            return
        while True:
            try:
                self.sink.recv(2048)
            except (BlockingIOError, OSError):
                break


STAGES = ["input", "move", "history", "hit", "resolve", "snapshot", "send"]


def run(cfg, verify=False):
    sim = Sim(cfg)
    rec = {k: [] for k in STAGES + ["total"]}
    counts = {k: [] for k in ["active", "candidates", "narrow", "exact", "hits", "aoe", "bytes_per_client"]}
    pen = []
    mism = 0
    extra = 0
    missing = 0
    decode_bad = 0
    oracle_ticks = 0
    total = cfg["warmup"] + cfg["ticks"]
    clock = time.perf_counter_ns
    gc.disable()
    try:
        for tick in range(1, total + 1):
            sim.bot_inputs(tick)
            t0 = clock()
            sim.stage_input(tick); t1 = clock()
            sim.stage_move(tick); t2 = clock()
            sim.stage_history(tick); t3 = clock()
            ha, ht = sim.stage_hit(tick); t4 = clock()
            if verify and tick > cfg["warmup"]:
                fast = set(zip(ha.tolist(), ht.tolist()))
                ora = sim.oracle_hits(tick)
                extra += len(fast - ora)
                missing += len(ora - fast)
                oracle_ticks += 1
            t4b = clock()
            sim.stage_resolve(tick, ha, ht); t5 = clock()
            sim.stage_snapshot(tick); t6 = clock()
            sim.stage_send(); t7 = clock()
            sim.drain_sink()
            if verify and tick > cfg["warmup"] and tick % 5 == 0:
                decode_bad += sim.verify_decode(tick)
            if tick > cfg["warmup"]:
                ms = [(t1 - t0), (t2 - t1), (t3 - t2), (t4 - t3), (t5 - t4b), (t6 - t5), (t7 - t6)]
                for k, v in zip(STAGES, ms):
                    rec[k].append(v / 1e6)
                rec["total"].append(sum(ms) / 1e6)
                counts["active"].append(sim.n_active)
                counts["candidates"].append(sim.n_cand)
                counts["narrow"].append(sim.n_narrow)
                counts["exact"].append(sim.n_exact)
                counts["hits"].append(int(ha.size))
                counts["aoe"].append(sim.n_aoe)
                counts["bytes_per_client"].append(sum(len(p) for p in sim.payloads) / sim.N)
                if tick % 30 == 0:
                    pen.append(sim.penetration())
    finally:
        gc.enable()
        sim.close()
    pc = lambda v, q: float(np.percentile(v, q)) if len(v) else 0.0
    out = dict(name=cfg["name"], N=cfg["N"], cfg=cfg,
               tick={q: pc(rec["total"], q) for q in (50, 95, 99, 99.9)}, tick_max=float(max(rec["total"])),
               stage_p50={k: pc(rec[k], 50) for k in STAGES}, stage_p99={k: pc(rec[k], 99) for k in STAGES},
               active_p50=pc(counts["active"], 50), active_max=float(max(counts["active"])),
               candidates_p50=pc(counts["candidates"], 50), candidates_p99=pc(counts["candidates"], 99),
               narrow_p99=pc(counts["narrow"], 99), exact_p99=pc(counts["exact"], 99), hits_p99=pc(counts["hits"], 99), aoe_p99=pc(counts["aoe"], 99),
               bytes_per_client_p50=pc(counts["bytes_per_client"], 50), bytes_per_client_p99=pc(counts["bytes_per_client"], 99),
               distinct_baselines=float(np.mean(sim.distinct_baselines)),
               penetration_mean=float(np.mean([p[0] for p in pen])) if pen else 0.0,
               penetration_max=float(max(p[1] for p in pen)) if pen else 0.0,
               sweep_inflation_m=sim.T["infl"], root_radius_m=sim.T["root_r"], max_reach_m=sim.T["reach"])
    if verify:
        out.update(oracle_ticks=oracle_ticks, extra_hits=extra, missing_hits=missing, decode_bad_clients=decode_bad)
    return out


def base_cfg(**kw):
    cfg = dict(name="", N=200, arena="disc", disc_r=15.0, corr_w=6.0, corr_l=40.0, H=8, SUB=12, burst=False,
               snap="shared", net=True, aoe_frac=0.25, crowd_iters=4, ticks=900, warmup=300, max_lag=9, loss=0.02,
               seed=20260916, prefilter=True)
    cfg.update(kw)
    return cfg


def pin_process():
    """Keep the single-threaded run on one performance core (hybrid CPUs) and raise priority."""
    if platform.system() != "Windows":
        return "not pinned"
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    size = ctypes.c_ulong(0)
    k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
        return "not pinned"
    raw = buf.raw
    cores, off = [], 0
    while off < size.value:
        rel_size = int.from_bytes(raw[off + 4:off + 8], "little")
        eff = raw[off + 9]
        mask = int.from_bytes(raw[off + 32:off + 40], "little")
        cores.append((eff, mask))
        off += rel_size
    best = max(e for e, _ in cores)
    pmasks = [m for e, m in cores if e == best]
    chosen = pmasks[1] if len(pmasks) > 1 else pmasks[0]
    chosen &= -chosen                         # first logical processor of that core
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    h = ctypes.c_void_p(k32.GetCurrentProcess())
    k32.SetProcessAffinityMask(h, ctypes.c_size_t(chosen))
    k32.SetPriorityClass(h, 0x00000080)       # HIGH_PRIORITY_CLASS
    return f"pinned to logical CPU mask 0x{chosen:x} (efficiency class {best}, {len(pmasks)} performance cores)"


def fmt_row(r):
    t, s50, s99 = r["tick"], r["stage_p50"], r["stage_p99"]
    kbps = r["bytes_per_client_p50"] * 8 * 30 / 1000
    return (f"| {r['name']} | {r['N']} | {t[50]:.2f} | {t[99]:.2f} | {t[99.9]:.2f} | {r['tick_max']:.2f} | "
            f"{s99['input']:.2f} | {s99['move']:.2f} | {s99['history']:.2f} | {s50['hit']:.2f} | {s99['hit']:.2f} | "
            f"{s99['resolve']:.2f} | {s99['snapshot']:.2f} | {s99['send']:.2f} | {r['active_p50']:.0f}/{r['active_max']:.0f} | "
            f"{r['candidates_p99']:.0f} | {r['narrow_p99']:.0f} | {r['exact_p99']:.0f} | {r['hits_p99']:.0f} | {r['aoe_p99']:.0f} | "
            f"{r['bytes_per_client_p50']:.0f} | {kbps:.0f} | {r['distinct_baselines']:.1f} | {r['sweep_inflation_m']:.3f} |")


HEADER = ("| scenario | N | tick p50 ms | p99 | p99.9 | max | input p99 | move p99 | history p99 | hit p50 | hit p99 | "
          "resolve p99 | snapshot p99 | send p99 | active attacks p50/max | root-pass pairs p99 | capsule×segment pairs p99 | exact kernel tests p99 | "
          "contacts p99 | AoE applications p99 | bytes/client p50 | kbit/s per client | distinct baselines | sweep inflation m |\n"
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def main():
    suite = sys.argv[1] if len(sys.argv) > 1 else "all"
    here = Path(__file__).resolve().parent
    pin = pin_process()
    lines, results = [], []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit(f"# Dense melee numpy harness — suite `{suite}`\n")
    emit(f"- CPU: {platform.processor()} · Python {platform.python_version()} · numpy {np.__version__} · {pin}")
    emit("- Single thread. Every stage is batched numpy driven by the interpreter: treat all ms as an upper bound.\n")

    def table(title, cfgs, verify=False):
        emit(f"\n## {title}\n")
        emit(HEADER)
        notes = []
        for c in cfgs:
            r = run(c, verify)
            results.append(r)
            emit(fmt_row(r))
            if verify:
                notes.append(f"- {r['name']}: oracle ticks {r['oracle_ticks']} · extra hits {r['extra_hits']} · "
                             f"missing hits {r['missing_hits']} · snapshot decode mismatches {r['decode_bad_clients']}")
            if c.get("report_pen"):
                notes.append(f"- {r['name']}: crowd mean penetration {r['penetration_mean']:.3f} m · max {r['penetration_max']:.3f} m "
                             f"(disc radius {BODY_R} m)")
        if notes:
            emit("")
            for n in notes:
                emit(n)

    if suite in ("verify", "all"):
        table("V — correctness against the exhaustive float64 oracle (every recorded tick)", [
            base_cfg(name="verify Plaza15 burst H16", H=16, burst=True, ticks=240, warmup=150),
            base_cfg(name="verify Gate6 burst H16", arena="corridor", H=16, burst=True, ticks=240, warmup=150),
            base_cfg(name="verify Blob8 avg H8 SUB8", disc_r=8.0, H=8, SUB=8, ticks=240, warmup=150),
        ], verify=True)
    if suite in ("table", "all"):
        cfgs = []
        for arena, label, extra in (("corridor", "Gate6 6x40 m", {}), ("disc", "Plaza15 r=15 m", {"disc_r": 15.0}),
                                    ("disc", "Blob8 r=8 m (~1 player/m2)", {"disc_r": 8.0})):
            for burst in (False, True):
                for H in (8, 16):
                    cfgs.append(base_cfg(name=f"{label} {'burst' if burst else 'avg'} H{H}", arena=arena, burst=burst, H=H,
                                         report_pen=True, **extra))
        table("A — 100v100 (N=200) full pipeline incl. loopback UDP", cfgs)
    if suite in ("ablate", "all"):
        table("B — ablations at the worst case (Blob8, synchronized burst, H=16)", [
            base_cfg(name="per-client delta encode (200 encodes/tick)", disc_r=8.0, burst=True, H=16, snap="per_client", ticks=450),
            base_cfg(name="shared encode per distinct ack baseline", disc_r=8.0, burst=True, H=16),
            base_cfg(name="shared encode, no UDP syscalls", disc_r=8.0, burst=True, H=16, net=False),
            base_cfg(name="no sphere prefilter (exact kernel on every capsule×segment)", disc_r=8.0, burst=True, H=16, prefilter=False),
            base_cfg(name="SUB=3 sweep samples (34 cm tolerance)", disc_r=8.0, burst=True, H=16, SUB=3),
            base_cfg(name="SUB=8 sweep samples (10 cm tolerance)", disc_r=8.0, burst=True, H=16, SUB=8),
            base_cfg(name="SUB=16 sweep samples (4.5 cm tolerance)", disc_r=8.0, burst=True, H=16, SUB=16),
            base_cfg(name="crowd iterations 1", disc_r=8.0, burst=True, H=16, crowd_iters=1, report_pen=True),
            base_cfg(name="crowd iterations 8", disc_r=8.0, burst=True, H=16, crowd_iters=8, report_pen=True),
            base_cfg(name="Gate6 crowd iterations 8", arena="corridor", burst=True, H=16, crowd_iters=8, report_pen=True),
        ])
    if suite in ("scale", "all"):
        cfgs = []
        for N in (200, 400, 800, 1600):
            cfgs.append(base_cfg(name="fixed density ~1/m2", N=N, disc_r=math.sqrt(N / math.pi), burst=True, H=16,
                                 ticks=600 if N < 1600 else 300, warmup=300 if N < 800 else 450))
        for N in (400, 800):
            cfgs.append(base_cfg(name="fixed area r=15 m", N=N, disc_r=15.0, burst=True, H=16, ticks=600, warmup=450))
        table("C — scaling N (synchronized burst, H=16, SUB=12)", cfgs)

    (here / f"{suite}_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (here / f"{suite}_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
