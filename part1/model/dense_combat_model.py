"""Reproducible workload model, NOT an engine or network performance benchmark.

Run: python part1/model/dense_combat_model.py
Writes deterministic results beside this file. Only standard-library Python.
Geometry is 2D static circles versus straight attack segments; it does not
implement the proposed 3D temporal combat solver.
"""
import json
import math
import random
from pathlib import Path


def distance2(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = dx * dx + dy * dy
    t = max(0, min(1, ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / d)) if d else 0
    return (p[0]-a[0]-t*dx)**2 + (p[1]-a[1]-t*dy)**2


def examine(points, seed, length=3.0, radius=0.45, cell=2.0):
    rng = random.Random(seed)
    grid = {}
    for i, (x, y) in enumerate(points):
        grid.setdefault((math.floor(x/cell), math.floor(y/cell)), []).append(i)
    counts, hits, mismatches = [], 0, 0
    for i, a in enumerate(points):
        theta = rng.random() * math.tau
        b = (a[0]+length*math.cos(theta), a[1]+length*math.sin(theta))
        lo = (math.floor((min(a[0], b[0])-radius)/cell), math.floor((min(a[1], b[1])-radius)/cell))
        hi = (math.floor((max(a[0], b[0])+radius)/cell), math.floor((max(a[1], b[1])+radius)/cell))
        candidates = set()
        for x in range(lo[0], hi[0]+1):
            for y in range(lo[1], hi[1]+1):
                candidates.update(grid.get((x,y), []))
        candidates.discard(i)
        oracle = {j for j, p in enumerate(points) if j != i and distance2(p,a,b) <= radius**2}
        accelerated = {j for j in candidates if distance2(points[j],a,b) <= radius**2}
        mismatches += len(oracle.symmetric_difference(accelerated))
        hits += len(oracle)
        counts.append(len(candidates))
    return dict(players=len(points), root_tests=len(points)*(len(points)-1),
                grid_candidates=sum(counts), candidate_mean=sum(counts)/len(points),
                candidate_max=max(counts), actual_hits=hits, mismatches=mismatches)


def main():
    rng = random.Random(20260916)
    cases = {
        'uniform_100m': [(rng.uniform(-50,50),rng.uniform(-50,50)) for _ in range(200)],
        'packed_0.9m': [((i%20-9.5)*0.9,(i//20-4.5)*0.9) for i in range(200)],
        'all_overlap_adversarial': [(0.0,0.0)]*200,
    }
    geometry = {name: examine(points, 11) for name, points in cases.items()}
    # More seeds, arena sizes, weapon lengths and cell sizes check cell-boundary errors.
    randomized = []
    for seed in range(20):
        rr = random.Random(seed)
        pts = [(rr.uniform(-8,8),rr.uniform(-8,8)) for _ in range(200)]
        randomized.append(examine(pts, seed, length=0.5+seed*0.4, cell=0.5+(seed%5)))
    assert all(x['mismatches'] == 0 for x in [*geometry.values(), *randomized])
    # Tip moving on a semicircle: endpoint chord misses a target on the true arc.
    a,b,target = (2.0,0.0),(-2.0,0.0),(0.0,2.0)
    path = [(2*math.cos(math.pi*i/16),2*math.sin(math.pi*i/16)) for i in range(17)]
    curved = dict(chord_hit=distance2(target,a,b)<=0.1**2,
                  subdivided_hit=any(distance2(target,path[i],path[i+1])<=0.1**2 for i in range(16)),
                  chord_error_m=2.0, segment_sagitta_m=2*(1-math.cos(math.pi/32)))
    assert not curved['chord_hit'] and curved['subdivided_hit']
    # Rewind broad phase counterexample: current-time candidates cannot answer past hits.
    past,current = (1.0,0.0),(10.0,0.0)
    rewind = dict(past_hit=distance2(past,(0,0),(2,0))<=0.45**2,
                  current_hit=distance2(current,(0,0),(2,0))<=0.45**2)
    assert rewind['past_hit'] and not rewind['current_hit']
    workload = []
    for k in (24,64,199):
        for h in (8,16,32):
            for sub in (1,4):
                workload.append(dict(K=k,H=h,subsegments=sub,burst_tests=200*k*h*3*sub,
                                     mean_tests=(200*2*4/30)*k*h*3*sub))
    # Full all-to-all, excluding self. Model assumes every state changes.
    bandwidth = []
    for state_bytes in (24,48):
        bps=199*30*state_bytes*8
        event_bps=199*2*24*3*8  # 2 actions/s, 24 B, 3 copies (hypothesis)
        bandwidth.append(dict(state_bytes=state_bytes, state_mbps=bps/1e6,
                              total_mbps_with_events_and_35pct_overhead=(bps+event_bps)*1.35/1e6,
                              server_mbps=(bps+event_bps)*1.35*200/1e6))
    result=dict(description=__doc__, geometry=geometry,
                randomized_attack_queries=20*200, randomized_mismatches=sum(x['mismatches'] for x in randomized),
                rotational_counterexample=curved, rewind_counterexample=rewind,
                primitive_workload=workload, bandwidth=bandwidth,
                aoe_all_pairs=200*199, aoe_pair_events_broadcast_to_200=200*199*200,
                history_mib_32byte_10samples={str(h):200*h*32*10/1048576 for h in (8,16,32)})
    dest=Path(__file__).with_name('dense_combat_model_results.json')
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('description','primitive_workload')},indent=2))


if __name__ == '__main__':
    main()
