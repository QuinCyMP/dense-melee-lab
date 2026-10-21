"""Discrete-event loss/jitter model, explicitly NOT a WAN or rendering benchmark.
Compares full-frame gating with independent versioned entity-group updates.
Models 200 receivers, 200 actors, all actors present; no distance culling.
"""
import hashlib, heapq, json, math, random
from pathlib import Path

def percentile(values,p):
    values=sorted(values)
    return values[max(0,math.ceil(len(values)*p)-1)] if values else 0

def scenario(loss,oneway,jitter,burst=False):
    rng=random.Random(611930)
    peers,ticks,groups=200,900,8
    full=[-1]*peers
    independent=[[-1]*groups for _ in range(peers)]
    masks={};pending=[];dropped=sent=stale=0;full_ages=[];part_ages=[]
    # Synthetic telegraphs start every 30 ticks and persist for 12 ticks.
    # Independent chunk application does not require unrelated groups to arrive.
    first_cue={};full_first_cue={}
    for tick in range(ticks+10):
        now=tick/30
        if tick<ticks:
            for peer in range(peers):
                for group in range(groups):
                    sent+=1
                    if rng.random()<loss or (burst and 10<=now<10.2):
                        dropped+=1;continue
                    arrival=now+max(0,(oneway+rng.uniform(-jitter,jitter))/1000)
                    heapq.heappush(pending,(arrival,peer,group,tick))
        while pending and pending[0][0]<=now:
            arrival,peer,group,version=heapq.heappop(pending)
            key=(peer,version);masks[key]=masks.get(key,0)|(1<<group)
            if masks[key]==(1<<groups)-1:
                full[peer]=max(full[peer],version)
                if version%30<12:full_first_cue.setdefault((peer,version//30),arrival-version//30)
                del masks[key]
            if version>independent[peer][group]:
                independent[peer][group]=version
                if version%30<12:first_cue.setdefault((peer,group,version//30),arrival-version//30)
            else:stale+=1
        if 30<=tick<ticks:
            full_ages.extend((tick-v)*1000/30 for v in full if v>=0)
            part_ages.extend((tick-v)*1000/30 for row in independent for v in row if v>=0)
        # Expire assembly records; stale full-frame fragments must not leak memory.
        if tick%30==0:masks={k:v for k,v in masks.items() if k[1]>=tick-30}
    deadlines=peers*groups*(ticks//30)
    timely=sum(v<=.4 for v in first_cue.values())
    return dict(loss_rate=loss,one_way_ms=oneway,jitter_plus_minus_ms=jitter,common_200ms_burst=burst,
        sent=sent,dropped=dropped,stale_ignored=stale,
        full_frame_state_age_ms={p:percentile(full_ages,q) for p,q in [('p99',.99),('max',1)]},
        independent_group_state_age_ms={p:percentile(part_ages,q) for p,q in [('p99',.99),('max',1)]},
        independent_cue_deadlines=deadlines,independent_cues_before_400ms=timely,
        independent_late_or_missing_cues=deadlines-timely,
        full_frame_late_or_missing_cues=peers*(ticks//30)-sum(v<=.4 for v in full_first_cue.values()))

if __name__=='__main__':
    results=[scenario(.01,50,20),scenario(.05,125,20),scenario(.01,125,20,True)]
    result=dict(kind='discrete-event model, not actual network impairment',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scenarios=results)
    path=Path(__file__).parent/'results'/'delivery_model.json'
    if path.exists():raise SystemExit('Result exists; preserve it before rerunning')
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
