"""Run server and 200 UDP endpoints concurrently; preserve both outputs, no fake WAN."""
import datetime, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
from build_identity import check_build
check_build()
root=Path(__file__).resolve().parent
name=sys.argv[1]; ticks=int(sys.argv[2]) if len(sys.argv)>2 else 900
if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
    raise SystemExit('invalid name')
path=root/'results'/(name+'.json')
if path.exists(): raise SystemExit('choose a fresh name')
exe=root/'build'/'combat_lab.exe'
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [exe,root/'combat_lab.cpp',root/'network_lab.h']}
start=time.perf_counter()
server=subprocess.Popen([str(exe),'server',str(ticks),'39091'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,cwd=root)
client=subprocess.Popen([str(exe),'clients',str(ticks),'39091'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,cwd=root)
out={}
try:
    for label,p in [('server',server),('clients',client)]:
        stdout,stderr=p.communicate(timeout=ticks/30+50)
        entry=dict(exit_code=p.returncode,stdout=stdout,stderr=stderr)
        try:entry['result']=json.loads(stdout)
        except json.JSONDecodeError:pass
        out[label]=entry
finally:
    for p in [server,client]:
        if p.poll() is None:p.kill();p.wait()
result=dict(name=name,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),ticks=ticks,
    topology='127.0.0.1; separate native server and 200-socket client processes; no WAN impairment',
    platform=platform.platform(),logical_processors=os.cpu_count(),hashes=hashes,
    elapsed_seconds=time.perf_counter()-start,**out)
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
sys.exit(max(e['exit_code'] for e in out.values()))
