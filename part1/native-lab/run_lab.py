"""Run one native experiment, persist stdout/errors + source hash + machine metadata.
Usage: python run_lab.py NAME verify|bench [args...]
No result is labeled as an engine or production network benchmark.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from build_identity import check_build

ROOT = Path(__file__).resolve().parent
check_build()
name, *args = sys.argv[1:]
if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
    raise SystemExit('Experiment name must contain only letters, digits, _ or -')
out = ROOT / 'results'
out.mkdir(exist_ok=True)
command = [str(ROOT / 'build' / 'combat_lab.exe'), *args]
path=out / (name+'.json')
if path.exists():
    raise SystemExit(f'Result already exists; choose a fresh experiment name: {path}')
source_hash=hashlib.sha256((ROOT/'combat_lab.cpp').read_bytes()).hexdigest()
header_hash=hashlib.sha256((ROOT/'network_lab.h').read_bytes()).hexdigest()
binary_hash=hashlib.sha256((ROOT/'build'/'combat_lab.exe').read_bytes()).hexdigest()
started = time.perf_counter()
p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
result = dict(name=name, command=args, timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              platform=platform.platform(), logical_processors=os.cpu_count(),
              source_sha256=source_hash, network_header_sha256=header_hash,
              binary_sha256=binary_hash,
              compiler='Zig 0.16.0 c++ / LLVM, -O3 -fno-fast-math -mcpu=x86_64_v3',
              elapsed_seconds=time.perf_counter()-started, exit_code=p.returncode,
              stdout=p.stdout, stderr=p.stderr)
try:
    result['result'] = json.loads(p.stdout)
except json.JSONDecodeError:
    pass
path=out / (name+'.json')
if path.exists():
    raise SystemExit(f'Result already exists; choose a fresh experiment name: {path}')
path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
raise SystemExit(p.returncode)
