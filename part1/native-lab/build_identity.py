"""Prevent accidentally timing a stale binary after source changes or rollback."""
import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STAMP=ROOT/'build'/'identity.json'

def identities():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ['combat_lab.cpp','network_lab.h','build/combat_lab.exe']}

def check_build():
    if not STAMP.exists() or json.loads(STAMP.read_text())!=identities():
        raise SystemExit('Missing/stale build identity. Run build.cmd before benchmarking; no executable was launched.')

if __name__=='__main__':
    if sys.argv[1:]!=['record']:raise SystemExit('Use record immediately after a successful build')
    STAMP.write_text(json.dumps(identities(),indent=2)+'\n',encoding='utf-8')
