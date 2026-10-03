#!/usr/bin/env python3
"""Run the repository checks locally and retain logs; never install dependencies silently."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
out=root/'validation'/'local'
out.mkdir(parents=True,exist_ok=True)
commands=[
 [sys.executable,'-m','compileall','-q','ablationlab','tests'],
 [sys.executable,'-m','ablationlab','doctor'],
 [sys.executable,'-m','pytest','-q','-ra','--junitxml='+str(out/'pytest.xml')],
]
results=[]
for index,command in enumerate(commands,1):
    start=time.monotonic()
    result=subprocess.run(command,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/f'{index:02d}.log').write_text(result.stdout,encoding='utf-8')
    print(result.stdout,flush=True)
    results.append({'command':command,'returncode':result.returncode,'seconds':time.monotonic()-start})
(out/'commands.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
sys.exit(0 if all(r['returncode']==0 for r in results) else 1)
