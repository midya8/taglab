#!/usr/bin/env bash
set -euo pipefail
installed=$1
source=$2
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export PYTHONPATH="$installed"
python - <<'PY'
import importlib,pkgutil,taglab
assert taglab.__version__=='0.4.0'
mods=list(pkgutil.walk_packages(taglab.__path__,taglab.__name__+'.'))
for m in mods:importlib.import_module(m.name)
assert len(mods)==10,[m.name for m in mods]
print('all eleven modules including root imported outside checkout')
PY
python -m taglab count --symbols '[0,1,2]' --label 0 --count 1 > count.json
python -m taglab demo > summary.json
python -m taglab normalization > normalization.json
python -m taglab condition --symbols '[0,1]' --allowed '[[0],[0,1,2]]' > event.json
python -m taglab condition --symbols '[0,1]' --allowed '[[],[0,1,2]]' > impossible.json
python -m taglab sample --symbols '[0,1]' --allowed '[[0,1,2],[0,1,2]]' --seed 17 > sample.json
python -m taglab infer --symbols '[-2, 0, 2]' --family chain > chain.json
python -m taglab infer --symbols '[1]' --family local > local.json
python -m taglab train --output fresh.json > fresh-summary.json
python -m taglab audit > stored-audit.json
python -m taglab audit --input fresh.json > fresh-audit.json
python -m taglab export --output web > export.json
python - <<'PY'
import json
from taglab.experiment import saved,close
close(json.load(open('fresh.json')),saved())
assert json.load(open('fresh-audit.json'))['states']==327
assert json.load(open('stored-audit.json'))['valid'] is True
assert json.load(open('normalization.json'))['global_branch_A']=='1/11'
assert len(json.load(open('chain.json'))['inference']['viterbi'])==3
assert len(json.load(open('local.json'))['inference']['viterbi'])==1
assert len(json.load(open('export.json'))['files'])==4
assert len(json.load(open('sample.json'))['inference']['trace'])==2
assert json.load(open('sample.json'))['inference']['mode']=='joint'
assert json.load(open('count.json'))['inference']['conditional']['viterbi'].count(0)==1
assert json.load(open('event.json'))['inference']['possible'] is True
assert len(json.load(open('event.json'))['inference']['log_event_gradient'])==39
assert json.load(open('impossible.json'))['inference']['conditional'] is None
print('all nine commands, six fresh fits, 327-state audits and independent input paths pass outside checkout')
PY
for p in archive-0.js archive-1.js archive-2.js data.js; do cmp "web/$p" "$source/docs/$p"; done
printf '%s\n' 'all four archive/loader exports byte-exact outside checkout'
