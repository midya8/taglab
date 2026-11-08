import base64,gzip,hashlib,json
from pathlib import Path
from importlib.resources import files
from .experiment import saved,digest


def archive(report,directory):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode();encoded=base64.b64encode(gzip.compress(raw,mtime=0)).decode()
    parts=[]
    for i,start in enumerate(range(0,len(encoded),160000)):
        name=f'report-{i}.b64';root.joinpath(name).write_text(encoded[start:start+160000]+'\n');parts.append(name)
    root.joinpath('report.json').write_text(json.dumps(dict(encoding='gzip+base64',sha256=hashlib.sha256(raw).hexdigest(),parts=parts),sort_keys=True)+'\n')
    return dict(bytes=len(raw),encoded_bytes=len(encoded),parts=parts)


def export(directory):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True);source=files('taglab').joinpath('artifacts');meta=json.loads(source.joinpath('report.json').read_text());saved()
    result=[]
    for i,name in enumerate(meta['parts']):
        filename=f'archive-{i}.js';root.joinpath(filename).write_text(f'globalThis.TAG_PARTS = globalThis.TAG_PARTS || []; TAG_PARTS[{i}] = '+json.dumps(source.joinpath(name).read_text().strip())+';\n');result.append(filename)
    root.joinpath('data.js').write_text('globalThis.TAG_READY=(async()=>{const bytes=Uint8Array.from(atob(TAG_PARTS.join("")),c=>c.charCodeAt(0));const raw=new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());const hash=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",raw)),b=>b.toString(16).padStart(2,"0")).join("");if(hash!=='+json.dumps(meta['sha256'])+')throw Error("Archive hash mismatch");globalThis.TAG_DATA=JSON.parse(new TextDecoder().decode(raw));return TAG_DATA;})();\n');result.append('data.js')
    return result
