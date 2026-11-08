import argparse,json,sys
from pathlib import Path
from .experiment import experiment,saved,audit
from .model import predict
from .normalization import comparison
from .export import export
from .conditioning import label_event


def main():
    p=argparse.ArgumentParser(description='TagLab: conditional sequence scores, not a text tagging service');sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('demo');sub.add_parser('normalization')
    q=sub.add_parser('train');q.add_argument('--output',required=True)
    q=sub.add_parser('audit');q.add_argument('--input')
    q=sub.add_parser('infer');q.add_argument('--symbols',required=True);q.add_argument('--family',choices=['chain','local'],default='chain')
    q=sub.add_parser('condition');q.add_argument('--symbols',required=True);q.add_argument('--allowed',required=True);q.add_argument('--family',choices=['chain','local'],default='chain')
    q=sub.add_parser('export');q.add_argument('--output',required=True)
    args=p.parse_args()
    try:
        if args.command=='demo':
            r=saved();out=dict(selected=r['selected'],models=[{k:m[k] for k in ['id','status','validation','test']} for m in r['models']])
        elif args.command=='normalization':out=comparison()
        elif args.command=='train':
            r=experiment();Path(args.output).write_text(json.dumps(r,sort_keys=True,allow_nan=False)+'\n');out=dict(output=args.output,selected=r['selected'])
        elif args.command=='audit':out=audit(json.loads(Path(args.input).read_text()) if args.input else saved())
        elif args.command=='export':out=dict(files=export(args.output))
        else:
            x=json.loads(args.symbols);r=saved();m=next(m for m in r['models'] if m['id']==r['selected'][args.family]);out=dict(model_id=m['id'],symbols=x,inference=label_event(m['theta'],x,json.loads(args.allowed)) if args.command=='condition' else predict(m['theta'],x))
        print(json.dumps(out,sort_keys=True,allow_nan=False))
    except (ValueError,KeyError,TypeError,StopIteration,OSError) as exc:p.error(str(exc))


if __name__=='__main__':main()
