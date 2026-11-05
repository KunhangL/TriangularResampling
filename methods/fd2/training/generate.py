"""Native FD2 streaming inference; unified 263D NPZ and real evaluator smoke."""
import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from .native.utils.initialize import instantiate
from .io_utils import atomic_json,sha

def caption_at(row,frame):
    if 'segments' not in row: return row['caption']
    for segment in row['segments']:
        if frame < round(segment['end_sec']*20): return segment['caption']
    return row['segments'][-1]['caption']

def load_model(config,checkpoint):
    cfg=OmegaConf.load(config)
    model=instantiate(cfg.model.target,**cfg.model.params)
    stored=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    incompatible=model.load_state_dict(stored['state_dict'],strict=False)
    if set(incompatible.missing_keys)-{'_extra_state'} or incompatible.unexpected_keys:
        raise ValueError(f'Checkpoint mismatch {incompatible}')
    if 'ema_state' not in stored: raise ValueError('Missing EMA inference weights')
    with torch.no_grad():
        for param,shadow in zip(model.parameters(),stored['ema_state']['shadow_params']):
            param.copy_(shadow)
    # Training intentionally bounds the unique-caption cache. Streaming reuses
    # a small fixed prompt set; use FD2's native cache so UMT5 is not rerun at
    # every frame. Encoder weights/outputs and sampling remain unchanged.
    model.text_module.cache_encoded=True
    return model.eval().cuda(),cfg

def check_evaluator(path,config):
    from common.evaluation import eval_windowed_long_motion as evaluator
    loaded=evaluator.load_generated_npz(str(path))
    # In addition to loader compatibility, run the actual trained motion encoder.
    from .native.metrics.HumanML3D263.t2m import T2MMetrics
    metric=T2MMetrics(OmegaConf.load(config).metrics.t2m.params).cuda().eval()
    with np.load(path,allow_pickle=False) as data:
        window=torch.from_numpy(data['motion'][:200]).cuda()[None]
    with torch.no_grad():
        embedding=metric._get_motion_embeddings(window,[200])
    if embedding.shape!=(1,512) or not torch.isfinite(embedding).all():
        raise ValueError('Real motion evaluator cannot encode generated 263D motion')
    return dict(shape=list(embedding.shape),finite=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--check-evaluator',action='store_true')
    args=p.parse_args()
    all_rows=[json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
    rows=all_rows[args.shard_index::args.num_shards]
    model=None
    cfg=OmegaConf.load(args.config)
    config_sha=sha(args.config)
    checkpoint_sha=sha(args.checkpoint)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    records=[]
    todo=[]
    for row in rows:
        path=out/f'{args.method}__{row["prompt_id"]}__seed{args.seed}.npz'
        reusable=False
        if path.exists():
            try:
                with np.load(path,allow_pickle=False) as saved:
                    reusable=(saved['motion'].shape==(2400,263) and np.isfinite(saved['motion']).all()
                              and str(saved['checkpoint_sha256'])==checkpoint_sha and str(saved['config_sha256'])==config_sha
                              and str(saved['method'])==args.method and str(saved['prompt_id'])==row['prompt_id']
                              and int(saved['seed'])==args.seed)
            except Exception:
                reusable=False
        if reusable:
            print('REUSE_VERIFIED_MOTION',path,flush=True)
            record=dict(path=str(path),sha256=sha(path),shape=[2400,263],fps=20)
            if args.check_evaluator:record['evaluator']=check_evaluator(path,args.config)
            records.append(record)
        else:
            todo.append((row,path))
    if todo:
        model,cfg=load_model(args.config,args.checkpoint)
        device=next(model.parameters()).device
        unique_texts=['']
        for row,_ in todo:
            if 'segments' in row:
                for seg in row['segments']:
                    if seg['caption'] not in unique_texts: unique_texts.append(seg['caption'])
            elif row['caption'] not in unique_texts:
                unique_texts.append(row['caption'])
        missing_texts = [t for t in unique_texts if t not in model.text_module.text_cache]
        for t in unique_texts:
            if t in model.text_module.text_cache:
                model.text_module.text_cache[t] = model.text_module.text_cache[t].to(device)
        if missing_texts:
            free_gib = torch.cuda.mem_get_info(device)[0] / (1024 ** 3)
            t5_dev = device if free_gib >= 11.0 else torch.device('cpu')
            model.text_module.text_encoder.model.to(t5_dev)
            with torch.no_grad():
                for s in range(0,len(missing_texts),4):
                    sub=missing_texts[s:s+4]
                    for t,f in zip(sub,model.text_module.text_encoder(sub,t5_dev)):
                        model.text_module.text_cache[t]=f.to(device)
            model.text_module.text_encoder.model.cpu()
            torch.cuda.empty_cache()

        def _rollback_shared():
            r0=torch.randn_like(model.generated[0][:,model.seq_len:,...])
            for i in range(model.batch_size):
                model.generated[i][:,:model.seq_len,...]=model.generated[i][:,model.seq_len:,...].clone()
                model.generated[i][:,model.seq_len:,...]=r0.clone()
            model.current_step-=model.time_scheduler.get_step_rollback(model.seq_len)
            model.condition_frames-=model.seq_len
            model.current_commit-=model.seq_len
            model.text_module.trim_stream(model.seq_len)
        model._rollback=_rollback_shared

        bs=max(1,int(args.batch_size))
        for start in range(0,len(todo),bs):
            chunk=todo[start:start+bs]
            B=len(chunk)
            torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
            model.init_generated(200,batch_size=1)
            if B>1:
                g0=model.generated[0]
                model.batch_size=B
                model.generated=[g0.clone() for _ in range(B)]
                model.text_module.init_stream(B)
            frames=[[] for _ in range(B)]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                # c-1 trailing conditions flush the last requested frames.
                for frame in range(2400+model.time_scheduler.chunk_size-1):
                    captions=[caption_at(r,min(frame,2399)) for r,_ in chunk]
                    result=model.stream_generate_step(dict(text=captions))['generated']
                    if result[0].numel():
                        for i in range(B):
                            frames[i].append(result[i].float().cpu())
                    if frame%400==0: print('STREAM',args.method,chunk[0][0]['prompt_id'],f'batch={B}',frame,flush=True)
            for i,(row,path) in enumerate(chunk):
                motion=torch.cat(frames[i]).numpy()
                assert motion.shape==(2400,263) and np.isfinite(motion).all()
                tmp=path.with_name(path.name+'.partial.npz')
                np.savez_compressed(tmp,motion=motion,fps=np.int64(20),seed=np.int64(args.seed),method=args.method,prompt_id=row['prompt_id'],
                                    caption=row['caption'],text_tokens=np.array(row['text_tokens']),checkpoint_sha256=checkpoint_sha,config_sha256=config_sha,
                                    representation='humanml3d263',representation_dim=np.int64(263),target_seconds=np.int64(120),checkpoint=str(Path(args.checkpoint).resolve()),
                                    text_frames=np.array([round(x['end_sec']*20) for x in row.get('segments',[])],dtype=np.int64))
                tmp.replace(path)
                record=dict(path=str(path),sha256=sha(path),shape=list(motion.shape),fps=20)
                if args.check_evaluator: record['evaluator']=check_evaluator(path,args.config)
                records.append(record)
    marker=out/(f'GENERATION.SHARD_{args.shard_index}_OF_{args.num_shards}.json' if args.num_shards>1 else 'GENERATION.COMPLETE.json')
    atomic_json(marker,dict(method=args.method,records=records,checkpoint_sha256=checkpoint_sha,manifest_sha256=sha(args.manifest),smoke_only=len(all_rows)==1))

if __name__=='__main__':main()
