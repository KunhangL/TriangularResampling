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
    p.add_argument('--check-evaluator',action='store_true')
    args=p.parse_args()
    rows=[json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
    model=None
    cfg=OmegaConf.load(args.config)
    checkpoint_sha=sha(args.checkpoint)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    records=[]
    for row in rows:
        path=out/f'{args.method}__{row["prompt_id"]}__seed{args.seed}.npz'
        reusable=False
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                reusable=(saved['motion'].shape==(2400,263) and np.isfinite(saved['motion']).all()
                          and str(saved['checkpoint_sha256'])==checkpoint_sha and str(saved['config_sha256'])==sha(args.config)
                          and str(saved['method'])==args.method and str(saved['prompt_id'])==row['prompt_id']
                          and int(saved['seed'])==args.seed)
        if reusable:
            print('REUSE_VERIFIED_MOTION',path,flush=True)
            record=dict(path=str(path),sha256=sha(path),shape=[2400,263],fps=20)
            if args.check_evaluator:record['evaluator']=check_evaluator(path,args.config)
            records.append(record)
            continue
        if model is None:model,cfg=load_model(args.config,args.checkpoint)
        torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        model.init_generated(200,batch_size=1)
        frames=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            # c-1 trailing conditions flush the last requested frames.
            for frame in range(2400+model.time_scheduler.chunk_size-1):
                caption=caption_at(row,min(frame,2399))
                result=model.stream_generate_step(dict(text=[caption]))['generated'][0]
                if result.numel(): frames.append(result.float().cpu())
                if frame%400==0: print('STREAM',args.method,row['prompt_id'],frame,flush=True)
        motion=torch.cat(frames).numpy()
        assert motion.shape==(2400,263) and np.isfinite(motion).all()
        np.savez_compressed(path,motion=motion,fps=np.int64(20),seed=np.int64(args.seed),method=args.method,prompt_id=row['prompt_id'],
                            caption=row['caption'],text_tokens=np.array(row['text_tokens']),checkpoint_sha256=checkpoint_sha,config_sha256=sha(args.config),
                            representation='humanml3d263',representation_dim=np.int64(263),target_seconds=np.int64(120),checkpoint=str(Path(args.checkpoint).resolve()),
                            text_frames=np.array([round(x['end_sec']*20) for x in row.get('segments',[])],dtype=np.int64))
        record=dict(path=str(path),sha256=sha(path),shape=list(motion.shape),fps=20)
        if args.check_evaluator: record['evaluator']=check_evaluator(path,args.config)
        records.append(record)
    atomic_json(out/'GENERATION.COMPLETE.json',dict(method=args.method,records=records,checkpoint_sha256=checkpoint_sha,manifest_sha256=sha(args.manifest),smoke_only=len(rows)==1))

if __name__=='__main__':main()
