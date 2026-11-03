"""Native FD2 post-training with isolated TR, deterministic resumable data stream."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .data import FrozenMotionDataset, collate_fn
from .native.train_df import CustomLightningModule
from .io_utils import atomic_json, sha

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])

def fingerprint(cfg):
    value=OmegaConf.to_container(cfg,resolve=True)
    # Output/logging choices do not change training semantics.
    keep={k:value[k] for k in ('model','data','optimizer','lr_scheduler','seed')}
    return hashlib.sha256(json.dumps(keep,sort_keys=True).encode()).hexdigest()

def atomic_checkpoint(trainer, destination):
    """Publish only a fully written Lightning checkpoint on shared storage."""
    destination = Path(destination)
    temporary = destination.with_name(destination.name + '.partial')
    trainer.save_checkpoint(str(temporary))
    os.replace(temporary, destination)

class SeededCropDataset(Dataset):
    def __init__(self,dataset,seed): self.dataset,self.seed=dataset,seed
    def __len__(self): return len(self.dataset)
    def __getitem__(self,key):
        epoch,index=key
        previous=random.getstate()
        random.seed(self.seed+epoch*1000000007+index)
        try: return self.dataset[index]
        finally: random.setstate(previous)

class EpochBatchSampler(Sampler):
    """Uniform epoch permutations, anchored to consumed (not prefetched) steps."""
    def __init__(self,length,batch_size,seed,start_step=0):
        self.length,self.batch_size,self.seed,self.start_step=length,batch_size,seed,start_step
    def __len__(self): return math.ceil(self.length/self.batch_size)
    def __iter__(self):
        batches=len(self)
        epoch,offset=divmod(self.start_step,batches)
        generator=torch.Generator().manual_seed(self.seed+epoch)
        perm=torch.randperm(self.length,generator=generator).tolist()
        for n in range(offset,batches):
            yield [(epoch,i) for i in perm[n*self.batch_size:(n+1)*self.batch_size]]
        self.start_step=(epoch+1)*batches

class FD2TrainingModule(CustomLightningModule):
    def __init__(self,cfg):
        self.pending_rng=None
        self.gradient_checks=0
        self.run_started=time.time()
        self.start_step=0
        super().__init__(cfg)
    def write_progress(self,phase,**extra):
        value=dict(phase=phase,global_step=int(self.global_step),max_steps=int(self.cfg.trainer.max_steps),
                   updated_at=time.time(),pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   start_step=self.start_step,elapsed_seconds=time.time()-self.run_started,
                   learning_rate=self.trainer.optimizers[0].param_groups[0]['lr'] if self.trainer.optimizers else None,
                   gradient_checks=self.gradient_checks,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,**extra)
        atomic_json(Path(self.cfg.save_dir)/'progress.json',value)
    def _step(self,batch,is_training=True):
        if is_training: return self.model(batch)
        # Native evaluation; never replay, even if called outside Lightning eval.
        from methods.fd2.models.upstream.diffusion_forcing_wan import DiffForcingWanModel
        return DiffForcingWanModel.forward(self.model,batch)
    def on_after_backward(self):
        grads=[p.grad for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads):
            raise FloatingPointError('Non-finite or missing training gradients')
        self.gradient_checks+=1
    def update_test(self,batch):
        super().update_test(batch)
        # The frozen data uses the existing raw-data layout, not the release's
        # nested HumanML3D263 folders. Render GT from the exact same test batch.
        for i,(name,dataset) in enumerate(zip(batch['name'],batch['dataset'])):
            folder=Path(self.cfg.save_dir)/dataset/'gt_feature'
            folder.mkdir(parents=True,exist_ok=True)
            length=int(batch['feature_length'][i])
            np.save(folder/(name+'.npy'),batch['feature'][i,:length].float().cpu().numpy())
    def process_test_results(self):
        from .native.visualization.visualize import render_video
        for folder in Path(self.cfg.save_dir).glob('*/gt_feature'):
            dataset=folder.parent.name
            gt_video=folder.parent/'gt_video'
            render_video(str(folder),str(gt_video),self.cfg.test_setting,
                         frames_dir=str(folder.parent/'frames'),representation=self.representation)
            self.cfg.test_setting[dataset]=dict(compare_folders=[str(gt_video)],compare_names=['Ground Truth'])
        super().process_test_results()
    def on_save_checkpoint(self,checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['fd2_tr_resume']=dict(version=1,fingerprint=fingerprint(self.cfg),rng=rng_state(),next_step=int(self.global_step),gradient_checks=self.gradient_checks)
    def on_load_checkpoint(self,checkpoint):
        state=checkpoint['fd2_tr_resume']
        if state['fingerprint'] != fingerprint(self.cfg) or state['next_step'] != checkpoint['global_step']:
            raise ValueError('Resume configuration/cursor mismatch')
        super().on_load_checkpoint(checkpoint)
        self.pending_rng=state['rng']
        self.gradient_checks=state['gradient_checks']
    def on_train_start(self):
        super().on_train_start()
        self.start_step=int(self.global_step)
        self.run_started=time.time()
        if self.pending_rng is not None:
            restore_rng(self.pending_rng)
            self.pending_rng=None
        self.write_progress('training')
    def on_train_batch_end(self,outputs,batch,batch_idx):
        super().on_train_batch_end(outputs,batch,batch_idx)
        if int(self.global_step) % int(self.cfg.validation.save_every_n_steps)==0:
            # Save after EMA update, not the earlier ModelCheckpoint callback.
            destination=Path(self.cfg.save_dir)/f'step_step={self.global_step}.ckpt'
            atomic_checkpoint(self.trainer, destination)
        if int(self.global_step)%10==0 or self.global_step<=2:
            loss=outputs.get('loss') if isinstance(outputs,dict) else outputs
            self.write_progress('training',loss=float(loss.detach()) if loss is not None else None)
    def on_validation_start(self):
        self.write_progress('validation_and_rendering')
    def on_validation_end(self):
        values={str(k):float(v.detach().cpu()) for k,v in self.trainer.callback_metrics.items()
                if isinstance(v,torch.Tensor) and v.numel()==1 and ('val_loss/' in str(k) or 'metrics/' in str(k))}
        atomic_json(Path(self.cfg.save_dir)/'validation'/f'step_{self.global_step}.json',
                    dict(global_step=int(self.global_step),metrics=values,updated_at=time.time()))
        self.write_progress('training',last_validation_step=int(self.global_step),validation_metrics=values)

def request_segment_stop(trainer):
    """Stop cleanly without Lightning inserting an off-cadence validation.

    Lightning treats should_stop like early stopping and otherwise validates
    immediately. A scheduler walltime boundary is not an evaluation milestone.
    This runtime limit is reset by the next Trainer; no config or LR is changed.
    """
    interval=getattr(trainer,'val_check_batch',None)
    if isinstance(interval,int) and interval>0 and trainer.global_step%interval:
        trainer.limit_val_batches=0
    trainer.should_stop=True

class StopAfter(Callback):
    def __init__(self,step): self.step=step
    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        if trainer.global_step>=self.step: request_segment_stop(trainer)

class WallTimeBudget(Callback):
    """Per-allocation clock, intentionally not restored from a prior job allocation."""
    def __init__(self, duration):
        days,hours,minutes,seconds = map(int,duration.split(':'))
        self.seconds = ((days*24+hours)*60+minutes)*60+seconds
        self.started = None
    def on_train_start(self, trainer, pl_module):
        self.started = time.monotonic()
    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        if self.started is not None and time.monotonic()-self.started >= self.seconds:
            request_segment_stop(trainer)

def load_initial_weights(module,path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    mismatch=module.model.load_state_dict(checkpoint['state_dict'],strict=False)
    if set(mismatch.missing_keys)!={'_extra_state'} or mismatch.unexpected_keys:
        raise ValueError(f'Official model mismatch: {mismatch}')
    if 'ema_state' not in checkpoint: raise ValueError('Official EMA is missing')
    module.ema.load_state_dict(checkpoint['ema_state'])
    return int(checkpoint['global_step'])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume')
    parser.add_argument('--smoke-stop',type=int)
    parser.add_argument('--smoke-batch-size',type=int,default=2)
    parser.add_argument('--stop-at-step',type=int,
                        help='End a scheduler segment without changing the 30k LR horizon')
    parser.add_argument('--max-time',help='Lightning duration, DD:HH:MM:SS; cleanly save before the job walltime')
    args=parser.parse_args()
    cfg=OmegaConf.load(args.config)
    if int(cfg.trainer.get('accumulate_grad_batches',1))!=1:
        raise ValueError('Resumable FD2 loader currently requires accumulation=1')
    if cfg.get('resources',{}).get('memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(float(cfg.resources.memory_fraction))
    cfg.save_dir=str(Path(args.output).resolve())
    Path(cfg.save_dir).mkdir(parents=True,exist_ok=True)
    if args.smoke_stop:
        cfg.data.train_bs=args.smoke_batch_size
        cfg.data.num_workers=0
        cfg.metrics=OmegaConf.create({})
    seed_everything(cfg.seed,workers=True)
    torch.set_float32_matmul_precision('high')
    dataset=FrozenMotionDataset(cfg,'train')
    start=0
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False,mmap=True)
        start=int(checkpoint['global_step'])
        if checkpoint['fd2_tr_resume']['fingerprint']!=fingerprint(cfg):
            raise ValueError('Resume fingerprint mismatch before DataLoader construction')
        del checkpoint
    sampler=EpochBatchSampler(len(dataset),cfg.data.train_bs,cfg.seed,start)
    workers=cfg.data.num_workers
    train_loader=DataLoader(SeededCropDataset(dataset,cfg.seed),batch_sampler=sampler,num_workers=workers,collate_fn=collate_fn,
                            generator=torch.Generator().manual_seed(cfg.seed+90000),**({'persistent_workers':True,'prefetch_factor':8} if workers else {}))
    val_loaders=[]
    if not args.smoke_stop:
        for split,bs in [('val',cfg.data.val_bs),('test',cfg.data.test_bs)]:
            val_loaders.append(DataLoader(FrozenMotionDataset(cfg,split),batch_size=bs,num_workers=workers,collate_fn=collate_fn,shuffle=False,
                                           generator=torch.Generator().manual_seed(cfg.seed+91000)))
    module=FD2TrainingModule(cfg)
    initial_step=load_initial_weights(module,cfg.init_checkpoint) if not args.resume else None
    callbacks=[StopAfter(args.smoke_stop)] if args.smoke_stop else []
    if args.stop_at_step:
        callbacks.append(StopAfter(args.stop_at_step))
    if args.max_time:
        callbacks.append(WallTimeBudget(args.max_time))
    trainer=Trainer(**OmegaConf.to_container(cfg.trainer),callbacks=callbacks,logger=False,default_root_dir=cfg.save_dir,
                    enable_checkpointing=False,num_sanity_val_steps=0,enable_progress_bar=False,
                    val_check_interval=cfg.validation.validation_steps,check_val_every_n_epoch=None,limit_val_batches=0 if args.smoke_stop else 1.)
    OmegaConf.save(cfg,Path(cfg.save_dir)/'config.yaml',resolve=True)
    # Only our hash-verified local full-state checkpoints are resumed here;
    # PyTorch's weights-only default cannot restore optimizer/RNG/OmegaConf state.
    trainer.fit(module,train_dataloaders=train_loader,val_dataloaders=val_loaders or None,ckpt_path=args.resume,weights_only=False)
    atomic_checkpoint(trainer, Path(cfg.save_dir)/'last.ckpt')
    marker = 'TRAINING.COMPLETE.json' if args.smoke_stop or trainer.global_step >= cfg.trainer.max_steps else 'SEGMENT.COMPLETE.json'
    atomic_json(Path(cfg.save_dir)/marker,dict(global_step=int(trainer.global_step),initial_step=initial_step,
                gradient_checks=module.gradient_checks,checkpoint=str(Path(cfg.save_dir)/'last.ckpt'),sha256=sha(Path(cfg.save_dir)/'last.ckpt'),
                smoke_only=bool(args.smoke_stop),max_steps=cfg.trainer.max_steps,scheduler_T_max=cfg.lr_scheduler.params.T_max))

if __name__=='__main__':main()
