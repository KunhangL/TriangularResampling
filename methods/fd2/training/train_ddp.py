"""Same FD2/TR objective, global-batch DDP and rank-aware full-state resume."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf

from methods.fd2.training.data import FrozenMotionDataset, collate_fn
from methods.fd2.training.train import FD2TrainingModule, SeededCropDataset, fingerprint, rng_state, restore_rng, load_initial_weights, request_segment_stop
from methods.fd2.training.native.train_df import CustomLightningModule
from methods.fd2.training.io_utils import atomic_json, sha

class GlobalBatchSampler(Sampler):
    """Shard the same global epoch permutation; no sample padding in training."""
    def __init__(self,length,batch,seed,rank,world,start=0):
        self.length,self.batch,self.seed,self.rank,self.world,self.start=length,batch,seed,rank,world,start
        assert 0<=rank<world and batch>=world
    def __len__(self):return math.ceil(self.length/self.batch)
    def __iter__(self):
        epoch,offset=divmod(self.start,len(self))
        perm=torch.randperm(self.length,generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()
        for k in range(offset,len(self)):
            whole=perm[k*self.batch:(k+1)*self.batch]
            local=whole[self.rank::self.world]
            # A rank-empty tail still performs a zero-weight forward/backward.
            n=len(local)
            if not local:local=[whole[0]]
            yield [(epoch,index,len(whole),n) for index in local]
        self.start=(epoch+1)*len(self)

class TrainDataset(Dataset):
    def __init__(self,base,seed):self.dataset=SeededCropDataset(base,seed)
    def __len__(self):return len(self.dataset)
    def __getitem__(self,key):
        epoch,index,total,local=key
        return self.dataset[(epoch,index)],total,local

def train_collate(items):
    batch=collate_fn([x[0] for x in items])
    batch['_global_count']=items[0][1];batch['_local_count']=items[0][2]
    return batch

class EvalDataset(Dataset):
    def __init__(self,base,rank,world):
        self.base=base;self.indices=list(range(rank,len(base),world))
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):return self.base[self.indices[i]]

def state_fingerprint(cfg,world):
    # 'wisteria-ddp-v1' / checkpoint['wisteria_ddp'] are format tags of the released
    # checkpoints; keep them unchanged so existing checkpoints stay resumable.
    return hashlib.sha256((fingerprint(cfg)+':wisteria-ddp-v1:'+str(world)).encode()).hexdigest()

def save_checkpoint(trainer,path):
    """All ranks contribute state; only rank zero atomically publishes bytes."""
    module=trainer.lightning_module
    local=dict(rng=rng_state(),tr_rng=module.model.tr_rng.get_state(),rank=trainer.global_rank)
    states=[None]*trainer.world_size
    dist.all_gather_object(states,local)
    module.rank_states=states
    destination=Path(path);tmp=destination.with_name(destination.name+'.partial')
    trainer.save_checkpoint(str(tmp))
    trainer.strategy.barrier()
    if trainer.is_global_zero:os.replace(tmp,destination)
    elif tmp.exists():tmp.unlink()
    trainer.strategy.barrier()

class DDPModule(FD2TrainingModule):
    def __init__(self,cfg,world):
        self.world=world;self.rank_states=None;self.restored_rank_state=None;self.step_seconds=[]
        super().__init__(cfg)
    def write_progress(self,phase,**extra):
        if self.trainer.is_global_zero:
            super().write_progress(phase,world_size=self.world,**extra)
    def training_step(self,batch,batch_idx):
        out=self._step(batch,is_training=True)
        # DDP averages gradients across ranks. Weight unequal final shards so
        # this is exactly the global per-sample mean, not a mean of rank means.
        scale=self.world*batch['_local_count']/batch['_global_count']
        self.log('train_loss/total',out['total'].detach(),on_step=True,on_epoch=False,sync_dist=True)
        return out['total']*scale
    def on_train_batch_start(self,batch,batch_idx):
        torch.cuda.synchronize();self.step_started=time.monotonic()
        super().on_train_batch_start(batch,batch_idx)
    def on_train_batch_end(self,outputs,batch,batch_idx):
        # Native EMA update and diagnostics; do not invoke the single-rank writer.
        CustomLightningModule.on_train_batch_end(self,outputs,batch,batch_idx)
        torch.cuda.synchronize()
        duration=torch.tensor(time.monotonic()-self.step_started,device=self.device)
        dist.all_reduce(duration,op=dist.ReduceOp.MAX)
        self.step_seconds.append(float(duration))
        if self.global_step%self.cfg.validation.save_every_n_steps==0:
            save_checkpoint(self.trainer,Path(self.cfg.save_dir)/f'step_step={self.global_step}.ckpt')
        self.write_progress('training',step_seconds=float(duration),loss=float(outputs['loss']))
    def on_save_checkpoint(self,checkpoint):
        super().on_save_checkpoint(checkpoint)
        assert self.rank_states is not None and len(self.rank_states)==self.world
        checkpoint['wisteria_ddp']=dict(version=1,world_size=self.world,
            fingerprint=state_fingerprint(self.cfg,self.world),rank_states=self.rank_states,
            next_step=int(self.global_step),global_batch=int(self.cfg.data.train_bs))
    def on_load_checkpoint(self,checkpoint):
        value=checkpoint['wisteria_ddp']
        assert value['version']==1 and value['world_size']==self.world
        assert value['fingerprint']==state_fingerprint(self.cfg,self.world)
        assert value['next_step']==checkpoint['global_step']
        super().on_load_checkpoint(checkpoint)
        rank=int(os.environ['RANK'])
        self.restored_rank_state=value['rank_states'][rank]
        self.pending_rng=self.restored_rank_state['rng']
    def on_train_start(self):
        super().on_train_start()
        if self.restored_rank_state:
            self.model.tr_rng.set_state(self.restored_rank_state['tr_rng'].cpu())
        else:
            seed=int(self.cfg.seed)+100003*self.global_rank
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
            self.model.tr_rng.manual_seed(seed+700001)
    def on_validation_end(self):
        if self.trainer.is_global_zero:super().on_validation_end()
        self.trainer.strategy.barrier()

class SynchronizedStop(Callback):
    def __init__(self,stop_step=None,max_seconds=None):
        self.stop_step,self.max_seconds=stop_step,max_seconds
    def on_train_start(self,trainer,module):self.started=time.monotonic()
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        flag=bool((self.stop_step and trainer.global_step>=self.stop_step) or
                  (self.max_seconds and time.monotonic()-self.started>=self.max_seconds))
        value=torch.tensor(int(flag),device=module.device);dist.all_reduce(value,op=dist.ReduceOp.MAX)
        if value.item():request_segment_stop(trainer)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume');p.add_argument('--stop-at-step',type=int);p.add_argument('--max-seconds',type=int)
    p.add_argument('--smoke-batch',type=int);p.add_argument('--no-validation',action='store_true')
    p.add_argument('--capacity-full-length',action='store_true')
    args=p.parse_args();rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    assert world in (2,4,8)
    torch.cuda.set_device(local)
    cfg=OmegaConf.load(args.config);cfg.save_dir=str(Path(args.output).absolute())
    Path(cfg.save_dir).mkdir(parents=True,exist_ok=True)
    if args.smoke_batch:cfg.data.train_bs=args.smoke_batch;cfg.data.num_workers=0
    if args.no_validation:cfg.metrics=OmegaConf.create({})
    assert cfg.trainer.max_steps==cfg.lr_scheduler.params.T_max==30000
    assert int(cfg.trainer.get('accumulate_grad_batches',1))==1
    seed_everything(cfg.seed,workers=True);torch.set_float32_matmul_precision('high')
    torch.cuda.set_per_process_memory_fraction(float(cfg.resources.memory_fraction),local)
    start=0
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False,mmap=True)
        assert ck['wisteria_ddp']['fingerprint']==state_fingerprint(cfg,world)
        start=int(ck['global_step']);del ck
    dataset=FrozenMotionDataset(cfg,'train')
    if args.capacity_full_length:
        assert args.stop_at_step and args.stop_at_step<=10,'Capacity probe only'
        dataset.records=[r for r in dataset.records if r['feature_length']>=cfg.data.window_length-1]
        assert len(dataset)>=cfg.data.train_bs
    sampler=GlobalBatchSampler(len(dataset),cfg.data.train_bs,cfg.seed,rank,world,start)
    workers=int(cfg.data.num_workers)
    loader=DataLoader(TrainDataset(dataset,cfg.seed),batch_sampler=sampler,collate_fn=train_collate,num_workers=workers,
        generator=torch.Generator().manual_seed(cfg.seed+90000+rank),**({'persistent_workers':True,'prefetch_factor':2} if workers else {}))
    val=[]
    if not args.no_validation:
        for split,global_bs in (('val',cfg.data.val_bs),('test',cfg.data.test_bs)):
            data=EvalDataset(FrozenMotionDataset(cfg,split),rank,world)
            assert len(data)>0,'Evaluation set smaller than world size'
            val.append(DataLoader(data,batch_size=max(1,global_bs//world),collate_fn=collate_fn,num_workers=workers,
                generator=torch.Generator().manual_seed(cfg.seed+91000+rank)))
    module=DDPModule(cfg,world)
    if not args.resume:load_initial_weights(module,cfg.init_checkpoint)
    local_world=int(os.environ.get('LOCAL_WORLD_SIZE',world))
    options=OmegaConf.to_container(cfg.trainer);options.update(devices=local_world,num_nodes=world//local_world)
    trainer=Trainer(**options,strategy=DDPStrategy(find_unused_parameters=True,broadcast_buffers=False),
        use_distributed_sampler=False,logger=False,enable_checkpointing=False,enable_progress_bar=False,
        num_sanity_val_steps=0,default_root_dir=cfg.save_dir,
        val_check_interval=cfg.validation.validation_steps,check_val_every_n_epoch=None,
        limit_val_batches=0 if args.no_validation else 1.,
        callbacks=[SynchronizedStop(args.stop_at_step,args.max_seconds)])
    if rank==0:OmegaConf.save(cfg,Path(cfg.save_dir)/'config.yaml',resolve=True)
    trainer.fit(module,loader,val or None,ckpt_path=args.resume,weights_only=False)
    save_checkpoint(trainer,Path(cfg.save_dir)/'last.ckpt')
    memory=dict(rank=rank,allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                reserved_gib=torch.cuda.max_memory_reserved()/2**30,gpu=torch.cuda.get_device_name())
    memories=[None]*world;dist.all_gather_object(memories,memory)
    if rank==0:
        atomic_json(Path(cfg.save_dir)/'SEGMENT.COMPLETE.json',dict(passed=True,global_step=trainer.global_step,
            start_step=start,world_size=world,global_batch=cfg.data.train_bs,rank_memory=memories,
            step_seconds=module.step_seconds,checkpoint=str(Path(cfg.save_dir)/'last.ckpt'),
            sha256=sha(Path(cfg.save_dir)/'last.ckpt'),smoke_only=bool(args.smoke_batch or args.capacity_full_length),time=time.time()))
    trainer.strategy.barrier()

if __name__=='__main__':main()
