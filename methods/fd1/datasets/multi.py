import bisect
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from common.utils.initialize import instantiate

class MultiDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.datasets = []
        self.cfg = cfg
        self.split = split

        if not hasattr(cfg.data, "datasets"):
             rank_zero_info("MultiDataset: cfg.data.datasets not found. No datasets initialized.")
             return

        rank_zero_info(f"Initializing MultiDataset for split {split} with {len(cfg.data.datasets)} sub-datasets...")

        for ds_conf in cfg.data.datasets:
            # ds_conf contains the specific configuration for this dataset (e.g., target, paths)
            # We create a new config object for this dataset where cfg.data is REPLACED by ds_conf.
            # This ensures that the dataset only sees its own parameters and not the global data config,
            # avoiding conflicts or inheriting unwanted global settings.
            ds_cfg = cfg.copy()
            
            # Replace the entire data section with the specific dataset config
            # This assumes ds_conf is a complete dataset configuration (including target, paths, etc.)
            ds_cfg.data = ds_conf
            
            target = ds_cfg.data.get("target", None)
            
            if target is None:
                raise ValueError(f"No target specified for dataset in MultiDataset: {ds_conf}")
                
            rank_zero_info(f"  - Initializing sub-dataset: {target}")
            
            # Instantiate the dataset with the isolated config
            dataset = instantiate(target, cfg=ds_cfg, split=split)
            self.datasets.append(dataset)
            
        # Use ConcatDataset to handle indexing and length automatically
        self.concat_dataset = ConcatDataset(self.datasets)
        rank_zero_info(f"MultiDataset loaded. Total samples: {len(self.concat_dataset)}")

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    output = {}
    keys = batch[0].keys()

    # Sequence keys: pad to max T in batch. Length keys: stack to tensor.
    # Prefix-based dispatch covers extra features emitted by HumanML3DDataset
    # (e.g. `feature_track` / `feature_track_length` for VAE 4-panel test
    # rendering).
    # `feature_text_end` / `token_text_end` are per-sample int lists (variable
    # length, consumed as list-of-lists by diffusion_forcing_wan); not features.
    _passthrough_feature_meta = {"feature_text_end", "token_text_end"}

    def _is_seq_key(k):
        if k in _passthrough_feature_meta:
            return False
        return k == "feature" or k == "token" or (
            k.startswith("feature_") and not k.endswith("_length")
        )

    def _is_len_key(k):
        return k in ("feature_length", "token_length") or (
            k.startswith("feature_") and k.endswith("_length")
        )

    for key in keys:
        if _is_seq_key(key):
            items = [
                torch.from_numpy(b[key]) if isinstance(b[key], np.ndarray) else b[key]
                for b in batch
            ]
            output[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=0
            )
        elif _is_len_key(key):
            output[key] = torch.tensor([b[key] for b in batch])
        else:
            output[key] = [b[key] for b in batch]
    return output
