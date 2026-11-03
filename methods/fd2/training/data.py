"""Token-free adapter of the frozen FD1 dataset membership to native FD2 crops."""
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from .native_datasets.babel import BabelDataset
from .native_datasets.humanml3d import HumanML3DDataset, collate_fn


class FrozenMotionDataset(Dataset):
    def __init__(self, cfg, split='train'):
        path = Path(cfg.data.membership_manifest)
        raw = path.read_bytes()
        expected = cfg.data.get('membership_sha256')
        if expected and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Frozen membership hash mismatch')
        membership = json.loads(raw)
        self.records = [x for group in membership['splits'][split] for x in group['samples']]
        if split != 'train':
            # Validation skips the features prepare_data flagged as malformed: two HumanML3D
            # test features with non-finite values (007975, M007975). Training has none.
            flagged = {p.split(' ', 1)[0] for p in membership.get('problems', [])}
            self.records = [x for x in self.records if f'{split}:{x["feature_path"]}' not in flagged]
        self.dimensions = int(cfg.model.params.input_dim)
        self.processors = {}
        for name,cls in [('HumanML3D',HumanML3DDataset),('BABEL',BabelDataset),('SEED',HumanML3DDataset)]:
            obj = cls.__new__(cls)
            fps = int(cfg.data.get('feature_fps',20))
            obj.cfg = OmegaConf.create({'data':dict(feature_fps=fps,token_fps=fps/4)})
            obj.window_length = int(cfg.data.get('window_length',200))
            obj.stream_mode = True
            self.processors[name] = obj

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        obj = self.processors[rec['dataset']]
        feature = np.load(rec['feature_path'],allow_pickle=False).astype(np.float32)
        if feature.shape != (rec['feature_length'],self.dimensions) or not np.isfinite(feature).all():
            raise ValueError('Invalid frozen motion feature: '+rec['feature_path'])
        text_data = obj.load_text(rec['text_path'])
        item = obj._process(dict(name=rec['name'],dataset=rec['dataset'],feature=feature,
                                 feature_length=len(feature),text_data=text_data))
        return {k:v for k,v in item.items() if not k.startswith('token')}
