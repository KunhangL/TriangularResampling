import os
import random
from typing import Dict, List

import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

from common.utils.initialize import instantiate


class LengthMismatchError(Exception):
    pass


class BabelDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        if self.split == "train":
            self.file_list = cfg.data.train_meta_paths
            self.min_length = cfg.data.min_length
            self.max_length = cfg.data.max_length
            self.window_length = cfg.data.get("window_length", cfg.data.max_length)
        elif self.split == "val":
            self.file_list = cfg.data.val_meta_paths
            self.min_length = cfg.data.min_length
            self.max_length = cfg.data.max_length
            self.window_length = cfg.data.get("window_length", cfg.data.max_length)
        elif self.split == "test":
            self.file_list = cfg.data.test_meta_paths
            self.min_length = 0
            self.max_length = cfg.data.max_length
            self.window_length = cfg.data.get("window_length", cfg.data.max_length)
        split_feature_key = f"{split}_feature_path"
        self.feature_path = cfg.data.get(
            split_feature_key, cfg.data.get("feature_path", None)
        )
        self.token_path = cfg.data.get("token_path", None)
        self.text_path = cfg.data.get("text_path", None)
        self.dataset = self._load_file_list()

        rank_zero_info(f"Loaded {len(self.dataset)} samples from {split} dataset.")

    def _load_file_list(self) -> List[str]:
        """Build a metadata list (paths + lengths) from each text file.

        Lazy-load semantics: features/tokens/text are NOT read into RAM here.
        We only persist file paths plus a peek of feature shape[0] for length
        filtering. Actual `np.load()` / text parsing happens in `_process()`
        per sample. Keeps the DataLoader-worker fork footprint tiny so cgroup
        memory accounting stays bounded.
        """
        dataset = []
        ignored_cnt = 0
        for path in self.file_list:
            if os.path.exists(path):
                data_path = os.path.dirname(path)
                dataset_name = os.path.basename(data_path)
                rank_zero_info(f"Loading {path} ...")
                with open(path, "r") as f:
                    for name in f:
                        name = name.strip()
                        if name:
                            data = {}
                            try:
                                data["name"] = name
                                data["dataset"] = dataset_name
                                ##############################
                                # feature (path + peek length)
                                ##############################
                                if self.feature_path is not None:
                                    feature_path = os.path.join(
                                        data_path, self.feature_path, name + ".npy"
                                    )
                                    feature_length = self._peek_npy_length(feature_path)
                                    if (feature_length < self.min_length
                                            or feature_length > self.max_length):
                                        raise LengthMismatchError(
                                            "Feature length out of bounds, skip it."
                                        )
                                    data["feature_path"] = feature_path
                                    data["feature_length"] = feature_length
                                ##############################
                                # token (path only)
                                ##############################
                                if self.token_path is not None:
                                    token_path = os.path.join(
                                        data_path,
                                        self.token_path,
                                        name + ".npy",
                                    )
                                    if not os.path.exists(token_path):
                                        raise FileNotFoundError(
                                            f"token missing: {token_path}"
                                        )
                                    data["token_path"] = token_path
                                ##############################
                                # text (path only)
                                ##############################
                                if self.text_path is not None:
                                    text_path = os.path.join(
                                        data_path, self.text_path, name + ".txt"
                                    )
                                    if not os.path.exists(text_path):
                                        raise FileNotFoundError(
                                            f"text missing: {text_path}"
                                        )
                                    data["text_path"] = text_path
                                ##############################
                                # end
                                ##############################
                                dataset.append(data)
                            except LengthMismatchError:
                                ignored_cnt += 1
                                pass
                            except Exception as e:
                                rank_zero_info(f"Error loading data for {name}: {e}")
                                pass
                            if self.cfg.debug and len(dataset) >= 100:
                                rank_zero_info(f"debug mode, break at {len(dataset)}")
                                break
        if ignored_cnt > 0:
            rank_zero_info(f"Ignored {ignored_cnt} samples due to length mismatch.")
        if len(dataset) == 0:
            rank_zero_info(
                f"No data found in {self.file_list}. Please check the file paths and ensure they are correct."
            )
        else:
            for i in range(3):
                tmp = random.choice(dataset)
                rank_zero_info(
                    f"Random data {tmp['name']}: length={tmp.get('feature_length', 'N/A')}"
                )
        return dataset

    @staticmethod
    def _peek_npy_length(path: str) -> int:
        """Read a .npy file's shape[0] without loading data into RAM."""
        arr = np.load(path, mmap_mode="r")
        return int(arr.shape[0])

    def load_feature(self, feature_path: str) -> np.ndarray:
        feature = np.load(feature_path).astype(np.float32)
        if np.isnan(feature).any():
            raise ValueError("NaN values found in feature, skip it.")
        if feature.shape[0] < self.min_length or feature.shape[0] > self.max_length:
            raise LengthMismatchError("Feature length out of bounds, skip it.")
        return feature

    def load_token(self, token_path: str) -> np.ndarray:
        token = np.load(token_path)
        if np.isnan(token).any():
            raise ValueError("NaN values found in token, skip it.")
        return token

    def load_text(self, text_path: str) -> List[Dict]:
        text_data = []
        with open(text_path, "r") as text_f:
            lines = text_f.readlines()
            for line in lines:
                text_dict = {}
                line_split = line.strip().split("#")
                caption = line_split[0]
                t_tokens = line_split[1].split(" ")
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                text_dict["caption"] = caption
                text_dict["tokens"] = t_tokens
                text_dict["f_tag"] = f_tag
                text_dict["to_tag"] = to_tag

                text_data.append(text_dict)
        return text_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        try:
            return self._process(data)
        except Exception as e:
            # Lazy-load may fail mid-batch (NaN, missing file, IO error). Skip
            # the sample — collate_fn filters None and shrinks the batch by 1
            # rather than crashing the whole run.
            rank_zero_info(
                f"Error processing {data.get('name', '?')}: {e}; skipping."
            )
            return None

    def _process(self, data):
        output = {}
        output["dataset"] = data["dataset"]
        output["name"] = data["name"]

        feature_fps = self.cfg.data.feature_fps
        token_fps = self.cfg.data.token_fps
        downsample_factor = 4

        # Determine crop window in token space
        token_start = 0
        max_token_length = self.window_length // downsample_factor
        force_token_start_by_name = dict(
            self.cfg.data.get("force_token_start_by_name", {}) or {}
        )

        if "token_path" in data:
            token = self.load_token(data["token_path"])
            token_length = token.shape[0]
            forced_start = force_token_start_by_name.get(str(data["name"]), None)
            if token_length > max_token_length:
                if forced_start is not None:
                    token_start = int(forced_start)
                elif self.split == "train":
                    token_start = random.randint(0, token_length - max_token_length)
                else:
                    token_start = 0
                token_start = max(
                    0, min(int(token_start), token_length - max_token_length)
                )
                token = token[token_start:token_start + max_token_length]
                token_length = max_token_length
            elif forced_start is not None:
                token_start = 0
            output["token"] = token
            output["token_length"] = token_length

        # Crop feature using the same window
        if "feature_path" in data:
            feature = self.load_feature(data["feature_path"])
            feature_start = token_start * downsample_factor
            feature_end = min(feature_start + self.window_length, feature.shape[0])
            feature = feature[feature_start:feature_end]
            output["feature"] = feature
            output["feature_length"] = feature.shape[0]

        # Crop and adjust text annotations
        if "text_path" in data:
            text_data = self.load_text(data["text_path"])
            crop_start_sec = token_start / token_fps
            crop_end_sec = (token_start + output.get("token_length", 0)) / token_fps

            output["text"] = []
            output["text_tokens"] = []
            output["feature_text_end"] = []
            output["token_text_end"] = []

            for text_dict in text_data:
                text, text_tokens, f_tag, to_tag = self.process_text_dict(text_dict)
                # Skip segments entirely outside the crop window
                if to_tag <= crop_start_sec or f_tag >= crop_end_sec:
                    continue
                # Adjust to_tag relative to crop start, clamp to crop window
                adjusted_to = min(to_tag, crop_end_sec) - crop_start_sec
                token_text_end = int(adjusted_to * token_fps + 0.5)
                # Skip segments that round to 0 tokens (barely overlapping)
                if token_text_end <= 0:
                    continue
                output["text"].append(text)
                output["text_tokens"].append(text_tokens)
                output["feature_text_end"].append(int(adjusted_to * feature_fps + 0.5))
                output["token_text_end"].append(token_text_end)

            # Fallback: no text overlaps with crop window
            if len(output["text"]) == 0:
                output["text"] = [""]
                output["text_tokens"] = [""]
                output["feature_text_end"] = [output.get("feature_length", 0)]
                output["token_text_end"] = [output.get("token_length", 0)]
            else:
                output["text_tokens"] = output["text_tokens"][-1]

        # Crop offset in token space. PHC/debug mlib lookup needs this to map the
        # sample-local clip back to the original full-motion timeline.
        output["token_start"] = token_start
        return output

    def process_text_dict(self, text_dict: Dict):
        text = text_dict["caption"]
        text_tokens = text_dict["tokens"]
        f_tag = text_dict["f_tag"]
        to_tag = text_dict["to_tag"]
        return text, text_tokens, f_tag, to_tag


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    output = {}
    keys = batch[0].keys()

    for key in keys:
        if key in ["feature", "token"]:
            # Pad sequences
            items = [
                torch.from_numpy(b[key]) if isinstance(b[key], np.ndarray) else b[key]
                for b in batch
            ]
            output[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=0
            )
        elif key in ["feature_length", "token_length"]:
            # Stack scalars
            output[key] = torch.tensor([b[key] for b in batch])
        else:
            # Default to list
            output[key] = [b[key] for b in batch]
    return output


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/default.yaml")
    dataset = BabelDataset(cfg, split="val")
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.data.train_bs, shuffle=True, collate_fn=collate_fn
    )
    for idx, data in tqdm(enumerate(dataloader)):
        pass
