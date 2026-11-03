# Third-Party Code Attribution

This repository is released under the Apache License 2.0 (see `LICENSE`). It builds on
and vendors code from the projects below. Third-party code remains under its original
license; all copyright notices are preserved in the source files, and modifications are
listed here.

---

## 1. FloodDiffusion (FD1 base)

**Location:** `methods/fd1/` and `common/` (model, dataset, metric and utility modules not
listed under other entries).
**Source:** FloodDiffusion, Shanda AI Research Tokyo.
**Project page:** https://shandaai.github.io/FloodDiffusion/ (paper: https://arxiv.org/abs/2512.03520).
**Assets:** https://huggingface.co/ShandaAI/FloodDiffusion.
**License:** Apache-2.0.
**Modifications:** Triangular Resampling training path (`methods/fd1/models/resampler.py`,
TR options in `methods/fd1/models/diffusion_forcing_wan.py`), long-sequence generation and
windowed evaluation, import paths adapted to this repository layout.

## 2. FloodDiffusion 2 (FD2 base)

**Locations:**

- `methods/fd2/upstream/FloodDiffusion2/`: unmodified copy of the official repository at
  commit `9761eb1c8506ad3f7f738d47a22ec51f885f87e9`, kept as the pinned reference together
  with its own `LICENSE` and `THIRD_PARTY_LICENSES.md`.
- `methods/fd2/models/upstream/`: copy of its `models/` package.
- `methods/fd2/training/native/`: copy of its `metrics/`, `utils/`, `visualization/` packages
  and `train_df.py`; `methods/fd2/training/native_datasets/`: copy of its `datasets/`.

**Source:** FloodDiffusion 2, https://github.com/caiyy17/FloodDiffusion2
(assets: https://huggingface.co/AlayaLab/FloodDiffusion2).
**License:** MIT (Copyright (c) 2026 FloodDiffusion 2 authors).
**Modifications in the copies:** import statements rewritten for this repository's package
layout; `utils/paths.py` and `utils/initialize.py` read the paths file from
`methods/fd2/configs/` instead of `./configs/`; the default `device`
argument of `T5EncoderModel` in `tools/t5.py` no longer calls `torch.cuda.current_device()`
at import time (resolved at construction instead). No modelling, training or evaluation
logic was changed.

## 3. Alibaba Wan model components

**Files:**

- `methods/fd1/models/tools/{t5,wan_model,wan_vae,wan_vae_1d,attention,tokenizers}.py`
- `methods/fd2/models/upstream/tools/{t5,wan_model,wan_vae,attention,tokenizers}.py`
- the same files under `methods/fd2/upstream/FloodDiffusion2/models/tools/`

**Source:** Alibaba Wan Team, https://github.com/Wan-Video/Wan2.2.
**License:** Apache-2.0 (Copyright 2024-2025 The Alibaba Wan Team Authors).
**Modifications:** adapted for streaming motion generation (causal attention, stream mode,
context handling) by FloodDiffusion / FloodDiffusion 2. `t5.py` is also based on Hugging
Face Transformers (Apache-2.0; Copyright 2018 Mesh TensorFlow authors, T5 Authors and
HuggingFace Inc. team).

## 4. Quaternion operations

**Files:** `common/utils/math/quaternion.py`,
`methods/fd2/training/native/visualization/HumanML3D263/quaternion.py` (and the upstream copy).
**Source:** Facebook Research, VideoPose3D, https://github.com/facebookresearch/VideoPose3D.
**License:** BSD-3-Clause (Copyright (c) 2018-present, Facebook, Inc.).
**Modifications:** NumPy adaptation.

## 5. Text-to-motion evaluation metrics

**Files:** `common/metrics/tools/{t2m_evaluator,utils,word_vectorizer}.py`,
`methods/fd2/training/native/metrics/HumanML3D263/{t2m_evaluator,word_vectorizer}.py`,
`methods/fd2/training/native/metrics/tools/utils.py` (and the upstream copies).
**Source:** text-to-motion, https://github.com/EricGuo5513/text-to-motion.
**License:** MIT.
**Modifications:** adapted to this project's evaluation pipeline (R-precision, FID,
matching distance, diversity, multimodality).

## 6. HumanML3D motion representation

**Files:** `common/utils/motion_process.py` (263D recovery),
`methods/fd2/training/native/visualization/HumanML3D263/*`,
`methods/fd2/training/native/metrics/HumanML3D263/to_humanml3d.py`,
`methods/fd2/training/native/metrics/tools/joints_to_humanml3d.py`.
**Source:** HumanML3D (Guo et al., CVPR 2022), https://github.com/EricGuo5513/HumanML3D.
**License:** MIT (Copyright (c) 2022 Chuan Guo).

## 7. Other representations vendored with FloodDiffusion 2

The FD2 copies also contain decode/metric code for the MotionStreamer 272-D representation
(MIT, Copyright (c) 2025 Lixing Xiao, https://github.com/Li-xingXiao/272-dim-Motion-Representation),
the NVIDIA kimodo / ardy SOMA family (Apache-2.0, Copyright (c) 2026 NVIDIA CORPORATION &
AFFILIATES, https://github.com/nv-tlabs/kimodo) and the MEI 138-D representation (MIT,
Copyright (c) 2026 CAI YIYI, https://github.com/caiyy17/MEI_Representation), together with
rotation utilities ported from PyTorch3D (BSD-3). They are not used by the 263D
experiments in this repository; see `methods/fd2/upstream/FloodDiffusion2/THIRD_PARTY_LICENSES.md`
for the full notices.

---

## License compatibility

| Component | License | Compatible with Apache-2.0 |
| --- | --- | --- |
| FloodDiffusion | Apache-2.0 | yes |
| FloodDiffusion 2 | MIT | yes |
| Alibaba Wan 2.2 | Apache-2.0 | yes |
| VideoPose3D quaternion | BSD-3-Clause | yes |
| text-to-motion / HumanML3D | MIT | yes |
| MotionStreamer, MEI | MIT | yes |
| NVIDIA kimodo / ardy | Apache-2.0 | yes |

Datasets (HumanML3D, BABEL / AMASS), body models (SMPL-H, SMPL-X) and pretrained weights
are **not** distributed with this repository and remain subject to their own terms. The
Apache-2.0 license of this repository covers the code only.

## Data release

Our data release on Hugging Face (https://huggingface.co/KunhangLI/TriangularResampling,
downloaded by `common/download_data.py`) contains extracted 263D motion features, text
annotations, VAE tokens and split / membership lists derived from
[BABEL](https://babel.is.tue.mpg.de/) (Punnakkal et al., CVPR 2021), which builds on
[AMASS](https://amass.is.tue.mpg.de/) (Mahmood et al., ICCV 2019). No raw AMASS data (SMPL
parameters or meshes) is distributed. The data is provided for non-commercial scientific
research only; by using it you agree to the BABEL and AMASS licenses.
