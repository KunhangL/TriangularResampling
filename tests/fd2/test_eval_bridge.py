"""The FD2 evaluation bridge must point the shared evaluator at the right feature files."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from methods.fd2.training.evaluate import prepare_evaluation_config


class EvaluationBridgeTest(unittest.TestCase):
    def _check(self, absolute):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw_data = tmp / "raw_data"
            for dataset, name in (("HumanML3D", "000002"), ("BABEL", "7350")):
                for sub, ext in (("new_joint_vecs", ".npy"), ("texts", ".txt")):
                    (raw_data / dataset / sub).mkdir(parents=True, exist_ok=True)
                    (raw_data / dataset / sub / f"{name}{ext}").write_bytes(b"")
            prefix = raw_data if absolute else Path("raw_data")
            samples = [dict(dataset=d, name=n, feature_path=str(prefix / d / "new_joint_vecs" / f"{n}.npy"),
                            text_path=str(prefix / d / "texts" / f"{n}.txt"))
                       for d, n in (("HumanML3D", "000002"), ("BABEL", "7350"))]
            membership = raw_data / "manifests" / "humanml_babel.json"
            membership.parent.mkdir(parents=True)
            membership.write_text(json.dumps({"splits": {"train": [{"count": 2, "samples": samples}]}}))
            config = tmp / "fd2.yaml"
            OmegaConf.save(OmegaConf.create({"dirs": {"raw_data": "raw_data"},
                                             "data": {"membership_manifest": "raw_data/manifests/humanml_babel.json"}}),
                           config)
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                evaluation = OmegaConf.load(prepare_evaluation_config(str(config), "eval_configs"))
                for spec, probe in zip(evaluation.data.datasets, ("000002", "7350")):
                    root = os.path.dirname(spec.train_meta_paths[0])
                    # FD1 readers and the evaluator join <split dir>/<feature_path>/<id>.npy
                    self.assertTrue(os.path.isfile(os.path.join(root, spec.feature_path, probe + ".npy")), spec)
                    self.assertTrue(os.path.isfile(os.path.join(root, spec.text_path, probe + ".txt")), spec)
            finally:
                os.chdir(cwd)

    def test_relative_membership_paths(self):
        self._check(absolute=False)

    def test_absolute_membership_paths(self):
        self._check(absolute=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
