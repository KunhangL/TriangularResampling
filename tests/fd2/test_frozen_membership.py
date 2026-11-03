"""The frozen FD2 membership: hash pin, and validation skips the samples prepare_data flagged."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from methods.fd2.training.data import FrozenMotionDataset


def sample(name):
    return dict(dataset='HumanML3D', name=name, feature_length=40,
                feature_path=f'raw_data/HumanML3D/new_joint_vecs/{name}.npy',
                text_path=f'raw_data/HumanML3D/texts/{name}.txt')


class FrozenMembershipTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'humanml_babel.json'
        membership = dict(splits=dict(train=[dict(count=2, samples=[sample('a'), sample('b')])],
                                      val=[dict(count=3, samples=[sample('a'), sample('b'), sample('c')])]),
                          problems=['val:raw_data/HumanML3D/new_joint_vecs/b.npy non-finite values'])
        self.path.write_text(json.dumps(membership))

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, sha=None):
        sha = sha or hashlib.sha256(self.path.read_bytes()).hexdigest()
        return OmegaConf.create(dict(data=dict(membership_manifest=str(self.path), membership_sha256=sha),
                                     model=dict(params=dict(input_dim=263))))

    def test_validation_skips_flagged_samples(self):
        self.assertEqual([x['name'] for x in FrozenMotionDataset(self.config(), 'val').records], ['a', 'c'])

    def test_training_keeps_every_sample(self):
        self.assertEqual([x['name'] for x in FrozenMotionDataset(self.config(), 'train').records], ['a', 'b'])

    def test_membership_hash_is_checked(self):
        with self.assertRaises(ValueError):
            FrozenMotionDataset(self.config(sha='0' * 64), 'val')


if __name__ == '__main__':
    unittest.main(verbosity=2)
