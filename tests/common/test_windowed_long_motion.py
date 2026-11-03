import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from methods.fd1.generate_long_sequences import _as_list, _save_motion_npz
from common.metrics.tools.utils import calculate_kid
from common.evaluation.eval_windowed_long_motion import (
    frechet_distance,
    load_generated_npz,
    load_manifest,
    r_precision_at_k,
    slice_motion_windows,
    summarize_curve,
)


class WindowedLongMotionTests(unittest.TestCase):
    def test_scalar_cli_value_becomes_singleton_list(self):
        self.assertEqual(_as_list(42), [42])

    def test_kid_small_samples_shift_and_reproducibility(self):
        rng = np.random.default_rng(7)
        # Still below the legacy hard-coded 1000 subset size, but large enough
        # that the unbiased finite-sample KID estimate is close to zero.
        real = rng.normal(size=(512, 16))
        identical_a = calculate_kid(real, real.copy(), n_subsets=10, seed=13)
        identical_b = calculate_kid(real, real.copy(), n_subsets=10, seed=13)
        shifted = calculate_kid(real, real + 2.0, n_subsets=10, seed=13)
        self.assertEqual(identical_a, identical_b)
        self.assertLess(abs(identical_a[0]), 0.1)
        self.assertGreater(shifted[0], identical_a[0])
        self.assertTrue(np.isfinite(identical_a).all())

    def test_kid_requires_two_samples(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            calculate_kid(np.zeros((1, 4)), np.zeros((3, 4)))

    def test_fid_identical_zero_and_shift_positive(self):
        rng = np.random.default_rng(3)
        embeddings = rng.normal(size=(80, 12))
        self.assertAlmostEqual(
            frechet_distance(embeddings, embeddings.copy()), 0.0, places=8)
        self.assertGreater(frechet_distance(embeddings, embeddings + 1.5), 0.0)

    def test_window_slicing_exact_and_partial_tail_warning(self):
        motion = np.zeros((2400, 138), dtype=np.float32)
        windows = slice_motion_windows(
            motion, window_frames=200, expected_windows=12)
        self.assertEqual(len(windows), 12)
        self.assertTrue(all(window.shape == (200, 138) for window in windows))

        # warnings.catch_warnings instead of assertWarns: the latter probes every
        # loaded module for __warningregistry__, which trips lazy imports.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            partial = slice_motion_windows(
                np.zeros((450, 138), dtype=np.float32), window_frames=200)
        self.assertEqual(len(partial), 2)
        self.assertTrue(any(issubclass(w.category, RuntimeWarning)
                            and "partial tail" in str(w.message) for w in caught))

    def test_linear_curve_summary_units(self):
        times = np.arange(5.0, 116.0, 10.0)
        values = 2.0 + 0.1 * times
        summary = summarize_curve(times, values)
        self.assertAlmostEqual(summary["delta"], 11.0)
        self.assertAlmostEqual(summary["normalized_auc"], 8.0)
        self.assertAlmostEqual(summary["slope_per_minute"], 6.0)

        r_summary = summarize_curve(
            times, 0.8 - 0.001 * times, r_precision=True)
        self.assertAlmostEqual(
            r_summary["slope_percentage_points_per_minute"], -6.0)

    def test_r_at_3_aligned_and_shuffled(self):
        rng = np.random.default_rng(19)
        text = rng.normal(size=(16, 32))
        aligned, count, official = r_precision_at_k(
            text, text.copy(), top_k=3)
        shuffled, _, _ = r_precision_at_k(
            text, np.roll(text, 5, axis=0), top_k=3)
        self.assertAlmostEqual(aligned, 1.0)
        self.assertLess(shuffled, aligned)
        self.assertEqual(count, 16)
        self.assertFalse(official)

    def test_duplicate_caption_manifest_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.jsonl"
            items = [
                {"prompt_id": "a", "caption": "A person walks!",
                 "text_tokens": ["walk/VERB"]},
                {"prompt_id": "b", "caption": "a PERSON walks",
                 "text_tokens": ["walk/VERB"]},
            ]
            path.write_text(
                "".join(json.dumps(item) + "\n" for item in items),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate normalized caption"):
                load_manifest(path)

    def test_metric_npz_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "120s" / "prompt" / "seed_42" / "rf.npz"
            motion = np.arange(2400 * 138, dtype=np.float32).reshape(2400, 138)
            _save_motion_npz(
                str(path), motion, fps=20, seed=42, target_seconds=120,
                prompt_id="prompt", caption="a person walks",
                text_tokens=["a/DET", "person/NOUN", "walks/VERB"],
                method="rf", checkpoint="outputs/rf/step_step=15000.ckpt")
            loaded = load_generated_npz(path)
            self.assertEqual(loaded["motion"].dtype, np.float32)
            self.assertEqual(loaded["motion"].shape, (2400, 138))
            self.assertEqual(
                loaded["text_tokens"], ["a/DET", "person/NOUN", "walks/VERB"])
            self.assertEqual(
                loaded["checkpoint"], "outputs/rf/step_step=15000.ckpt")


if __name__ == "__main__":
    unittest.main()
