import math
import unittest
from methods.fd1.models.tr_schedule import resampling_shift


class ScheduleTest(unittest.TestCase):
    def test_legacy(self):
        for k in (0, 5000, 30000, float("inf")):
            self.assertEqual(resampling_shift(k), .6)

    def test_cosine(self):
        cfg = dict(mode="cosine_log", start=3, end=.6, duration=15000)
        self.assertEqual(resampling_shift(0, cfg), 3)
        self.assertAlmostEqual(resampling_shift(7500, cfg), math.sqrt(1.8))
        self.assertEqual(resampling_shift(15000, cfg), .6)
        self.assertEqual(resampling_shift(30000, cfg), .6)
        values = [resampling_shift(k, cfg) for k in range(30001)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertEqual(values[12345:], [resampling_shift(k, cfg) for k in range(12345, 30001)])

    def test_reject_invalid(self):
        for cfg in (dict(mode="bad"), dict(mode="constant", value=0),
                    dict(mode="cosine_log", duration=0)):
            with self.assertRaises(ValueError):
                resampling_shift(0, cfg)


if __name__ == "__main__":
    unittest.main()
