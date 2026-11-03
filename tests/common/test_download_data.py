"""Offline checks of the data-release helper (no network access)."""
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from common.download_data import extract, sha256


class DownloadDataTest(unittest.TestCase):
    def test_sha256_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob"
            path.write_bytes(b"x" * 100_000)
            self.assertEqual(sha256(path, block=4096), hashlib.sha256(b"x" * 100_000).hexdigest())

    def test_extract_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "BABEL.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("BABEL/train.txt", "1\n")
                handle.writestr("BABEL/texts/1.txt", "walk#walk/VERB#0.0#1.0\n")
            root = extract(archive, Path(tmp) / "raw_data", "BABEL")
            self.assertEqual((root / "train.txt").read_text(), "1\n")
            self.assertTrue((root / "texts" / "1.txt").is_file())

    def test_extract_rejects_foreign_paths(self):
        for bad in ("../escape.txt", "HumanML3D/train.txt", "BABEL/../../escape.txt"):
            with tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / "BABEL.zip"
                with zipfile.ZipFile(archive, "w") as handle:
                    handle.writestr(bad, "x")
                with self.assertRaises(SystemExit):
                    extract(archive, Path(tmp) / "raw_data", "BABEL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
