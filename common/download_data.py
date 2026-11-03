"""Download the Triangular Resampling data release from Hugging Face.

The release contains

- ``BABEL.zip``: our 263D BABEL data (full BABEL sequences at 20 fps), extracted to
  ``<raw_data>/BABEL/``: ``new_joint_vecs/``, ``texts/``, the FD1 VAE tokens
  ``TOKENS_20251030_085836_vae_wan_z4/`` and the split files;
- ``manifests.zip``: FD2 metadata, extracted to ``<raw_data>/manifests/``: the frozen
  HumanML3D + BABEL training membership ``humanml_babel.json``.

HumanML3D itself is not part of this release; it comes with the FloodDiffusion
release (``python -m methods.fd1.download_assets --with-dataset``).

Every file is checked against the SHA-256 pinned below before it is used.

    python -m common.download_data                      # BABEL + FD2 manifests
    python -m common.download_data --items babel        # BABEL only
"""
import argparse
import hashlib
import zipfile
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download

REPO_ID = "KunhangLI/TriangularResampling"
REVISION = "b78fdff596cd8f371df7c2557224d2dac61252b0"
FILES = {
    # name: (archive in the Hugging Face repository, SHA-256, top-level directory inside it)
    "babel": ("BABEL.zip", "681fdd86ebc70d24bec18b1b83e04f0070a8ae8fc3cd3a33ee7f1cd7ba2dfdcf", "BABEL"),
    "manifests": ("manifests.zip", "71b2b95ff2c072500562aa8ebe2b42d03dbbd4102b2d07f7e9c83e566d88cc32", "manifests"),
}


def sha256(path, block=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(item, revision=REVISION):
    """Download one release file into the Hugging Face cache and verify it."""
    filename, expected, _ = FILES[item]
    print(f"Downloading {filename} from {REPO_ID}@{revision} ...")
    path = Path(hf_hub_download(repo_id=REPO_ID, filename=filename, revision=revision))
    if expected is not None:
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"{filename}: SHA-256 {actual} does not match the pinned {expected}")
    return path


def extract(archive, raw_data, top):
    """Extract a release archive into raw_data/; it may only contain paths under top/."""
    raw_data = Path(raw_data)
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            parts = PurePosixPath(name).parts
            if not parts or parts[0] != top or ".." in parts or PurePosixPath(name).is_absolute():
                raise SystemExit(f"unexpected path in {archive}: {name!r}")
        print(f"Extracting {Path(archive).name} to {raw_data}/ ...")
        handle.extractall(raw_data)
    return raw_data / top


def download(raw_data="raw_data", items=("babel", "manifests"), revision=REVISION):
    raw_data = Path(raw_data)
    raw_data.mkdir(parents=True, exist_ok=True)
    done = {item: extract(fetch(item, revision), raw_data, FILES[item][2]) for item in items}
    for item, path in done.items():
        print(f"DATA_OK {item}: {path}")
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-data", default="raw_data", help="destination raw-data directory")
    parser.add_argument("--items", default="babel,manifests",
                        help="comma-separated subset of: " + ", ".join(FILES))
    parser.add_argument("--revision", default=REVISION, help="Hugging Face revision to download")
    args = parser.parse_args()
    items = [item.strip() for item in args.items.split(",") if item.strip()]
    unknown = sorted(set(items) - set(FILES))
    if unknown:
        parser.error(f"unknown items: {unknown}")
    download(args.raw_data, items, args.revision)


if __name__ == "__main__":
    main()
