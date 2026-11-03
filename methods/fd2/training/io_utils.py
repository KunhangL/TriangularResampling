"""Small file helpers shared by the FD2 training, generation and evaluation entries."""
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    """SHA-256 of a file, streamed in 8 MiB blocks."""
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    """Write JSON through a temporary file so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)
