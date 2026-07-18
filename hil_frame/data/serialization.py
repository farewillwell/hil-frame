from __future__ import annotations

import gzip
import hashlib
import pickle
from pathlib import Path
from typing import Any


def dumps(obj: Any) -> bytes:
    return gzip.compress(pickle.dumps(obj, protocol=5), mtime=0)


def loads(payload: bytes) -> Any:
    return pickle.loads(gzip.decompress(payload))


def checksum_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_payload(path: str | Path) -> tuple[bytes, str]:
    payload = Path(path).read_bytes()
    return payload, checksum_bytes(payload)
