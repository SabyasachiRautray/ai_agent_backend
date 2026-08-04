# Pure-Python stand-in for the compiled xxhash package.
# Only used because Windows Application Control blocks the real DLL.
# langsmith only uses this for internal trace-ID hashing, so exact
# xxHash compatibility doesn't matter — any stable hash works.
import hashlib


def _digest(data: bytes, seed: int, size: int) -> bytes:
    return hashlib.blake2b(data, digest_size=size, key=(seed & 0xFFFFFFFF).to_bytes(4, "little")).digest()


class _Base:
    _size = 8

    def __init__(self, data: bytes = b"", seed: int = 0):
        self._seed = seed
        self._buf = bytearray(data)

    def update(self, data: bytes):
        self._buf.extend(data)
        return self

    def digest(self) -> bytes:
        return _digest(bytes(self._buf), self._seed, self._size)

    def intdigest(self) -> int:
        return int.from_bytes(self.digest(), "big")

    def hexdigest(self) -> str:
        return self.digest().hex()

    def reset(self):
        self._buf = bytearray()
        return self


class xxh32(_Base): _size = 4
class xxh64(_Base): _size = 8
class xxh3_64(_Base): _size = 8
class xxh3_128(_Base): _size = 16
xxh128 = xxh3_128


def _make(cls):
    return (
        lambda data, seed=0: cls(data, seed).digest(),
        lambda data, seed=0: cls(data, seed).intdigest(),
        lambda data, seed=0: cls(data, seed).hexdigest(),
    )


xxh32_digest, xxh32_intdigest, xxh32_hexdigest = _make(xxh32)
xxh64_digest, xxh64_intdigest, xxh64_hexdigest = _make(xxh64)
xxh3_64_digest, xxh3_64_intdigest, xxh3_64_hexdigest = _make(xxh3_64)
xxh3_128_digest, xxh3_128_intdigest, xxh3_128_hexdigest = _make(xxh3_128)
xxh128_digest, xxh128_intdigest, xxh128_hexdigest = xxh3_128_digest, xxh3_128_intdigest, xxh3_128_hexdigest

__version__ = "0.0.0-shim"