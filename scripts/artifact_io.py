"""Read an artifact, or its numbered byte parts, as one seekable binary file."""
from bisect import bisect_right
import io
import operator
import os
from pathlib import Path


class _SplitReader(io.RawIOBase):
    def __init__(self, parts):
        super().__init__()
        self._files = []
        self._offsets = [0]
        self._pos = 0
        try:
            for part in parts:
                f = part.open("rb")
                self._files.append(f)
                size = os.fstat(f.fileno()).st_size
                if not size:
                    raise ValueError(f"Empty artifact part: {part}")
                self._offsets.append(self._offsets[-1] + size)
        except BaseException:
            self.close()
            raise

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        if self.closed:
            raise ValueError("I/O operation on closed artifact")
        return self._pos

    def seek(self, offset, whence=os.SEEK_SET):
        if self.closed:
            raise ValueError("I/O operation on closed artifact")
        offset = operator.index(offset)
        if whence == os.SEEK_SET:
            pos = offset
        elif whence == os.SEEK_CUR:
            pos = self._pos + offset
        elif whence == os.SEEK_END:
            pos = self._offsets[-1] + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if pos < 0:
            raise ValueError("Negative seek position")
        self._pos = pos
        return pos

    def readinto(self, buffer):
        if self.closed:
            raise ValueError("I/O operation on closed artifact")
        view = memoryview(buffer).cast("B")
        done = 0
        while done < len(view) and self._pos < self._offsets[-1]:
            index = bisect_right(self._offsets, self._pos) - 1
            f = self._files[index]
            f.seek(self._pos - self._offsets[index])
            count = min(len(view) - done, self._offsets[index + 1] - self._pos)
            got = f.readinto(view[done:done + count])
            if got != count:
                raise OSError(f"Artifact part was truncated while reading: {f.name}")
            self._pos += got
            done += got
        return done

    def close(self):
        try:
            for f in self._files:
                f.close()
        finally:
            super().close()


def open_artifact(path):
    """Open PATH or consecutive PATH.part000, PATH.part001, ... for binary reads.

    The caller owns the returned stream. Parts are never joined on disk or loaded
    wholesale into memory. A bundle's MANIFEST.sha256 lists each part with its own hash and size.
    """
    path = Path(path)
    if path.exists():
        return path.open("rb")
    parts = sorted(path.parent.glob(path.name + ".part*"))
    if not parts:
        raise FileNotFoundError(f"Missing artifact {path} (or numbered .part files)")
    for index, part in enumerate(parts):
        expected = f"{path.name}.part{index:03d}"
        if part.name != expected:
            raise ValueError(f"Missing or misnumbered artifact part: expected {expected}, found {part.name}")
    return io.BufferedReader(_SplitReader(parts))
