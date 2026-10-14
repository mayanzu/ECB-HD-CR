"""Verify the distributed files; generated outputs are outside this manifest."""
import hashlib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent


def verify():
    count = 0
    for line in (ROOT/'SNAPSHOT_MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        checksum, name = line.split('  ', 1)
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('Unsafe manifest path')
        path = ROOT.joinpath(*relative.parts)
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError('Checksum mismatch: ' + name)
        count += 1
    print(f'Verified {count} distributed files.', flush=True)
    return count


if __name__ == '__main__':
    verify()
