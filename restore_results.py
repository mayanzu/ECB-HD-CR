"""Restore checked raw run records without replacing an existing experiment."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parent


def restore(version=None):
    restored = present = 0
    with zipfile.ZipFile(ROOT/'results/raw_runs.zip') as archive:
        manifest = json.loads(archive.read('MANIFEST.json'))
        if len(manifest) != 1250 or set(archive.namelist()) != set(manifest) | {'MANIFEST.json'}:
            raise ValueError('Unexpected raw archive inventory')
        for name, record in manifest.items():
            path = PurePosixPath(name)
            if (path.is_absolute() or '..' in path.parts or '\\' in name
                    or len(path.parts) < 4 or path.parts[0] not in {'thesis_final_v7','review_revision_v1'}
                    or path.parts[1] != 'runs' or path.suffix != '.pkl'):
                raise ValueError('Unsafe raw archive entry')
            if version and path.parts[0] != version:
                continue
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError('Corrupt raw record: ' + name)
            target = ROOT/'results'/Path(*path.parts)
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
                    raise FileExistsError('Existing result differs; preserve/move it before restoring: ' + name)
                present += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            restored += 1
    print(f'Raw results: restored={restored}, already verified={present}', flush=True)
    return restored + present


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=['thesis_final_v7','review_revision_v1'])
    restore(parser.parse_args().version)
