"""Native curl downloads; resumable wheels verified against PyPI SHA-256."""
import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    'transformers': '5.20.2', 'bitsandbytes': '0.50.2', 'torchvision': '0.25.0',
    'huggingface-hub': '2.2.0',
    'tokenizers': '0.23.3', 'typer': '0.27.3', 'hf-xet': '1.7.0',
    'httpx2': '2.16.0', 'httpcore2': '2.16.0', 'truststore': '0.10.5',
    'h11': '0.16.0', 'annotated-doc': '0.0.5', 'shellingham': '1.5.4',
}


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(item):
    name, version = item
    metadata = subprocess.run(['curl.exe', '-fLsS', '--max-time', '30',
                               f'https://pypi.org/pypi/{name}/{version}/json'],
                              capture_output=True, check=True, text=True, encoding='utf-8')
    info = json.loads(metadata.stdout)
    compatible = re.compile(r'(py3-none-any|cp3(8|9|10|11|12|13)-abi3-win_amd64|cp313-cp313-win_amd64|py3-none-win_amd64)\.whl$')
    wheel = next(file for file in info['urls'] if compatible.search(file['filename']))
    if Path(wheel['filename']).name != wheel['filename']:
        raise ValueError('Invalid wheel filename')
    target = ROOT / 'tmp' / 'wheels' / wheel['filename']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve().parent != (ROOT/'tmp'/'wheels').resolve():
        raise ValueError('Wheel target escapes the runtime cache')

    def quarantine():
        target.rename(target.with_name(target.name + '.corrupt-' + uuid4().hex))

    expected = wheel['digests']['sha256']
    if target.exists() and checksum(target) == expected:
        return str(target)
    if target.exists() and target.stat().st_size >= wheel['size']:
        quarantine()
    print(f'Downloading {name}=={version}', flush=True)
    command = ['curl.exe', '-fLsS', '--retry', '2', '--connect-timeout', '20', '--max-time', '1200']
    if target.exists():
        command += ['-C', '-']
    subprocess.run(command + ['-o', str(target), wheel['url']], check=True)
    if checksum(target) != expected:
        quarantine()
        raise ValueError(f'Checksum mismatch: {target.name}')
    print(f'Verified {name}=={version}', flush=True)
    return str(target)


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, PACKAGES.items()))
    return subprocess.call([sys.executable, '-B', '-m', 'pip', 'install', '--no-index',
                            '--find-links', str(ROOT/'tmp'/'wheels'),
                            'transformers==5.20.2', 'bitsandbytes==0.50.2',
                            'torchvision==0.25.0'])


if __name__ == '__main__':
    raise SystemExit(main())
