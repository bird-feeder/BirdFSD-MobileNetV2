import hashlib
import json
import shutil
from glob import glob
from pathlib import Path

from loguru import logger
from tqdm import tqdm


def gen_hash(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
    results = hashlib.md5(content).hexdigest()
    return results


def main(root_path='local-files/picam'):
    files = glob(f'{root_path}/**/*.jpg', recursive=True)
    with open('hashes.json') as j:
        stored_hashes = json.load(j)
    shutil.copy2('hashes.json', '_hashes.json')

    hashes = []
    i = 0
    for file in tqdm(files):
        _hash = {Path(file).name: gen_hash(file)}
        if _hash not in stored_hashes:
            i += 1
            hashes.append(_hash)
    logger.debug(f'Hashed {i} new images')

    with open('hashes.json', 'w') as j:
        json.dump(stored_hashes + hashes, j)

    Path('_hashes.json').unlink()


if __name__ == '__main__':
    main()
