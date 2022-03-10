import concurrent.futures
import json
import sys
from pathlib import Path
from urllib.error import HTTPError

import requests
from loguru import logger
from tqdm import tqdm


def main(export_file):
    with open(export_file) as j:
        data = json.load(j)

    labels = list(
        set(
            sum(
                sum([[x['rectanglelabels'] for x in data[n]['label']]
                     for n in range(len(data))
                     if 'label' in list(data[n].keys())], []), [])))

    Path(f'dataset').mkdir(exist_ok=True)

    for label in labels:
        Path(f'dataset/{label}').mkdir(exist_ok=True)

    not_downloaded = []

    def process(x):
        global i
        i += 1
        try:
            for label in x['label']:
                r = requests.get(x['image'])
                with open(
                        f'dataset/{label["rectanglelabels"][0]}/{Path(x["image"]).name}',
                        'wb') as f:
                    f.write(r.content)
        except (HTTPError, KeyError) as e:
            logger.warning(f'{sys.exc_info()}: {x}')
            not_downloaded.append(x)
        logger.debug(f'{i}/{len(data)}')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        results = [executor.submit(process, x) for x in data]
        for future in concurrent.futures.as_completed(results):
            futures.append(future.result())

    with open('not_downloaded.json', 'w') as j:
        json.dump(not_downloaded, j, indent=4)


# if __name__ == '__main__':
#     i = 0
#     main('project-1-at-2022-03-10-06-43-25f7f02b.json')
