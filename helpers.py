import json
import os
import tarfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger


def load_export_data(project_id, TOKEN):
    url = f'https://ls.aibird.me/api/projects/{project_id}/export?exportType=JSON_MIN'
    headers = requests.structures.CaseInsensitiveDict()
    headers["Authorization"] = f'Token {os.environ["TOKEN"]}'
    resp = requests.get(url, headers=headers)
    data = resp.json()
    data = json.dumps(data).replace('ls.aibird.me/data/local-files/?d=', 'srv.aibird.me/')
    data = json.loads(data)
    return data


def to_tar(input_path):
    folder_name = Path(input_path).name
    with tarfile.open(f'{folder_name}.tar', 'w') as tar:
        tar.add(input_path, folder_name)
    logger.info(f'Archived {input_path}')
