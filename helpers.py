import json
import os
import shutil
import tarfile
from glob import glob
from pathlib import Path

import ipyplot
import requests
from loguru import logger


def load_export_data(project_id, TOKEN, export_all=False):
    url = f'https://ls.aibird.me/api/projects/{project_id}/export?exportType=JSON_MIN'
    if export_all:
        url = f'https://ls.aibird.me/api/projects/{project_id}/export?exportType=JSON&download_all_tasks=true'
    headers = requests.structures.CaseInsensitiveDict()
    headers["Authorization"] = f'Token {os.environ["TOKEN"]}'
    resp = requests.get(url, headers=headers)
    data = resp.json()
    data = json.dumps(data).replace('ls.aibird.me/data/local-files/?d=',
                                    'srv.aibird.me/')
    data = json.loads(data)
    return data


def to_tar(input_path):
    folder_name = Path(input_path).name
    with tarfile.open(f'{folder_name}.tar', 'w') as tar:
        tar.add(input_path, folder_name)
    logger.info(f'Archived {input_path}')


def plot_batch(image_batch, reloaded_predicted_label_batch, fname=None):
    html_ = ipyplot.plot_images(image_batch, reloaded_predicted_label_batch)
    if not fname:
        fname = f'plot_{int(time.time())}.html'
    with open(fname, 'w') as f:
        f.write(html_)


def change_path_in_data_file():
    data_files = glob('picam/**/data_*.json', recursive=True)
    for data_file in data_files:
        with open(data_file) as f:
            from_ = '/gpfs_common/share03/rwkays/malyeta/megadetector/picam'
            from_2 = '/gpfs_common/share03/rwkays/malyeta/megadetector_picam/picam'
            from_3 = '/gpfs_common/share03/rwkays/bdscholt/megadetector_picam/picam'
            to_ = str(
                Path(f'picam/{Path(data_file).parts[-2]}/with_detections').
                absolute())
            lines = f.read().replace(from_,
                                     to_).replace(from_2,
                                                  to_).replace(from_3, to_)

        with open(data_file, 'w') as f:
            f.write(lines)


def generate_cropping_code(picam_root_folder):  # '../picam'
    folders = glob(f'{picam_root_folder}/**')
    for folder in folders:
        Path(f'{folder}/with_detections_cropped').mkdir(exist_ok=True)
        if glob(f'{folder}/data_*.json'):
            data_file = glob(f'{folder}/data_*.json')[0]
            print(f'cd {Path(data_file).parent}; python ../../crop_detections.py {Path(data_file).name} {folder}/with_detections_cropped -i {folder}/with_detections')
            print()
            print('mv with_detections_cropped$(pwd)/with_detections _with_detections_cropped && rm -rf with_detections_cropped && mv _with_detections_cropped with_detections_cropped')
            print('\n\n--------------------------\n\n')


def rename_cropped_files(picam_root_folder):  # '../picam'
    folders = glob(f'{picam_root_folder}/**/with_detections_cropped')
    for folder in folders:
        files = glob(f'{folder}/*.jpg')
        for file in files:
            for x in range(10):
                new_name = file.replace(f'.jpg__', '').replace('_mdvunknown', '')
                if '_mdvunknown' not in new_name:
                    break
            Path(file).rename(new_name)
