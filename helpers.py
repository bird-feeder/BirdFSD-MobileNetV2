import json
import os
import tarfile
from glob import glob
from pathlib import Path

import ipyplot
import requests
from dotenv import load_dotenv
from loguru import logger


def load_export_data(project_id, TOKEN, export_all=False):
    load_dotenv()
    url = f'{os.environ["LS_HOST"]}/api/projects/{project_id}/export?exportType=JSON_MIN'
    if export_all:
        url = f'{os.environ["LS_HOST"]}/api/projects/{project_id}/export?exportType=JSON&download_all_tasks=true'
    headers = requests.structures.CaseInsensitiveDict()
    headers["Authorization"] = f'Token {os.environ["TOKEN"]}'
    resp = requests.get(url, headers=headers)
    data = resp.json()
    LS_domain_name = os.environ['LS_HOST'].split('//')[1]
    SRV_domain_name = os.environ['SRV_HOST'].split('//')[1]
    data = json.dumps(data).replace(
        f'{os.environ["LS_domain_name"]}/data/local-files/?d=',
        f'{os.environ["SRV_domain_name"]}/')
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


def _change_path_in_data_file(paths_to_remove):
    data_files = glob('picam/**/data_*.json', recursive=True)
    for data_file in data_files:
        with open(data_file) as f:
            to_ = str(
                Path(f'picam/{Path(data_file).parts[-2]}/with_detections').
                absolute())
            for _path in paths_to_remove:
                lines = f.read().replace(_path, to_)

        with open(data_file, 'w') as f:
            f.write(lines)


def _generate_lazy_cropping_code(picam_root_folder):  # '../picam'
    """Requires https://github.com/microsoft/CameraTraps/blob/main/classification/crop_detections.py"""
    folders = glob(f'{picam_root_folder}/**')
    for folder in folders:
        Path(f'{folder}/with_detections_cropped').mkdir(exist_ok=True)
        if glob(f'{folder}/data_*.json'):
            data_file = glob(f'{folder}/data_*.json')[0]
            print(
                f'cd {Path(data_file).parent}; python ../../crop_detections.py {Path(data_file).name} {folder}/with_detections_cropped -i {folder}/with_detections'
            )
            print()
            print(
                'mv with_detections_cropped$(pwd)/with_detections _with_detections_cropped && rm -rf with_detections_cropped && mv _with_detections_cropped with_detections_cropped'
            )
            print('\n\n--------------------------\n\n')


def _rename_cropped_files(picam_root_folder):  # '../picam'
    folders = glob(f'{picam_root_folder}/**/with_detections_cropped')
    for folder in folders:
        files = glob(f'{folder}/*.jpg')
        for file in files:
            for x in range(10):
                new_name = file.replace(f'.jpg__',
                                        '').replace('_mdvunknown', '')
                if '_mdvunknown' not in new_name:
                    break
            Path(file).rename(new_name)
