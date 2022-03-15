import argparse
import imghdr
import json
import os
import shutil
import signal
import sys
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image
from requests.structures import CaseInsensitiveDict
from loguru import logger
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import model_predict
from mongodb_helpers import get_mongodb_data


def keyboard_interrupt_handler(sig, frame):
    logger.info(f'KeyboardInterrupt (ID: {sig}) has been caught...')
    sys.exit(0)


def mkdirs():
    Path('tmp').mkdir(exist_ok=True)
    Path('tmp/downloaded').mkdir(exist_ok=True)
    Path('tmp/cropped').mkdir(exist_ok=True)


def make_headers():
    load_dotenv()
    TOKEN = os.environ['TOKEN']
    headers = CaseInsensitiveDict()
    headers["Content-type"] = "application/json"
    headers["Authorization"] = f"Token {TOKEN}"
    return headers


def get_all_tasks(headers, project_id):
    logger.debug('Getting tasks data... This might take few minutes...')
    url = f'{os.environ["LS_HOST"]}/api/projects/{project_id}/tasks?page_size=10000'
    resp = requests.get(url,
                        headers=headers,
                        data=json.dumps({'project': project_id}))
    with open('tasks_latest.json', 'w') as j:
        json.dump(resp.json(), j)
    return resp.json()


def find_image(img_name):
    for im in md_data:
        if Path(im['file']).name == img_name:
            return im


def predict(image_path):
    model = model_predict.create_model(class_names)
    model.load_weights(pretrained_weights)
    image = model_predict.preprocess(image_path)
    pred, prob = model_predict.predict_from_exported(model, pretrained_weights,
                                                     class_names, image)
    return pred, prob


def load_local_image(img_path):
    """https://github.com/microsoft/CameraTraps/blob/main/classification/crop_detections.py"""
    try:
        with Image.open(img_path) as img:
            img.load()
        return img
    except OSError as e:
        exception_type = type(e).__name__
        logger.error(f'Unable to load {img_path}. {exception_type}: {e}.')
    return None


def save_crop(img, bbox_norm, square_crop, save):
    """https://github.com/microsoft/CameraTraps/blob/main/classification/crop_detections.py"""
    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    box_w = int(bbox_norm[2] * img_w)
    box_h = int(bbox_norm[3] * img_h)

    if square_crop:
        box_size = max(box_w, box_h)
        xmin = max(0, min(xmin - int((box_size - box_w) / 2), img_w - box_w))
        ymin = max(0, min(ymin - int((box_size - box_h) / 2), img_h - box_h))
        box_w = min(img_w, box_size)
        box_h = min(img_h, box_size)

    if box_w == 0 or box_h == 0:
        tqdm.write(f'Skipping size-0 crop (w={box_w}, h={box_h}) at {save}')
        return False

    crop = img.crop(box=[xmin, ymin, xmin + box_w,
                         ymin + box_h])  # [left, upper, right, lower]

    if square_crop and (box_w != box_h):
        crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)
    crop.save(save)
    return os.path.dirname(save)


def main(task_id):
    url = f'{os.environ["LS_HOST"]}/api/tasks/{task_id}'

    resp = requests.get(url, headers=headers)
    task_ = resp.json()
    if task_['predictions']:
        return
    img_in_task = task_['data']['image']

    LS_domain_name = os.environ['LS_HOST'].split('//')[1]
    SRV_domain_name = os.environ['SRV_HOST'].split('//')[1]
    url = task_['data']['image'].replace(f'{LS_domain_name}/data/local-files/?d=',
                                         f'{SRV_domain_name}/')
    img_name = Path(img_in_task).name
    img_relative_path = f'tmp/downloaded/{img_name}'

    bbox_res = find_image(img_name)

    r = requests.get(url)
    with open(img_relative_path, 'wb') as f:
        f.write(r.content)

    if not imghdr.what(img_relative_path):
        logger.error(f'Not a valid image file: {img_relative_path}')
        return

    img = load_local_image(img_relative_path)

    results = []
    scores = []
    bboxes = []

    for n, task in enumerate(bbox_res['detections']):
        if task['category'] != '1':
            continue

        bboxes.append([n, task['bbox']])
        out_cropped = f'tmp/cropped/{Path(img_name).stem}_{bboxes[0][0]}.jpg'
        save_crop(img, bboxes[0][1], False, out_cropped)

        pred, prob = predict(out_cropped)

        x, y, width, height = [x * 100 for x in task['bbox']]

        scores.append(prob)
        results.append({
            'from_name': 'label',
            'to_name': 'image',
            'type': 'rectanglelabels',
            'value': {
                'rectanglelabels': [pred],
                'x': x,
                'y': y,
                'width': width,
                'height': height
            },
            'score': prob
        })

    post_ = {
        'model_version': 'picam-detector_1647175692',
        'result': results,
        'score': np.mean(scores),
        'cluster': 0,
        'neighbors': {},
        'mislabeling': 0,
        'task': task_id
    }

    url = F'{os.environ["LS_HOST"]}/api/predictions/'
    resp = requests.post(url, headers=headers, data=json.dumps(post_))
    logger.debug(resp.json())


def get_weights(args):
    if not args.weights:
        try:
            pretrained_weights = sorted(
                glob(f'{Path(__file__).parent}/weights/*.h5'))[-1]
        except IndexError:
            raise FileNotFoundError(
                'No weights detected. You need to train the model at least once!'
            )
    else:
        pretrained_weights = args.weights
    logger.debug(f'Pretrained weights file: {pretrained_weights}')
    return pretrained_weights


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--project-id',
                        help='Project id number',
                        type=int,
                        required=True)
    parser.add_argument(
        '-w',
        '--weights',
        help='Path to the model weights to use. If empty, will use latest.',
        type=str,
        required=True,)
    return parser.parse_args()


if __name__ == '__main__':
    logger.add('apply_predictions.log')
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)
    args = opts()

    md_data = get_mongodb_data()
    headers = make_headers()
    mkdirs()

    class_names = 'class_names.npy'
    if not Path(class_names).exists():
        raise FileNotFoundError(
            'No class names detected. You need to train the model at least once!'
        )
    pretrained_weights = pretrained_weights()
    project_id = args.project_id

    project_tasks = get_all_tasks(headers, project_id)
    tasks_ids = [t_['id'] for t_ in project_tasks]

    logger.debug('Starting prediction...')
    try:
        for task_id in tqdm(tasks_ids):
            main(task_id)
    except Exception as e:
        logger.exception(e)
    finally:
        shutil.rmtree('tmp')
        sys.exit(0)
