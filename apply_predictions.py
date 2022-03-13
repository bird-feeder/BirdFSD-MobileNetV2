import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from requests.structures import CaseInsensitiveDict
from loguru import logger
from tqdm import tqdm

import model_predict
from mongodb_helpers import get_mongodb_data


def make_headers():
    load_dotenv()
    TOKEN = os.environ['TOKEN']
    headers = CaseInsensitiveDict()
    headers["Content-type"] = "application/json"
    headers["Authorization"] = f"Token {TOKEN}"
    return headers


def get_all_tasks(project_id):
    logger.debug('Getting tasks data... This might take few minutes...')
    url = f"https://ls.aibird.me/api/projects/{project_id}/tasks?page_size=10000"
    headers = make_headers()
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


def main(task_id):
    headers = make_headers()
    url = f"https://ls.aibird.me/api/tasks/{task_id}"
    resp = requests.get(url, headers=headers)
    task_ = resp.json()
    if not task_['predictions']:
        img = task_['data']['image']
    else:
        return
    md_preds = find_image(Path(img).name)
    results = []
    scores = []
    for item in md_preds['detections']:
        if item['category'] != '1':
            continue
        x, y, width, height = [x * 100 for x in item['bbox']]

        for img_tuple in images:
            if img_tuple[0] == Path(img).name:
                logger.debug(Path(img).name)
                pred, prob = predict(img_tuple[1])
                scores.append(prob)
                break

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
        "model_version": "Megadetector",
        "result": results,
        "score": np.mean(scores),
        "cluster": 0,
        "neighbors": {},
        "mislabeling": 0,
        "task": task_id
    }

    url = "https://ls.aibird.me/api/predictions/"
    resp = requests.post(url, headers=headers, data=json.dumps(post_))
    logger.debug(resp.json())
    return resp


if __name__ == '__main__':
    logger.add('logs/apply_predictions.log')

    class_names = 'class_names.npy'
    pretrained_weights = 'weights/1647175692.h5'

    images = glob('dataset_cropped/**/*.jpg', recursive=True)
    images = [(Path(x).name, x) for x in images]

    # if len(sys.argv) == 1:
    #     raise Exception('You need to provide a path to the output data file!')
    # if not Path(sys.argv[1]).exists():
    #     raise FileNotFoundError('The path you entered does not exist!')
    # md_data_file = sys.argv[1]

    # with open(md_data_file) as j:
    #     md_data = json.load(j)

    md_data = get_mongodb_data()

    data = get_all_tasks(project_id=6)

    tasks_ids = [x['id'] for x in data]

    i = 0
    for cur_task in tqdm(tasks_ids):
        try:
            out = main(cur_task)
            if out:
                i += 1
        except KeyboardInterrupt:
            sys.exit('Interrupted by the user...')

    logger.info(f'Total number of predictions applied: {i}')
