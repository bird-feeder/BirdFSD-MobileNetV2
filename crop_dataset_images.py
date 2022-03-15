import imghdr
import os
import shutil
from glob import glob
from pathlib import Path

import cv2
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from loguru import logger
from tqdm import tqdm

from helpers import load_export_data, to_tar


def get_img(img_url):
    fname = Path(img_url).name
    return [x for x in dataset_files if Path(x).name == fname][0]


def convert_from_ls(value):
    """https://labelstud.io/guide/export.html#Units-of-image-annotations"""
    if 'original_width' not in value or 'original_height' not in value:
        return None
    w, h = value['original_width'], value['original_height']

    if all([key in value for key in ['x', 'y', 'width', 'height']]):
        return w * value['x'] / 100.0, \
               h * value['y'] / 100.0, \
               w * value['width'] / 100.0, \
               h * value['height'] / 100.0


def display_image(image):
    plt.grid(False)
    plt.imshow(image)


def crop_images(item):
    for label in item['label']:
        label_name = label['rectanglelabels'][0]
        Path(f'dataset_cropped/{label_name}').mkdir(exist_ok=True)
        img_file = get_img(item['image'])
        img = cv2.imread(img_file)
        x, y, w, h = [round(x) for x in convert_from_ls(label)]
        crop_img = img[y:y + h, x:x + w]
        im = Image.fromarray(crop_img)
        im.save(f'dataset_cropped/{label_name}/{Path(item["image"]).name}')


def main(data):  # project export min-json file
    df = pd.DataFrame(data, columns=['id', 'annotation_id', 'image', 'label'])
    df.dropna(inplace=True)

    Path('dataset_cropped').mkdir(exist_ok=True)

    for item in tqdm(data):
        if not item.get('label'):
            logger.debug(f'No `label` key in: {item}')
            continue
        for label in item['label']:
            label_name = label['rectanglelabels'][0]
            Path(f'dataset_cropped/{label_name}').mkdir(exist_ok=True)
            img_file = get_img(item['image'])
            if not Path(img_file).exists() or not imghdr.what(img_file):
                logger.debug(f'Does not contain a valid image file {item}')
                continue
            img = cv2.imread(img_file)
            x, y, w, h = [round(x) for x in convert_from_ls(label)]
            crop_img = img[y:y + h, x:x + w]
            im = Image.fromarray(crop_img)
            im.save(f'dataset_cropped/{label_name}/{Path(item["image"]).name}',
                    format='JPEG',
                    quality=100,
                    subsampling=0)

    for _class in ['no animal', 'severe occultation', 'distorted image']:
        shutil.rmtree(f'dataset_cropped/{_class}')
    to_tar('dataset_cropped')


if __name__ == '__main__':
    load_dotenv()
    data = load_export_data(project_id=1, TOKEN=os.environ['TOKEN'])
    if not Path('dataset').exists():
        raise FileNotFoundError
    dataset_files = glob('dataset/**/*.jpg', recursive=True)
    main(data=data)
