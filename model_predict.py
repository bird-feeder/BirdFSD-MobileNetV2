import argparse
import os
import warnings
from glob import glob
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from loguru import logger
from rich.console import Console
from rich.table import Table


def preprocess(image_path):
    logger.debug(f'Image path: {image_path}')
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image


def create_model(class_names):
    classifier_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'
    classifier = tf.keras.Sequential(
        [hub.KerasLayer(classifier_model, input_shape=(224, 224) + (3, ))])
    feature_extractor_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
    feature_extractor_layer = hub.KerasLayer(feature_extractor_model,
                                             input_shape=(224, 224, 3),
                                             trainable=False)
    class_names = np.load(class_names)
    num_classes = len(class_names)
    model = tf.keras.Sequential(
        [feature_extractor_layer,
         tf.keras.layers.Dense(num_classes, activation=tf.keras.activations.softmax)])
    return model


def predict_from_exported(model, pretrained_weights_path, class_names,
                          image_batch):
    reloaded_result_batch = model.predict(image_batch)
    reloaded_predicted_id = tf.math.argmax(reloaded_result_batch, axis=-1)
    prob = reloaded_result_batch.flatten()[reloaded_predicted_id]
    prob = round(float(prob), 2)

    if isinstance(class_names, str):
        class_names = np.load(class_names)
    reloaded_predicted_label_batch = class_names[reloaded_predicted_id]
    logger.debug(f'Prediction: {reloaded_predicted_label_batch} ({prob})')
    return reloaded_predicted_label_batch, prob


def list_input(args):
    exts = ['jpg', 'jpeg', 'JPG', 'JPEG']
    if Path(args.input).is_dir():
        if args.recursive:
            input_files = sum([glob(f'{args.input}/**/*.{ext}', recursive=True) for ext in exts], [])
        else:
            input_files = sum([glob(f'{args.input}/*.{ext}') for ext in exts], [])
    else:
        input_files = [args.input]
    return input_files


def pretty_table(list_):
    results_df = pd.DataFrame(list_)
    table = Table(title='Model Predictions', style='#44475a')
    for col, style in zip(results_df.columns, ['#f1fa8c', '#8be9fd', '#bd93f9']):
        table.add_column(col, style=style)
    for val in results_df.values:
        table.add_row(*[str(x) for x in val])
    Console().print(table)


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the image file or the images directory')
    parser.add_argument('-r', '--recursive', help='Find images recursively in the input folder')
    return parser.parse_args()


if __name__ == '__main__':
    args = opts()

    class_names = 'class_names.npy'
    pretrained_weights = f'{Path(__file__).parent}/weights/03_14_2022__1647175692.h5'
    model = create_model(class_names)
    model.load_weights(pretrained_weights)

    input_files = list_input(args)
    results = []
    
    for input_file in input_files:
        image = preprocess(input_file)
        pred, score = predict_from_exported(model, pretrained_weights, class_names, image)
        results.append({'Image': '/'.join(Path(input_file).parts[-2:]), 'Prediction': pred, 'Probability': score})

    pretty_table(results)
