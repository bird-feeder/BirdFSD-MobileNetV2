import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from loguru import logger
from tensorflow.keras.applications.resnet50 import decode_predictions


def preprocess(image_path):
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


if __name__ == '__main__':
    class_names = 'class_names.npy'
    pretrained_weights = 'weights/1647175692.h5'
    model = create_model(class_names)
    model.load_weights(pretrained_weights)

    image_path = 'dataset_cropped/Carolina Chickadee/mo-picam1-4879.jpg'
    image = preprocess(image_path)

    predict_from_exported(model, pretrained_weights, class_names, image)
