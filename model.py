import os
import datetime
import time
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)

import PIL.Image as Image
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from loguru import logger


def load_pretrained(pretrained_model='mobilenet_v2'):
    if pretrained_model == 'mobilenet_v2':
        classifier_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'
    elif pretrained_model == 'inception_v3':
        classifier_model = 'https://tfhub.dev/google/imagenet/inception_v3/classification/5'
    return classifier_model


def prepare_dataset(classifier_model, data_root='dataset_cropped'):
    IMAGE_SHAPE = (224, 224)
    classifier = tf.keras.Sequential(
        [hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE + (3, ))])

    batch_size = 32
    img_height = 224
    img_width = 224

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_root),
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_root),
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = np.array(train_ds.class_names)
    logger.debug(class_names)

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names


def feature_extractor(train_ds, loaded_feature_extractor_model='mobilenet_v2'):
    for image_batch, labels_batch in train_ds:
        logger.debug(image_batch.shape)
        logger.debug(labels_batch.shape)
        break
    if loaded_feature_extractor_model == 'mobilenet_v2':
        feature_extractor_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
    elif loaded_feature_extractor_model == 'mobilenet_v2':
        feature_extractor_model = 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4'

    feature_extractor_layer = hub.KerasLayer(feature_extractor_model,
                                             input_shape=(224, 224, 3),
                                             trainable=False)

    feature_batch = feature_extractor_layer(image_batch)
    logger.debug(f'feature_batch.shape: {feature_batch.shape}')

    num_classes = len(class_names)
    logger.debug(f'num_classes: {num_classes}')
    return feature_extractor_layer, image_batch


def build_model(feature_extractor_layer, image_batch):
    model = tf.keras.Sequential(
        [feature_extractor_layer,
         tf.keras.layers.Dense(num_classes)])
    model.summary()
    predictions = model(image_batch)
    logger.debug(predictions.shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc'])
    return model


def tb_callback():
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)
    return tensorboard_callback


def train_model(NUM_EPOCHS=100):
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=NUM_EPOCHS,
                        callbacks=[tb_callback()])
    return model, history


def predict(model, image_batch, plot_preds=False):
    predicted_batch = model.predict(image_batch)
    predicted_id = tf.math.argmax(predicted_batch, axis=-1)
    predicted_label_batch = class_names[predicted_id]
    logger.debug(predicted_label_batch)

    if plot_preds:
        plt.figure(figsize=(10, 9))
        plt.subplots_adjust(hspace=0.5)
        for n in range(30):
            plt.subplot(6, 5, n + 1)
            plt.imshow(image_batch[n])
            plt.title(predicted_label_batch[n].title())
            plt.axis('off')
        _ = plt.suptitle('Model predictions')


def export_model(model, class_names):
    export_path = f'saved_models/{int(time.time())}'
    model.save(export_path)
    np.save(f'saved_models/{int(time.time())}/class_names.npy', class_names)
    logger.debug(export_path)


def predict_from_exported(export_path,
                          class_names,
                          image_batch,
                          plot_preds=False):
    reloaded = tf.keras.models.load_model(export_path)
    reloaded_result_batch = reloaded.predict(image_batch)
    reloaded_predicted_id = tf.math.argmax(reloaded_result_batch, axis=-1)
    reloaded_predicted_label_batch = class_names[reloaded_predicted_id]
    logger.debug(f'Prediction: {reloaded_predicted_label_batch}')

    if plot_preds:
        plt.figure(figsize=(10, 9))
        plt.subplots_adjust(hspace=0.5)
        for n in range(30):
            plt.subplot(6, 5, n + 1)
            plt.imshow(image_batch[n])
            plt.title(reloaded_predicted_label_batch[n].title())
            plt.axis('off')
        _ = plt.suptitle('Model predictions')
    return reloaded_predicted_label_batch
