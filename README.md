# model

## Setting up the training dataset

### Split the annotated dataset into classes
```
python split_data_to_classes.py
```

### Crop the images in the split dataset
```
python crop_detections.py
```

### Train the model
```
python model_train.py
```

### Make predictions
#### Options
```
➜ python model_predict.py -h
usage: model_predict.py [-h] -i INPUT [-r RECURSIVE] [-w WEIGHTS]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the image file or the images directory
  -r RECURSIVE, --recursive RECURSIVE
                        Find images recursively in the input folder
  -w WEIGHTS, --weights WEIGHTS
                        Path to the model weights to use. If empty, will use latest.
```

#### Usage example
```sh
python model_predict.py -i <path to cropped image or path to directory with cropped images>
```
