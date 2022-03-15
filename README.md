# model

## Setting up the training dataset

### Split the annotated dataset into classes
```sh
python split_data_to_classes.py
```

### Crop the images in the split dataset
```sh
python crop_detections.py
```

## Train the model
```sh
python model_train.py
```

## Make predictions
### Options
```sh
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

### Usage example
```sh
python model_predict.py -i <path>  # <path> is path to a cropped image or to a directory with cropped images
```

## Apply predictions to a label-studio project
### Options
```sh
➜ python apply_predictions.py -h
usage: apply_predictions.py [-h] -p PROJECT_ID [-w WEIGHTS] [-s MIN_SCORE]

optional arguments:
  -h, --help            show this help message and exit
  -p PROJECT_ID, --project-id PROJECT_ID
                        Project id number
  -w WEIGHTS, --weights WEIGHTS
                        Path to the model weights to use. If empty, will use
                        latest
  -s MIN_SCORE, --min-score MIN_SCORE
                        Minimum prediction score to accept as valid
                        prediction. Accept all if left empty
```

### Usage example
```sh
python apply_predictions.py -p 2
```
