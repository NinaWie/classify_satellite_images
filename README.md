# Predict labels for satellite images 

The aim is to classify satellite images into region-type classes, where here we distinguish the 8 classes "city", "suburb", "grass", "forest", "forest-grass", "field", "houses-field", "water".

The idea is to train first in an unsupervised fashion in a three step process:
* Extract features with VGG16 pre-trained model
* simplify features with PCA
* K-Means clustering

Then, the labels can be inferred from the outputs, simply plotting the images belonging to one cluster and thereby giving the cluster some label. This is all documented in the notebook vgg16_kmeans.ipynb in this repository

Secondly, the labels attained this way can be used to train in a supervised fashion. Train a model with running `python train_model.py`, where you have to modify how the training data is loaded in the beginning of the file. 

Once train_model was successful and a model was saved, one can make inference on new data with test_model.py. The steps are explained more in detail further below. 

## Requirements

It is possible to run the code in a virtual environment where the requirements are loaded:
Create a virtual environment:

```sh
python3 -m venv env
```

Activate the environment:

```sh
source env/bin/activate
```

Install in editable mode for development:

```sh
pip install  -r requirements.txt
```

## Extract features and cluster

All code for extracting the features and clustering can be found in the notebook vgg_kmeans. You have to adjust the file paths, apart from that it should be well documented in the notebook itself how to run it.

### Dataset

One possible dataset that was used is the public [Merced Landuse Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html). Apart from that, I had private data available which I might make available soon.  

## Train DL model

I aim to update the code such that it is more flexible for different input data. So far, you need to adjust the first part of the code in the train_model.py file in order to load your own data. In the end, the image data need to be of size (#data x 256 x 256 x 3) and the ground truth labels need to be one hot encoded vectors of size (#data x #classes), but for one-hot-encoding you might also use the function provided. 

### Test DL model on new data

Run 
```bash
test_model.py [-h] -path_to_data PATH_TO_DATA -path_to_model
                     PATH_TO_MODEL -mapping_file MAPPING_FILE
                     [-out_dir OUT_DIR] [-start START] [-end END]
```

* path_to_data is some folder containing tif files. They are tiled up and processed one by one
* path_to_model is the patht to the model directory containing the model.h5 file and model.json file
* mapping_file specifies the path to the mapping.json file, which is located actually in this codebase and this directory as well
* out_dir: specify some directory to dump the output files (json files) 
* start and end: to process multiple tif files, specify which ones to start and end with
