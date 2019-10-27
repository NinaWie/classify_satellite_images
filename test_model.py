import numpy as np
import os
import cv2
import json
import keras
import argparse
import sys

from keras.applications.vgg16 import preprocess_input
from keras.models import model_from_json


def classify_tif_image(k, l):
    """
    This loads the TIF file and creates the 256*256 tiles out of it. it then classifies each tile to one of the 9 classes.

    :param k:
    :param l:
    :return:
    """

    img = cv2.imread(inp_path + "/{}-{}.tif".format(k, l))
    print(inp_path + "/{}-{}.tif".format(k, l))
    print("Finished Loading Image")
    shape = img.shape
    imcount = 0
    img_arr = []
    filenames = []
    for i in range(0, shape[0] - shape[0] % 256, 256):
        for j in range(0, shape[1] - shape[1] % 256, 256):
            tile = img[i:i + 256, j:j + 256, :]
            assert (tile.shape == (256, 256, 3))
            imcount += 1
            img_arr.append(tile)
            filenames.append("{}-{}_{}_{}".format(k, l, i, j))
    assert (len(filenames) == len(img_arr))
    img_arr = np.asarray(img_arr)
    print(img_arr.shape)

    # final data:
    img_data = preprocess_input(img_arr.astype(np.float))

    sizes = img_data.shape
    print(img_data.shape)

    # load json and create model
    json_file = open(os.path.join(model_path, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(model_path, "model.h5"))
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    out = loaded_model.predict(img_data)

    mapping = json.load(open(mapping_file, "r"))

    res_dir = {filenames[i]: str(mapping[str(np.argmax(out[i]))]) for i in range(len(out))}

    output_path = "{}-{}_pred_labels.json".format(k, l)
    json.dump(res_dir, open(os.path.join(args.out_dir, output_path), "w"))
    print("Saved predicted labels in a dictionary in ", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train/test neural network for recognizing pitch type from joint trajectories')
    parser.add_argument('-path_to_data', help='path to data to predict labels for - e.g. tiles', required=True)
    parser.add_argument('-path_to_model', help='path to model.h5 and model.json', required=True)
    parser.add_argument('-mapping_file', help='path to mapping file', required=True)
    parser.add_argument('-out_dir', default=".", help='path to output the predicted labels', required=False)
    parser.add_argument('-start', default="1", help='number from which image it should start', required=False)
    parser.add_argument('-end', default="2", help='number to which image it should process', required=False)
    args = parser.parse_args()

    inp_path = args.path_to_data
    model_path = args.path_to_model
    mapping_file = args.mapping_file
    from_num = int(args.start)
    to_num = int(args.end)

    if not os.path.exists:
        print("ERROR: PATH DOES NOT EXIST!")
        sys.exit()
    for k in range(from_num, to_num):
        for l in range(17, 20, 2):
            classify_tif_image(k, l)
