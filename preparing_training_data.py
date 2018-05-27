import os.path
import cv2
from imutils import paths
import numpy as np
import pandas as pd
import imutils
import sys

LETTER_IMAGES_FOLDER = "letter_imgs"
TRAINING_DATA = 'train.csv'
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

data = []


def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
                               cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image


for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    label = image_file.split(os.path.sep)[-2]
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = resize_to_fit(image, 20, 20)
    image_pixels = np.array(image)
    flatten_pixels = image_pixels.flatten()
    row = np.append(flatten_pixels, label)
    data.append(row)

df = pd.DataFrame(data=data)
df.to_csv(TRAINING_DATA, sep=',', encoding='utf-8', index=False)
