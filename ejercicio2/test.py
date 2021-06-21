import cv2 
import numpy as np
from glob import glob
from enum import Enum
import os
import sklearn 
import sklearn.neighbors
import matplotlib.pyplot as plt
import pickle
from evaluation import evaluate_detector, precision_and_recall, interpolated_average_precision
import sys
from image_utils import non_max_suppression

#Region HOG
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
#Endregion


def extract_hog_features(img):
	resized_img = resize(img, (128*4, 64*4))

	fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
	cells_per_block=(2, 2), visualize=True)

	plt.axis("off")
	plt.imshow(hog_image, cmap="gray")
	return fd

imgdir = 'C:\\Users\\joaco\\Desktop\\Computer vision ejercicio 2\\data\\face_detection\\val_face_detection_images\\seen_Eric_Bana_0001.jpg'
img = cv2.imread(imgdir,cv2.IMREAD_GRAYSCALE)
print(extract_hog_features(img))