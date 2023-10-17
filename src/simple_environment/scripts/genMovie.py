#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata
import seaborn as sns
import toml
import argparse
from matplotlib.colors import ListedColormap
#file_name="MAPELITES_BC_sim122_elites_freq1.csv"
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import glob
import cv2

from simple_environment.util.SearchHelper import *
from simple_environment.util.bc_calculate import *


def createMovie(folderPath, filename):
    globStr = os.path.join(folderPath, '*.png')
    imageFiles = sorted(glob.glob(globStr))

    # Grab the dimensions of the image
    img = cv2.imread(imageFiles[0])
    imageDims = img.shape[:2][::-1]

    # Create a video
    fourcc = cv2.cv.CV_FOURCC(*'MP4V')
    frameRate = 30
    video = cv2.VideoWriter(filename, fourcc, frameRate, imageDims)

    for imgFilename in imageFiles:
        img = cv2.imread(imgFilename)
        video.write(img)

    video.release()



if __name__ == "__main__":
    movieFilename = 'fitness-random.mp4'
    imageFolder = "../images-random/"
    createMovie(imageFolder, movieFilename) 