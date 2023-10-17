import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import numpy as np
import math
import os
import cv2
import csv
import glob
import seaborn as sns
import pandas as pd
from itertools import product

feature1Label = 'Num Turns'
feature1Precision = 2
feature1Range = (-256.0, 256.0)
feature2Label = 'Hand Size'
feature2Precision = 2
feature2Range = (-256.0, 256.0)

fitness_range = (0.0, -5645.295058451705)
fitness_scalar = -100.0 / fitness_range[1]

#image_title = 'MAP-Elites'
image_title = 'CMA-ME'

def normalize(unscaled_fitness):
    return max(0.0, fitness_scalar * (unscaled_fitness - fitness_range[1]))

logPaths = [
    #'/home/tehqin/Projects/HearthStone/Experiments/toy/rastrigin100/me/',
    #'/home/tehqin/Projects/HearthStone/Experiments/toy/rastrigin100/cma-me-imp/',
    '/home/tehqin/Projects/HearthStone/QualDivBenchmark/StrategySearch/logs',
    #'/home/tehqin/Projects/HearthStone/Experiments/rogue/ME/trial3',
    #'/home/tehqin/Projects/HearthStone/Experiments/rogue/CMA-ME/trial4',
        ]

logFilename = "elite_map_log_0.csv"

def createRecordList(mapData):
    recordList = []
    for cellData in mapData:
        data = [float(x) for x in cellData.split(":")]
        recordList.append(data)
    return recordList 

def createRecordMap(dataLabels, recordList):
    dataDict = {label:[] for label in dataLabels}
    for recordDatum in recordList:
        for i in range(len(dataLabels)):
            dataDict[dataLabels[i]].append(recordDatum[i])
    return dataDict

def createImage(rowData, filename):
    mapDims = tuple(map(int, rowData[0].split('x')))
    mapData = rowData[1:]

    dataLabels = [
            'CellRow',
            'CellCol',
            'CellSize',
            'IndividualId',
            #'WinCount',
            'Fitness',
            'Feature1',
            'Feature2',
        ]
    recordList = createRecordList(mapData)
    dataDict = createRecordMap(dataLabels, recordList)
  
    # Add the averages of the observed features
    indexPairs = [(x,y) for x,y in product(range(mapDims[0]), range(mapDims[1]))]
    dataLabels += [feature1Label, feature2Label] 
    newRecordList = []
    for recordDatum in recordList:
        cellRow = int(recordDatum[0])
        cellCol = int(recordDatum[1])
        #px = (cellRow+0.5) / mapDims[0]
        #py = (cellCol+0.5) / mapDims[1]
        px = (cellRow) / mapDims[0]
        py = (cellCol) / mapDims[1]
        f1value = round(px*(feature1Range[1]-feature1Range[0])+feature1Range[0], feature1Precision) 
        f2value = round(py*(feature2Range[1]-feature2Range[0])+feature2Range[0], feature2Precision)
        #featurePair = [f1value, f2value]
        featurePair = [cellRow, cellCol]
        recordDatum[4] = normalize(float(recordDatum[4]))
        #recordDatum[4] = float(recordDatum[4]) / 2.0
        print(cellRow, cellCol, recordDatum[4])
        newRecordList.append(recordDatum+featurePair)
        indexPairs.remove((cellRow,cellCol))
    # Put in the blank cells
    for x,y in indexPairs:
        #px = (x+0.5) / mapDims[0]
        #py = (y+0.5) / mapDims[1]
        #newRecordList.append([x,y,0,0,math.nan,0,0,0,x,y])
        newRecordList.append([x,y,0,0,math.nan,0,0,x,y])
    dataDict = createRecordMap(dataLabels, newRecordList)
    recordFrame = pd.DataFrame(dataDict)

    # Write the map for the cell fitness
    fitnessMap = recordFrame.pivot(index=feature2Label, columns=feature1Label, values='Fitness')
    fitnessMap.sort_index(level=1, ascending=False, inplace=True)
    #print(fitnessMap)
    with sns.axes_style("white"):
        numTicks = 7 #11
        numTicksX = mapDims[0] // numTicks + 1
        numTicksY = mapDims[1] // numTicks + 1
        plt.figure(figsize=(3,3))
        g = sns.heatmap(fitnessMap, annot=False, fmt=".0f",
                xticklabels=numTicksX, 
                yticklabels=numTicksY,
                vmin=0,
                vmax=100)
                #vmin=np.nanmin(fitnessMap),
                #vmax=np.nanmax(fitnessMap))
        fig = g.get_figure()
        matplotlib.rcParams.update({'font.size': 12})
        plt.axis('off')
        g.set(title=image_title)
        g.set(xticks=[])
        g.set(yticks=[])
        plt.tight_layout()
        fig.savefig(filename)
    plt.close('all')

def createImages(stepSize, rows, filenameTemplate):
    for endInterval in range(stepSize, len(rows), stepSize):
        print('Generating : {}'.format(endInterval))
        filename = filenameTemplate.format(endInterval)
        createImage(rows[endInterval], filename)

def createMovie(folderPath, filename):
    globStr = os.path.join(folderPath, '*.png')
    imageFiles = sorted(glob.glob(globStr))

    # Grab the dimensions of the image
    img = cv2.imread(imageFiles[0])
    imageDims = img.shape[:2][::-1]

    # Create a video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frameRate = 30
    video = cv2.VideoWriter(filename, fourcc, frameRate, imageDims)

    for imgFilename in imageFiles:
        img = cv2.imread(imgFilename)
        video.write(img)

    video.release()


def generateAll(folderPath):
    print('Generating: ', folderPath)
    logPath = os.path.join(folderPath, logFilename)
    with open(logPath, 'r') as csvfile:
        # Read all the data from the csv file
        allRows = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
    
        # First create the final image we need
        imageFilename = 'fitnessMap.png'
        createImage(allRows[-1], imageFilename)

        # Clear out the previous images
        tmpImageFolder = 'images/'
        for curFile in glob.glob(tmpImageFolder+'*'):
            os.remove(curFile)

        template = tmpImageFolder+'grid_{:05d}.png'
        createImages(4, allRows[1:], template)
        movieFilename = 'fitness.avi'
        createMovie(tmpImageFolder, movieFilename) 

for folderPath in logPaths:
    generateAll(folderPath)
