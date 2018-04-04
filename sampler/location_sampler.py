import numpy as np
import sys
import pdb

from PIL import Image
from numpy import genfromtxt
from matplotlib import pyplot as plt
from matplotlib import figure as fig
import xml.etree.ElementTree as ET
import matplotlib
import matplotlib.path as mplPath

def genPoints(nPoints,xMin,xMax,yMin,yMax):
    x = np.random.uniform(xMin,xMax,(nPoints,1))
    y = np.random.uniform(yMin,yMax,(nPoints,1))
    return (x,y)

def genPointsWeighted(nPoints,xMin,xMax,yMin,yMax,stateDensity):
    x = np.random.uniform(xMin,xMax,(nPoints,1))
    y = np.random.uniform(yMin,yMax,(nPoints,1))
    return (x,y)


def pointInPolygon(x,y,points):
	"Determines if pt inside polygon"
	outPath = mplPath.Path(points)
	return outPath.contains_point((x, y))


def get_borders(states_file):
    #init
    borders = []

    # get state borders  
    tree = ET.parse(states_file)
    root = tree.getroot()
    count = 0
    for child in root:
        #print child.attrib['name'],count
        count+=1
        stateBorder = np.empty((0,2),dtype=np.float64)
        for point in child:
            stateBorder = np.append(stateBorder,np.array([[point.attrib['lng'], point.attrib['lat']]]),axis=0)
        borders.append(stateBorder)

    return borders

def get_labels(states_file):
    #init
    labels = []

    # get state borders  
    tree = ET.parse(states_file)
    root = tree.getroot()
    count = 0
    for child in root:
        labels.append(child.attrib['name'])
        #print child.attrib['name'],count

    return labels

def get_points_in_states(borders,densityFlag,plotFlag):
    
    points = []
    nSamples = 1
    
    # no states
    nLabels = len(borders)
    
    # plot borders
    if plotFlag:
        for i in range(0,nLabels):
        plt.plot(borders[i][:,0], borders[i][:,1], 'ro-')
    
    # load population density
    if densityFlag:
        density = genfromtxt('density2.5Min.txt', delimiter=',')
#    print density.shape

        
    # sample points
    for i in range(0,nLabels):
        #print i
        nValidPoints = 0
        
        xMin = min(map(float,borders[i][:,0]))
        xMax = max(map(float,borders[i][:,0]))
        yMin = min(map(float,borders[i][:,1]))
        yMax = max(map(float,borders[i][:,1]))
        
        # load pop density for this rectangle
        if densityFlag:
#        #convert lng,lat to matrix coords
#        #30Sec
##        colMin = round((xMin + 180)*120)
##        colMax = round((xMax + 180)*120)
##        rowMin = round((85 - yMin)*120)
##        rowMax = round((85 - yMax)*120)
#        #2.5Min
            colMin = int(round((xMin + 180)*24))
            colMax = int(round((xMax + 180)*24))
            rowMin = int(round((85 - yMin)*24))
            rowMax = int(round((85 - yMax)*24))
            stateDensity = density[rowMax:rowMin,colMin:colMax]
            if (np.max(stateDensity) != 0):
                stateDensity = stateDensity/np.max(stateDensity)
            else:
                raise ValueError
                if plotFlag:
                    img = Image.fromarray(stateDensity*255)
                    img.show()
                
        xSamples =  np.empty([0,1],dtype=np.float64)
        ySamples =  np.empty([0,1],dtype=np.float64)
        while (nValidPoints < nSamples):
            x,y = genPoints(nSamples-nValidPoints,xMin,xMax,yMin,yMax)
            # generate points based on population density
#            x,y = genPointsWeighted(nSamples-nValidPoints,xMin,xMax,yMin,yMax,stateDensity)
            validPoints = np.full((nSamples-nValidPoints,), False, dtype=bool)
            for j in range(0,nSamples-nValidPoints):
                #check if point in populated area
                if densityFlag:
                    xGrid = int(round((x[j] + 180)*24)) - colMin - 1
                    yGrid = int(round((85 - y[j])*24)) - rowMax - 1
                if pointInPolygon(x[j],y[j],borders[i]):
#                    print xGrid,yGrid,colMin,colMax,rowMax,rowMin
                    if densityFlag:
                        if (stateDensity[yGrid,xGrid] > 1e-2):
                            validPoints[j] = True
            x = x[validPoints]
            y = y[validPoints]
            xSamples = np.concatenate((xSamples,x),axis=0)
            ySamples = np.concatenate((ySamples,y),axis=0)
            nValidPoints = len(xSamples)        #store
        points.append(np.concatenate((xSamples,ySamples),axis=1))
        
    # plot points
    if plotFlag:
        for i in range(0,nLabels):
            plt.plot(points[i][:,0], points[i][:,1], 'b.')

    return points