import time
import itertools
import googlemaps
import urllib
import numpy as np
from numpy import genfromtxt
import pdb
import os, os.path
import cv2
import numpy as np
from matplotlib import pyplot as plt
import streetview_tools

import sys  
sys.path.append('../sampler')  
import location_sampler

states_file = '../sampler/states.xml'
density_file = '../sampler/density2.5MinCutoff.txt'
fail_image = cv2.imread('../fail.jpg',1)
fail_image2 = cv2.imread('../fail2.jpg',1)


images_dir = '/home/monty/Pictures/DeepGeo/human_trials'	#X1
#images_dir = '/home/suddhu/Documents/courses/10701/project/images/'	#P51

gmaps_API = googlemaps.Client(key='***REMOVED***')
#geocoder_API = '***REMOVED***'

#streetview_API_key = '***REMOVED***' # deepgeo701@gmail.com resnet701
#streetview_API_key = '***REMOVED***' # X1 carbon suddhus@gmail.com
#streetview_API_key = '***REMOVED***' # P51 sudhars1?
streetview_API_key = '***REMOVED***'
#'***REMOVED***' #monty's wallet 
# '***REMOVED***' suddhu's wallet 

def reverse_geocode(coord):
    '''
    INPUT:  (1) tuple: latitude and longitude coordinates, in degrees
    OUTPUT: (1) string: full geocoded address
    '''
    result = gmaps_API.reverse_geocode(coord)
    return result[0]['formatted_address']


def get_elev(coord):
    '''
    INPUT:  (1) tuple: latitude and longitude coordinates, in degrees
    OUTPUT: (1) float: elevation of the lat/lng point in meters
    '''
    elev = gmaps_API.elevation((coord[0], coord[1]))[0]['elevation']
    return elev

def main():
    x = 50
    y = 2500
    heading = [0,90,180,270]
    borders= location_sampler.get_borders(states_file)
    nLabels = len(borders)
    # generate border map of USA
    for i in range(0,nLabels):
        plt.plot(borders[i][:,0], borders[i][:,1], 'r-')
    plt.show(0)
    plt.hold(True)

    coords = genfromtxt(images_dir + '/coords.txt', delimiter=',',dtype=np.float64)

    dir = images_dir + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    f = open( dir + "info.txt","a")

    for i in range(0,50):
        panoids = []
        
        lat = coords[i,0]
        lng = coords[i,1]
        #plot lat,lng on map
        # plt.plot(lng, lat, 'b.', markersize=2)
        # plt.pause(0.05)

        panoids = []
        count = 0
        while not(panoids):
            panoids = streetview_tools.panoids(lat=lat, lon=lng)
            sys.stdout.write('.')
            count+=1
            print count

        print  'index: ', i, lat,lng

        # plot lat,lng on map
        # plt.plot(lng, lat, 'b.', markersize=2)
        # plt.pause(0.05)

        for directions in heading:

            filename = streetview_tools.api_download(panoids[0]['panoid'], directions, dir, streetview_API_key, width=256, height=256,fov=90, pitch=0, extension='jpg', year=panoids[0]['year'])
            try:
                A = cv2.imread(filename,1)
                difference = cv2.subtract(A, fail_image)
                result1 = not np.any(difference)

                difference = cv2.subtract(A, fail_image2)
                result2 = not np.any(difference)

                if result1 is True:
                    print "Street View limit has been reached!"
                    # Todo - delete image 
                    os.remove(filename)
                    f.close()
                    sys.exit()
                elif result2 is True:
                    print "Street View Signature Error!"
                    # Todo - delete image 
                    os.remove(filename)
                    f.close()
                    sys.exit()

                cv2.imshow('current image',A)
                cv2.waitKey(1)
                f.write("%s \r %f %f \n" % ((filename), (lat), (lng)))
            except cv2.error:
                print "OpenCV error: moving along..."
    f.close()
if __name__ == '__main__':
    main()

