import time
import itertools
import googlemaps
import urllib
import numpy as np
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


images_dir = '/home/suddhu/Pictures/deepgeo/images/'	#X1
#images_dir = '/home/suddhu/Documents/courses/10701/project/images/'	#P51

#gmaps_API = googlemaps.Client(key='***REMOVED***')
#geocoder_API = '***REMOVED***'
#streetview_API_key = '***REMOVED***' # X1 carbon suddhus@gmail.com
#streetview_API_key = '***REMOVED***' # P51 sudhars1?
streetview_API_key = '***REMOVED***' #sudharshan.nitt
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
    #[x,y,z] = np.shape(state_points)
    # Alaska - 0, Wyoming - 49
    start = int(sys.argv[1])
    finish = int(sys.argv[2])

    print "State " + str(start) + " to " + str(finish)
    #x = 50
    y = 2500
    heading = [0,90,180,270]

    borders= location_sampler.get_borders(states_file)
    labels= location_sampler.get_labels(states_file)

    density = location_sampler.load_density(density_file)

    #location_sampler.plot_map(borders)
    nLabels = len(borders)

    # generate border map of USA
    for i in range(0,nLabels):
        plt.plot(borders[i][:,0], borders[i][:,1], 'r-')
    plt.show(0)
    plt.hold(True)
    # subset = [4,10,26,27,37]
    # y = [2500, 1337, 2500, 2500,2500]

#    for states in range(1,x):
    for states in range(start,finish): 

        dir = images_dir + str(labels[states]) + '/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        f = open( dir + "info.txt","a")

        # number of images already in the path 
        images_in_folder = (len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]) - 1)/4
        images_needed = y - images_in_folder

        print  str(labels[states]) + " needs " + str(images_needed) + " more images!"

        if images_needed < 0:
            continue    # more than 10K, moving on...

        for vals in range(0,images_needed):
            panoids = []
            
            while not(panoids):
                state_points = location_sampler.get_points_in_states(borders,1,density) # long, lat
                lat = state_points[states][0][1]
                lng = state_points[states][0][0]
                # lat=-33.856f93857571269 lng=151.2144895142714
                panoids = streetview_tools.panoids(lat=lat, lon=lng)
                sys.stdout.write('.')

            print  str(labels[states])  + " " + str(vals + images_in_folder)

            # plot lat,lng on map
            plt.plot(lng, lat, 'b.', markersize=2)
            plt.pause(0.05)

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

