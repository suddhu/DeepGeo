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
'''
This scraper utilizes the website instantstreetview.com to find
valid latitude/longitude coordinates for which there is streetview_tools data.
I played around with querying the streetview_tools API with locations from
road shapefiles in Colorado, but the miss rate was simply too high: it wasn't
worth having 50% of the downloaded images be images that said "Sorry, we have
no imagery here." It's possible to query the streetview_tools API directly in
javascript, but the Python interface doesn't allow this. Utilizing the
middleman of instantstreetview made things easier at the scale of my project,
but this technique wouldn't work on a larger scale.
'''

#gmaps_API = googlemaps.Client(key='***REMOVED***')
#geocoder_API = '***REMOVED***'
streetview_API_key = '***REMOVED***'
#'***REMOVED***' #monty's wallet 
# '***REMOVED***' suddhu's wallet 

def save_image(coord, heading, pitch=5, fov=90, loc='../images/'):
    '''
    INPUT:  (1) tuple: latitude and longitude coordinates, in degrees
            (2) integer: 0, 360 = N, 90= E, 180 = S, 270 = W
            (3) integer: -90 < pitch < 90 (degrees). 5 is a good middleground
            (4) integer: 20 < fov < 120. 90 is a natural middleground.
            (5) string: folder name to save images to
    OUTPUT: None

    This function will save google street view images facing N, E, S, and W
    for a given coordinate pair to 'loc'
    '''
    if heading == 0 or heading == 360:
        sufx = 'N'
    elif heading == 90:
        sufx = 'E'
    elif heading == 180:
        sufx = 'S'
    elif heading == 270:
        sufx = 'W'
    web_address = ('''https://maps.googleapis.com/maps/api/
                   streetview_tools?size=640x400&location={},{}
                   &fov={}&heading={}&pitch={}&key={}'''.format(
                   coord[0], coord[1], fov,
                   heading, pitch, streetview_API_key))

    filename = ('''{}/lat_{},long_{}_.png'''.format(
                loc, str(coord[0])[:8], sufx,))
    urllib.urlretrieve(web_address, filename=filename)


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
        dir = '/home/suddhu/Documents/courses/10701/project/images/' + str(labels[states]) + '/'
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
                    result = not np.any(difference)

                    if result is True:
                        print "Street View limit has been reached!"
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

