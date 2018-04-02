import time
import itertools
import googlemaps
import urllib
import numpy as np
import pdb
import os


import streetview_tools

import sys  
sys.path.append('../sampler')  
import location_sampler

states_file = '../sampler/states.xml'
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
gmaps_API = googlemaps.Client(key='***REMOVED***')
geocoder_API = '***REMOVED***'
streetview_API_key = '***REMOVED***' #monty's wallet 
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
    '''
    INPUT:  None
    OUTPUT: None

    Do everything. Go to the website, search and zoom in on Colorado, and get
    random valid coordinates for street view locations in Colorado.
    Reverse geocode the valid coordinates to get the elevation and full
    address of the location. Get the date the image was taken.
    Save everything to a .csv.
    '''

    #[x,y,z] = np.shape(state_points)
    x = 50
    y = 2500
    heading = [0,90,180,270]

    borders= location_sampler.get_borders(states_file)
    labels= location_sampler.get_labels(states_file)

    subset = [4,10,26,27,37]
    #coords = [-33.85693857571269,151.2144895142714]; 
#    for states in range(1,x):
    for states in subset: 
        dir = '/home/suddhu/Documents/courses/10701/project/images/' + str(labels[states]) + '/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        f = open( dir + "info.txt","w+")


        for vals in range(0,y):
            panoids = []
            while not(panoids):
                state_points = location_sampler.get_points_in_states(borders) # long, lat
                lat = state_points[states][0][1]
                lng = state_points[states][0][0]
                # lat=-33.85693857571269 lng=151.2144895142714
                panoids = streetview_tools.panoids(lat=lat, lon=lng)

            for directions in heading:
                filename = streetview_tools.api_download(panoids[0]['panoid'], directions, dir, streetview_API_key, width=256, height=256,fov=90, pitch=0, extension='jpg', year=panoids[0]['year'])
                f.write("%s \r %f %f \n" % ((filename), (lat), (lng)))
            print reverse_geocode([lat,lng])
            #print get_elev([lat,lng])
        f.close()
if __name__ == '__main__':
    main()

