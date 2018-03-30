import time
import itertools
import googlemaps
import urllib

import streetview

'''
This scraper utilizes the website instantstreetview.com to find
valid latitude/longitude coordinates for which there is streetview data.
I played around with querying the streetview API with locations from
road shapefiles in Colorado, but the miss rate was simply too high: it wasn't
worth having 50% of the downloaded images be images that said "Sorry, we have
no imagery here." It's possible to query the streetview API directly in
javascript, but the Python interface doesn't allow this. Utilizing the
middleman of instantstreetview made things easier at the scale of my project,
but this technique wouldn't work on a larger scale.
'''

gmaps_API = googlemaps.Client(key='***REMOVED***')
geocoder_API = '***REMOVED***'
streetview_API_key = '***REMOVED***'
dir = '/home/suddhu/Documents/courses/10701/project/images/'

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
                   streetview?size=640x400&location={},{}
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

    coords = [-33.85693857571269,151.2144895142714]; 
    panoids = streetview.panoids(lat=coords[0], lon=coords[1])
    heading = [0,90,180,270]
    for directions in heading:
        streetview.api_download(panoids[0]['panoid'], directions, dir, streetview_API_key, width=640, height=640,fov=90, pitch=0, extension='jpg', year=panoids[0]['year'])

    print reverse_geocode(coords)
    print get_elev(coords)

if __name__ == '__main__':
    main()

