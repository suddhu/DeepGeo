import googlemaps
import numpy as np 
import pdb

gmaps = googlemaps.Client(key='***REMOVED***')
place_id = "ChIJnQrgk4u6EmsRVqYSfnjhaOk"
# Geocoding an address
geocode_result = gmaps.reverse_geocode(place_id)

pdb.set_trace()

geocode_result[0]['geometry']['location']['lat']
geocode_result[0]['geometry']['location']['lng']

print len(geocode_result)
