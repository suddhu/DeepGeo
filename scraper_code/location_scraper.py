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


images_dir = '/home/monty/Pictures/DeepGeo/'


streetview_API_key = '***REMOVED***'


# def get_coords(panoid):
# 	return (lat,lng)



def main():


	
	#get all panoids from filenames
	#pass panoids to 
	panoid = 'AN3B_QgbO_zjOYgjfDHfIQ'

	lat,lng = streetview_tools.coord_info(panoid,streetview_API_key)
	print lat,lng


if __name__ == '__main__':
    main()

