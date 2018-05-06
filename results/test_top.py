import numpy as np
import pdb 
import pickle
import test_metrics
import os
import re
from numpy import genfromtxt
import matplotlib
import matplotlib.pyplot as plt
import sys  
sys.path.append('../sampler')  
import location_sampler

states_file = '../sampler/states.xml'


# output.npy has the prob. of each of the states (10K * 50 matrix). We can use this to find the dominant state of each 
output_file = 'outputs/output.npy'

pickle_file1 = "outputs/single_image_test_files.pickle"
pickle_file2 = "outputs/early-int-data.pickle"
pickle_file3 = "outputs/med-int-data.pickle"
pickle_file4 = "outputs/late-int-data.pickle"

state_center_file = "../sampler/state_center.npy"
test_image_path = "/home/monty/Pictures/DeepGeo/test_data"

show_image_and_map_plot = 1
# the pickle file has the test labels (we need) and the test file names (dont need). Label names has which state each label pertains to. 
# so we should be comparing the max(output label) with the test label for the corresponding image number. We can do a top 2/3 metric as well. 


def main():

	# -------------------------------------------------------------------------------------------------------
	# load stuff
	# output probabilities (10k * 50)
	# output = np.load(output_file)

	# test data - label_names size (50), test_labels size (100013), test_files size (100013)
	pickle_1 = open(pickle_file1,"rb")
	T1 = pickle.load(pickle_1)
	pickle_in = open(pickle_file3,"rb")
	T = pickle.load(pickle_in)

	state_centers = np.load(state_center_file)
	# keys in pickle dict
	for key in T1:
		print("key: %s size: %d" % (key, len(T1[key])))
	for key in T:
		print("key: %s size: %d" % (key, len(T[key])))

	test_labels = np.asarray(T["labels"])
	test_images = np.asarray(T["filenames"])
	output      = np.asarray(T["output"])
	label_names = np.asarray(T1["label_names"])

	# -------------------------------------------------------------------------------------------------------
	# create dictionary: key = filename, value = coordinates
	locations = {}
	root=test_image_path
	dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
	# for i in range(0,1):
	for i in range(0,len(dirlist)):
		infoFileName = test_image_path + '/' + dirlist[i] + '/info.txt'
		# f = open(infoFileName,'r')
		stateData = genfromtxt(infoFileName, delimiter=',',dtype=str)
		nLines = stateData.shape[0]
		for i in range(0,nLines,2):
		# print(i,coords[0])
			label = re.sub(r'.*test_data/', 'test_data/', stateData[i])
			label = label.replace('//','/')
			coords = [float(n) for n in stateData[i+1].split(' ')]
			locations[label] = np.array([coords])
	# for key,value in locations.items():
	# 	print(key,value)
	# 	print(type(key))

	#test images
	# print(np.argmax(output[0]))

	# -------------------------------------------------------------------------------------------------------
	# plot if correct/incorrect
	N = 1
	correctCoords   = np.empty((0,2))
	incorrectCoords = np.empty((0,2))
	stateCorrect = np.zeros((50,))
	stateCount = np.zeros((50,))
	missing = 0
	for i in range(0,test_images.shape[0]):
		name = str(test_images[i][0])
		name = re.sub(r'.*test_data/', 'test_data/', name)
		name = name[0:-1];
		try:
			coords = locations[name]
			# if (test_labels[i] == np.argmax(output[i])):
			stateCount[test_labels[i]] +=1
			if (test_labels[i] in np.argsort(output[i])[-N:]):
				correctCoords = np.concatenate((correctCoords,locations[name]),axis=0)
				stateCorrect[test_labels[i]] +=1
			else:
				incorrectCoords = np.concatenate((incorrectCoords,locations[name]),axis=0)
		except KeyError:
			# print('Key does not exist')
			missing+=1
	nCorrect = np.float(correctCoords.shape[0])
	nIncorrect = np.float(incorrectCoords.shape[0])
	print('correct: ',nCorrect)
	print('incorrect: ',nIncorrect)
	print('mislabelled: ',missing)
	print('accuracy: ', nCorrect/(nCorrect+nIncorrect))


if __name__ == '__main__':
    main()