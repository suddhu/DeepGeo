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
pickle_file2 = "outputs/early-int-human-data.pickle"
pickle_file3 = "outputs/med-int-human-data.pickle"
pickle_file4 = "outputs/late-int-human-data.pickle"

state_center_file = "../sampler/state_center.npy"
test_image_path = "/home/monty/Pictures/DeepGeo/human_trials"

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
	# label_names = np.asarray(T["label_names"])
	# test_labels = np.asarray(T["test_labels"])
	# test_images = np.asarray(T["test_files"])
	# label_names = np.asarray(T["label_names"])

	# -------------------------------------------------------------------------------------------------------
	# create dict storing which human corresponds to each sample
	infoFileName = test_image_path + '/' + '/info.txt'
	stateData = genfromtxt(infoFileName, delimiter=',',dtype=str)
	nLines = stateData.shape[0]
	gtLabelsDict = {}
	humanLabels = np.array([29,18,46,37,20,11,10,33,10,27,49,38,47,16,30,22,7,13,35,27,3,5,29,30,28,20,36,13,39,38,3,27,6,22,5,6,0,48,8,17,29,36,38,41,29,14,8,15,14,38],dtype=int)
	gtLabels = np.array([0,4,2,4,26,36,27,8,46,25,13,35,35,22,44,5,22,38,14,47,28,17,11,1,28,1,36,11,49,38,3,19,17,49,5,35,39,25,2,46,11,6,6,5,30,17,3,42,16,38],dtype=int)
	gtLabels1 = gtLabels
	gtLabels = np.repeat(gtLabels,8)
	subject = np.array([0,1,2,3,4],dtype=int)
	subject = np.repeat(subject,80)
	subjectDict = {}
	print('N:',nLines)
	for i in range(0,nLines,2):
	# print(i,coords[0])
		label = re.sub(r'.*Humans/', 'Humans/', stateData[i])
		label = label.replace('//','/')
		coords = [float(n) for n in stateData[i+1].split(' ')]
		gtLabelsDict[label] = gtLabels[i]
		subjectDict[label] = subject[i]
		# print(label,gtLabelsDict[label],subjectDict[label])

	# -------------------------------------------------------------------------------------------------------
	# display network performance for each round
	nCorrect = [0,0,0,0,0]
	for i in range(0,test_images.shape[0]):
		# print(i)
		name = str(test_images[i][0])
		name = re.sub(r'.*human_trials/', 'human_trials/', name)
		name = name.replace('human_trials','Humans')
		name = name[0:-1];
		if (subjectDict[name]==0):
			print('Name: ',name,'Guess: ',np.argmax(output[i]),'True: ',gtLabelsDict[name],'Subject: ',subjectDict[name])
		# print(np.argmax(output[i]))
		if (np.argmax(output[i])==gtLabelsDict[name]):
			nCorrect[subjectDict[name]]+=1
	print(nCorrect)

	borders= location_sampler.get_borders(states_file)
	
	# -------------------------------------------------------------------------------------------------------
	# Generate figure comparing performance on map
	#location_sampler.plot_map(borders)
	nLabels = len(borders)
	print(nLabels)
	plt.figure(1)
	# generate border map of USA
	for k in range(0,nLabels):
		borders[k] = borders[k].astype(np.float)
		plt.plot(borders[k][:,0], borders[k][:,1], 'k-', linewidth=1.0)
		plt.hold(True)

	#data from game 1 for generating figures
	iRound = 9
	#output labels different order from state borders
	reorder = np.array([11,33,49,13,46,42,43,32,23,30,41,27,10,2,3,6,5,12,45,24,48,38,15,20,34,22,39,9,21,28,37,0,16,44,1,4,25,47,31,40,7,17,26,19,18,29,36,8,14,35],dtype=int)
	humanGuess = np.array([10,33,27,46,18,37,10,20,29,11],dtype=int)
	netGuess = np.array([41,41,26,2,4,4,27,27,0,36],dtype=int)
	correct = np.array([46,8,25,2,4,4,27,26,0,36],dtype=int)
	plt.fill(borders[reorder[correct[iRound]]][:,0], borders[reorder[correct[iRound]]][:,1], color='#66ff33')
	plt.plot(borders[reorder[netGuess[iRound]]][:,0], borders[reorder[netGuess[iRound]]][:,1], 'b-', linewidth=2.0)
	plt.plot(borders[reorder[humanGuess[iRound]]][:,0], borders[reorder[humanGuess[iRound]]][:,1], 'r-', linewidth=2.0)
	plt.show()
	plt.axis('equal')
	plt.xlabel('lng')
	plt.ylabel('lat')


if __name__ == '__main__':
    main()