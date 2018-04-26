import numpy as np 
import pdb
from geopy.distance import great_circle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

sys.path.append('../sampler')  
import location_sampler


# TODO: label_names order different from the actual 0-49 order!! REDO stuff. 

def get_accuracy(test_labels,output,state_centers,N):
	output_labels = np.argsort(output,axis =1)[:,-N:]
	acc = 0
	nLabels = 50
	acc_per_state = np.zeros([nLabels])
	
	# % accuracy
	for i in range(0,output_labels.shape[0]):
		for j in range(0,output_labels.shape[1]):
			acc += (int(output_labels[i,j]) == test_labels[i])
			acc_per_state[test_labels[i]] += (int(output_labels[i,j]) == test_labels[i])

	# distance metric 
	dist = 0 
	for i in range(0,output_labels.shape[0]):
		match = False
		for j in range(0,output_labels.shape[1]):
			if (int(output_labels[i,j]) == test_labels[i]):
				match = True
				break

		if match == False:
			dist += great_circle(state_centers[output_labels[i,-1],:], state_centers[test_labels[i]]).kilometers

	# accu = [ 0 for _ in range(nLabels) ]
	# for l in range(nLabels):
	# 	nCorrect = sum([ l == p for (p, t) in zip(output_labels, test_labels) if t == l ])
	# 	n = sum([ l == t for t in test_labels ])
	# 	accu[l] = float(nCorrect)/n

	for k in range(0,50):
		acc_per_state[k] = acc_per_state[k]/np.sum(test_labels == k)
	return acc/output_labels.shape[0], acc_per_state, dist/output_labels.shape[0]

# def get_top_3_accuracy(test_labels,output_labels):
# def get_distinctness_score():


def show_image_and_map(test_labels,label_names,test_images,output,test_image_path):
	states_file = '../sampler/states.xml'
	borders= location_sampler.get_borders(states_file)
	labels= location_sampler.get_labels(states_file)
	#location_sampler.plot_map(borders)
	nLabels = len(borders)

	output_labels = np.argsort(output,axis =1)[:,-5:]
	acc = 0

	for i in range(0,output_labels.shape[0]):
		# plot the image first 
		im_path = test_image_path + '/' + label_names[test_labels[i]] + '/' + test_images[i].rsplit('/', 1)[-1]
		print(im_path)
		img = mpimg.imread(im_path)
		plt.figure(0)
		imgplot = plt.imshow(img)
		# plt.show()

		plt.figure(1)
		# generate border map of USA
		for k in range(0,nLabels):
			plt.plot(borders[k][:,0], borders[k][:,1], 'r-')
			plt.hold(True)

			# due to mismatch im plotting the wrong state! 
			idx = labels.index(label_names[test_labels[i]])
			idx1 = labels.index(label_names[int(output_labels[i,0])])
			idx2 = labels.index(label_names[int(output_labels[i,1])])
			idx3 = labels.index(label_names[int(output_labels[i,2])])
			idx4 = labels.index(label_names[int(output_labels[i,3])])
			idx5 = labels.index(label_names[int(output_labels[i,4])])

			plt.plot(borders[idx][:,0], borders[idx][:,1], 'c-',linewidth=5.0)
			
			plt.fill(borders[idx1][:,0], borders[idx1][:,1], 'g-')
			plt.fill(borders[idx2][:,0], borders[idx2][:,1], 'y-')
			plt.fill(borders[idx3][:,0], borders[idx3][:,1], 'm-')
			plt.fill(borders[idx4][:,0], borders[idx4][:,1], 'r-')
			plt.fill(borders[idx5][:,0], borders[idx5][:,1], 'k-')
		plt.show(0)
		input("Press Enter to continue...")
		plt.hold(False)

