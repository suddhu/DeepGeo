import numpy as np 
import pdb
from geopy.distance import great_circle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

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

			# plot for states

			# ground truth 
			idx = labels.index(label_names[test_labels[i]])
			plt.plot(borders[idx][:,0], borders[idx][:,1], 'c-',linewidth=5.0)

			idx1 = labels.index(label_names[int(output_labels[i,0])])
			idx2 = labels.index(label_names[int(output_labels[i,1])])
			idx3 = labels.index(label_names[int(output_labels[i,2])])
			idx4 = labels.index(label_names[int(output_labels[i,3])])
			idx5 = labels.index(label_names[int(output_labels[i,4])])

			plt.fill(borders[idx1][:,0], borders[idx1][:,1], color='#66ff33')
			plt.fill(borders[idx2][:,0], borders[idx2][:,1], color='#ffff00')
			plt.fill(borders[idx3][:,0], borders[idx3][:,1], color='#ff9900')
			plt.fill(borders[idx4][:,0], borders[idx4][:,1], color='#cc3300')
			plt.fill(borders[idx5][:,0], borders[idx5][:,1], color='#4d1300')

			# draw legend
			colors = ['#66ff33','#ffff00','#ff9900','#cc3300','#4d1300','#00ffff']
			LABELS = ['#1 prediction','#2 prediction','#3 prediction','#4 prediction','#5 prediction','Ground Truth']
			patches = [
			    mpatches.Patch(color=color, label=label)
			    for label, color in zip(LABELS, colors)]
			plt.legend(patches, LABELS, loc=1, frameon=False)

		plt.show(0)
		input("Press Enter to continue...")
		plt.hold(False)


def show_image_and_map_prob(test_labels,label_names,test_images,output,test_image_path):
	states_file = '../sampler/states.xml'
	borders= location_sampler.get_borders(states_file)
	labels= location_sampler.get_labels(states_file)
	#location_sampler.plot_map(borders)
	nLabels = len(borders)

	# 256 value colormap?
	# cmap = plt.cm.jet
	cmap = plt.get_cmap('YlOrRd')

	cmaplist = [cmap(i) for i in range(cmap.N)]

	output_vals = np.sort(output,axis =1)[:,:]
	output_labels = np.argsort(output,axis =1)[:,:]
	# pdb.set_trace()
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
			# plot for states

			# ground truth 
			idx = labels.index(label_names[test_labels[i]])
			plt.plot(borders[idx][:,0], borders[idx][:,1], 'c-',linewidth=5.0)

			for l in range(0,nLabels):
			# shade each based on probability
				idx_shade = labels.index(label_names[int(output_labels[i,l])])
				color_to_shade = int(round((output_vals[i,l]/np.max(output_vals[i,:]))*(len(cmaplist) - 1)))
				#print(color_to_shade)
				#pdb.set_trace()
				plt.fill(borders[idx_shade][:,0], borders[idx_shade][:,1], color=cmaplist[color_to_shade])

			# draw legend
			# colors = ['#66ff33','#ffff00','#ff9900','#cc3300','#4d1300','#00ffff']
			# LABELS = ['#1 prediction','#2 prediction','#3 prediction','#4 prediction','#5 prediction','Ground Truth']
			# patches = [
			#     mpatches.Patch(color=color, label=label)
			#     for label, color in zip(LABELS, colors)]
			# plt.legend(patches, LABELS, loc=1, frameon=False)
		plt.colorbar()
		plt.show(0)
		input("Press Enter to continue...")
		plt.hold(False)

def plot_graphs(acc_array):
	x = [1, 2, 3, 4]
	my_xticks = ['Top-1','Top-2','Top-3','Top-5']
	plt.xticks(x, my_xticks)

	plt.plot(x, acc_array, 'r-',label='Single Image')
	plt.yticks(np.arange(0, 1, 0.1))

	# Create the formatter using the function to_percent. This multiplies all the
	# default labels by 100, making them all percentages
	formatter = FuncFormatter(to_percent)

	# Set the formatter
	plt.gca().yaxis.set_major_formatter(formatter)

	plt.axis([0, 5, 0, 1])

	plt.hold(True)
	plt.axhline(0.02, color='k', linestyle='dashed', linewidth=1,label='Random Chance')
	plt.legend()
	plt.show()
	input("Press Enter to continue...")

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(int(100 * y))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
