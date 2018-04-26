import numpy as np 
import pdb
from geopy.distance import great_circle

# def get_accuracy(test_labels,output_labels):
#     acc = [ 0 for _ in range(nLabels) ]
#     for l in range(nLabels):
#         nCorrect = sum([ l == p for (p, t) in zip(pred, labels) if t == l ])
#         n = sum([ l == t for t in labels ])
#         acc[l] = float(nCorrect)/n
#     return acc


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