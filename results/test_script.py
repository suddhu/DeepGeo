import numpy as np
import pdb 
import pickle
import test_metrics

# output.npy has the prob. of each of the states (10K * 50 matrix). We can use this to find the dominant state of each 
output_file = 'output.npy'
pickle_file = "single_image_test_files.pickle"
state_center_file = "../sampler/state_center.npy"

# the pickle file has the test labels (we need) and the test file names (dont need). Label names has which state each label pertains to. 
# so we should be comparing the max(output label) with the test label for the corresponding image number. We can do a top 2/3 metric as well. 

def main():
	# output probabilities (10k * 50)
	output = np.load(output_file)
	print(len(output))

	# test data - label_names size (50), test_labels size (100013), test_files size (100013)
	pickle_in = open(pickle_file,"rb")
	T = pickle.load(pickle_in)

	state_centers = np.load(state_center_file)
	# keys in pickle dict
	for key in T:
		print("key: %s size: %d" % (key, len(T[key])))

	test_labels = np.asarray(T["test_labels"])
	
	# some accuracy metrics 
	#NOTE: per state accuracy seems to be bust as well (no explainable trends?)
	# NOTE: distance seems to be a pointless metric
	acc_1,acc_per_state_1,dist_1 = test_metrics.get_accuracy(test_labels,output, state_centers,1)
	acc_2,acc_per_state_2,dist_2 = test_metrics.get_accuracy(test_labels,output,state_centers,2)
	acc_3,acc_per_state_3,dist_3 = test_metrics.get_accuracy(test_labels,output,state_centers,3)
	acc_5,acc_per_state_5,dist_5 = test_metrics.get_accuracy(test_labels,output,state_centers,5)

	print(acc_1,acc_2,acc_3,acc_5)
	print(dist_1,dist_2,dist_3,dist_5)

    # get_distinctness_score()
	pdb.set_trace()

if __name__ == '__main__':
    main()