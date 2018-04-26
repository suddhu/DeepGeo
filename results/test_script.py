import numpy as np
import pdb 
import pickle
import test_metrics

# output.npy has the prob. of each of the states (10K * 50 matrix). We can use this to find the dominant state of each 
output_file = 'output.npy'
pickle_file = "single_image_test_files.pickle"
state_center_file = "../sampler/state_center.npy"
test_image_path = "/home/suddhu/Pictures/deepgeo/test_data"

show_image_and_map_plot = 0
# the pickle file has the test labels (we need) and the test file names (dont need). Label names has which state each label pertains to. 
# so we should be comparing the max(output label) with the test label for the corresponding image number. We can do a top 2/3 metric as well. 


def main():
	# output probabilities (10k * 50)
	output = np.load(output_file)

	# test data - label_names size (50), test_labels size (100013), test_files size (100013)
	pickle_in = open(pickle_file,"rb")
	T = pickle.load(pickle_in)

	state_centers = np.load(state_center_file)
	# keys in pickle dict
	for key in T:
		print("key: %s size: %d" % (key, len(T[key])))

	test_labels = np.asarray(T["test_labels"])
	test_images = np.asarray(T["test_files"])
	label_names = np.asarray(T["label_names"])
	# some accuracy metrics 
	#NOTE: per state accuracy seems to be bust as well (no explainable trends?)
	# NOTE: distance seems to be a pointless metric
	acc_1,acc_per_state_1,dist_1 = test_metrics.get_accuracy(test_labels,output, state_centers,1)
	acc_2,acc_per_state_2,dist_2 = test_metrics.get_accuracy(test_labels,output,state_centers,2)
	acc_3,acc_per_state_3,dist_3 = test_metrics.get_accuracy(test_labels,output,state_centers,3)
	acc_5,acc_per_state_5,dist_5 = test_metrics.get_accuracy(test_labels,output,state_centers,5)
	
	if show_image_and_map_plot:
		test_metrics.show_image_and_map(test_labels,label_names,test_images,output,test_image_path)

	#print(acc_1,acc_2,acc_3,acc_5)
	#print(dist_1,dist_2,dist_3,dist_5)
	for i in range(0,50):
		print( str(label_names[i]) + ": " + str(acc_per_state_5[i]))

	acc_array = [acc_1,acc_2,acc_3,acc_5]
	test_metrics.plot_graphs(acc_array)

    # get_distinctness_score()

if __name__ == '__main__':
    main()