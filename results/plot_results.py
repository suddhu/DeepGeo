import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np 
import pickle
import test_metrics


acc = [[0.2592,	0.3815,	0.4651,	0.5825],[0.2483,0.3612,	0.4407,	0.5532],[0.3832,0.5233,	0.6098,	0.7187],[0.2847,0.4096,	0.4944,	0.613]]

rand_chance = [0.02, 0.04, 0.06, 0.10]
acc = np.asarray(acc)


test_metrics.plot_graphs(acc[0,:], 'r-o','Single Image')
test_metrics.plot_graphs(acc[1,:], 'g-o','Early Integration')
test_metrics.plot_graphs(acc[2,:], 'b-o','Medium Integration')
test_metrics.plot_graphs(acc[3,:], 'm-o','Late Integration')
test_metrics.plot_graphs(rand_chance, 'k--','Random Chance')

#plt.axhline(0.02, color='k', linestyle='dashed', linewidth=2,label='Random Chance')
leg = plt.legend(loc=4,prop={'size': 10},fancybox=True)
leg.get_frame().set_alpha(0.5)


plt.show()