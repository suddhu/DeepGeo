from geopy.distance import great_circle
import numpy as np
import googlemaps
from matplotlib import pyplot as plt
import pdb

import sys  
sys.path.append('../scraper')  
import location_sampler

states_file = 'states.xml'

borders= np.asarray(location_sampler.get_borders(states_file))
labels= location_sampler.get_labels(states_file)

nLabels = len(borders)

state_center = np.zeros([nLabels,2])


for i in range(0,nLabels):
    plt.fill(borders[i][:,0], borders[i][:,1], 'r-')
    border_A = map(float,borders[i][:,0])
    border_B = map(float,borders[i][:,1])
    #pdb.set_trace()

    state_center[i,1] = np.mean(border_A)
    state_center[i,0] = np.mean(border_B)
    plt.plot(state_center[i,1], state_center[i,0], 'b.')

np.save("state_center.npy", state_center)
plt.show()
plt.hold(True)

juneau_ak = (58.3019, 134.4197)
honolulu_hi = (21.3069, 157.8583)
print(great_circle(juneau_ak, honolulu_hi).kilometers)
