import nifty8 as ift
import numpy as np

data_0 = np.load('test_0_observation.npy', allow_pickle=True).item()['data'].val[:,:,2]
data_1 = np.load('test_1_observation.npy', allow_pickle=True).item()['data'].val[:,:,1]
data_2 = np.load('14_6_0_observation.npy', allow_pickle=True).item()['data'].val[:,:,3]

max_0 = np.max(data_0)
max_1 = np.max(data_1)
max_2 = np.max(data_2)


pos_0 = np.unravel_index(np.argmax(data_0), data_0.shape)
pos_1 = np.unravel_index(np.argmax(data_1), data_1.shape)
pos_2 = np.unravel_index(np.argmax(data_2), data_2.shape)
print(max_0, 'at', pos_0)
print(max_1, 'at', pos_1)
print(max_2, 'at', pos_2)

#modify for many obs and check the maximum position, value and next hightest val...
