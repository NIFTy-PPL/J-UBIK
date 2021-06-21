import nifty7 as ift
import numpy as np

data_0 = np.load('test_0_observation.npy', allow_pickle=True).item()['data'].val[:,:,0]
data_1 = np.load('test_1_observation.npy', allow_pickle=True).item()['data'].val[:,:,0]

max_0 = np.max(data_0)
max_1 = np.max(data_1)

pos_0 = np.unravel_index(np.argmax(data_0), data_0.shape)
pos_1 = np.unravel_index(np.argmax(data_1), data_1.shape)
print(max_0, 'at', pos_0)
print(max_1, 'at', pos_1)

#modify for many obs and check the maximum position, value and next hightest val...
