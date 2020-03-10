import numpy as np
read_dictionary = np.load('feature_cache_context.npy',allow_pickle=True).item()
#print(read_dictionary)
print(read_dictionary[0])