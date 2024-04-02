import numpy as np

# Create a NumPy array
arr = np.array([1, 0, 2, 3, 0, 0, 4, 5, 0])

# Get the indices of elements that are zero
zero_indices = np.where(arr == 0)[0]

print(zero_indices)
