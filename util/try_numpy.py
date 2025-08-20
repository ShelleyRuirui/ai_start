import numpy as np

# Create a 1D array
arr = np.array([1, 2, 3])
print(f"Original array: {arr}")
print(f"Original shape: {arr.shape}")

# Expand dimensions at axis 0 (beginning)
expanded_arr_0 = np.expand_dims(arr, axis=0)
print(f"\nExpanded array (axis=0): {expanded_arr_0}")
print(f"Expanded shape (axis=0): {expanded_arr_0.shape}")

# Expand dimensions at axis -1 (end)
expanded_arr_end = np.expand_dims(arr, axis=-1)
print(f"\nExpanded array (axis=-1): {expanded_arr_end}")
print(f"Expanded shape (axis=-1): {expanded_arr_end.shape}")

# Expand for 2D array
arr_2d = np.array([[1,2,3], [4,5,6]])
print(f"\nOriginal 2D array: {arr_2d}")
print(f"Original 2D shape: {arr_2d.shape}")

# Expand dimensions at axis 0 (beginning)
expanded_arr_0 = np.expand_dims(arr_2d, axis=0)
print(f"\nExpanded array (axis=0): {expanded_arr_0}")
print(f"Expanded shape (axis=0): {expanded_arr_0.shape}")

# Expand dimensions at axis -1 (end)
expanded_arr_end = np.expand_dims(arr_2d, axis=-1)
print(f"\nExpanded array (axis=-1): {expanded_arr_end}")
print(f"Expanded shape (axis=-1): {expanded_arr_end.shape}")