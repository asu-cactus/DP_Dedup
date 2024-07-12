def longest_increasing_subsequence(arr):
    if not arr:
        return []

    n = len(arr)
    lis = [1] * n  # Initialize LIS values for all indexes as 1
    prev_index = [-1] * n  # To track the previous index in the LIS

    # Compute LIS values in a bottom-up manner
    for i in range(1, n):
        for j in range(0, i):
            if arr[i] > arr[j] and lis[i] < lis[j] + 1:
                lis[i] = lis[j] + 1
                prev_index[i] = j

    # Find the maximum value in lis[] and its index
    max_len = max(lis)
    max_index = lis.index(max_len)

    # Reconstruct the longest increasing subsequence
    lis_index = []
    # lis_sequence = []
    while max_index != -1:
        lis_index.append(max_index)
        # lis_sequence.append(arr[max_index])
        max_index = prev_index[max_index]

    # Reverse the lis_sequence since we built it backwards
    lis_index.reverse()
    lis_sequence = [arr[i] for i in lis_index]

    return lis_index, lis_sequence


original_accs = [
    0.8232,
    0.8331,
    0.8441,
    0.8499,
    0.8558,
    0.8604,
    0.8604,
    0.8604,
    0.8604,
    0.8757,
    0.8900,
    0.8964,
    0.8996,
    0.9010,
    0.9020,
    0.9027,
    0.9040,
    0.9053,
    0.9061,
]
acc_drops = [
    0.0076,
    0.0133,
    0.0199,
    0.0198,
    0.0183,
    0.0180,
    0.0184,
    0.0154,
    0.0190,
    0.0187,
    0.0189,
    0.0184,
    0.0190,
    0.0198,
    0.0186,
    0.0194,
    0.0204,
    0.0140,
    0.0150,
    0.0266,
]

# arr = [10, 22, 9, 33, 21, 50, 41, 60, 80]
arr = [acc - acc_drop for acc, acc_drop in zip(original_accs, acc_drops)]
for e in arr:
    print(f"{e:.4f}", end=", ")
print()
lis_index, longest_subsequence = longest_increasing_subsequence(arr)
print(len(lis_index))
for e in longest_subsequence:
    print(f"{e:.4f}", end=", ")
