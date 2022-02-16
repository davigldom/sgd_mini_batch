import numpy as np

# an_array = np.array([[1, 2], [3, 4]])
# shape = np.shape(an_array)
# print(an_array)
# print(shape)

# padded_array = np.full((3, 5), -1)
# print(padded_array)

# print(shape[0])
# padded_array[:shape[0],:shape[1]] = an_array
# print(padded_array)

a = np.array([20, 30, 40])
b = np.array([30, 20, 10])
c = np.array([10, 40, 30])

d = np.column_stack((a, b, c))
print(d)

print(np.max(d, axis=0))
print(np.argmax(d, axis=0))

a = np.empty(shape=(3, 1))
b = np.array([30, 20, 10])
d = np.column_stack((a, b))
print(d)
