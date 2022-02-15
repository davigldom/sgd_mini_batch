# X = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]
# y = np.array([1, 0, 0, 2, 2, 1, 0, 2, 1, 1, 0, 2, 2, 2, 0, 2])
# rng = np.random.default_rng()
# res = rng.choice(X, 6, replace=False, shuffle=False)
# print(res)

# unique, counts = np.unique(y, return_counts=True)
# freq = np.divide(counts, np.sum(counts))
# print(freq)

# y2 = np.where(y==0, freq[0]/5, y)
# y2 = np.where(y==1, freq[1]/4, y2)
# y2 = np.where(y==2, freq[2]/7, y2)
# print(y2)

# res = rng.choice(y, 6, replace=False, shuffle=False, p=y2)
# print(res)


# Xe_votos, Xp_votos, ye_votos, yp_votos = particion_entr_prueba(X_votos, y_votos, test=1/3)
# Xe_credito, Xp_credito, ye_credito, yp_credito = particion_entr_prueba(X_credito, y_credito, test=0.4)
# print(np.unique(y_votos, return_counts=True))
# print(np.unique(ye_votos, return_counts=True))
# print(np.unique(yp_votos, return_counts=True))

# unique, counts = np.unique(y_votos, return_counts=True)
# freq = np.divide(counts, np.sum(counts))
# print(unique, counts)
# print(freq)

# total_freq = np.sum(freq * counts)
# print(total_freq)

# y2 = np.where(y_votos==unique[0], freq[0]/total_freq, y_votos)
# y2 = np.where(y_votos==unique[1], freq[1]/total_freq, y2)
# y2 = np.where(y2==unique[2], freq[2]/counts[2], y2)
# print(y2.astype(float))

# print(type(y2[0]))

# test_size = round(len(y_votos) * 1/3)
# print(test_size)

# res = rng.choice(y_votos, test_size, replace=False, shuffle=False, p=y2.astype(float))
# print(np.unique(y_credito, return_counts=True))
# print(np.unique(ye_credito, return_counts=True))
# print(np.unique(res, return_counts=True))


# X = np.array([[15, 67, 128, 12, 54, 78], [43, 58, 139, 10, 17, 1], [17, 69, 78, 165, 47, 32]])
# std = np.std(X, axis=0)
# mean = np.mean(X, axis=0)

# random.sample(range(0, len(cl_array)), test_examples)
# print(random.sample(range(0, 10), 6))
# print(np.random.randint(0, 10, size=6))
# rng = np.random.default_rng()
# rints = rng.integers(low=0, high=10, size=6)
# print(rints)
