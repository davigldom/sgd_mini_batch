# from carga_datos import X_votos, y_votos
# import numpy as np

# def sigmoid(z):
#     e_z = np.exp(-z)
#     sig = 1 / (1 + e_z)

#     return sig

# classes = np.unique(y_votos)

# for index, cl in enumerate(classes):
#     y_votos = np.where(y_votos==cl, index, y_votos)

# rng = np.random.default_rng()
# perm_index = rng.permutation(len(y_votos))

# batches = np.split(perm_index, np.arange(64, len(y_votos), 64))
# batches = np.split(perm_index, np.arange(2, len(y_votos), 2))

# pesos_iniciales = 2 * rng.random(len(X_votos[0])) - 1
# pesos_iniciales = np.array([0.59038264, -0.0135339, -0.22768076, -0.33651525, -0.01122216, 0.77853177, 0.23740992, -0.76859903, -0.69740355, -0.3266695, 0.47856895, -0.24333404, -0.84773037, 0.74499456, 0.41746506, -0.14930274])
# print(pesos_iniciales)


# for batch in batches:
    # X_batch = X_votos[batch]
    # y_batch = y_votos[batch].astype(float)
    # X_batch = np.array([[-1, 1, 1, -1, -1, 1, 1, 1, 0, -1, 1, 1, -1, -1, 1, 1], [-1, 0, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1]])
    # y_batch = np.array([0, 1])

    # print(X_batch)
    # print(y_batch)

    # result = 0

    # for index, peso in enumerate(pesos_iniciales):
    #     result += peso * X_batch[0][index]

    # print(result)

    # dot = np.tensordot(pesos_iniciales, X_batch, (0, 1))
    # print(dot)

    # dot = X_batch.dot(pesos_iniciales)

    # print(dot)

    # print(len(dot))

    # sigma = 1 / (1 + np.exp(-dot))

    # sigma = sigmoid(np.dot(pesos_iniciales, X_batch))
    # print(sigma)
    # print(len(y_batch))

    # sigma = y_batch - sigma
    # print(sigma)

    # summation = sigma.dot(X_batch)
    # print(summation)

    # pesos_iniciales = pesos_iniciales + 0.1 * summation
    # print(pesos_iniciales)

    # break

import numpy as np
array = np.array(["Hola", "Hola", "Adiós", "Adiós", "Hola"])
# print(np.where(array=="Hola")[0])
# np.put(array, [np.where(array=="Hola")[0], np.where(array=="Adiós")[0]], [0, 1])
# print(array)

# keep_mask = x==50
# out = np.where(x>50,0,1)
# out[keep_mask] = 50

array2 = np.where(array == "Hola", 0, 1)
print(array2)
