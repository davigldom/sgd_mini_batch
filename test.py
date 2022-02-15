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
# array = np.array(["Hola", "Hola", "Adiós", "Adiós", "Hola"])
# print(np.where(array=="Hola")[0])
# np.put(array, [np.where(array=="Hola")[0], np.where(array=="Adiós")[0]], [0, 1])
# print(array)

# keep_mask = x==50
# out = np.where(x>50,0,1)
# out[keep_mask] = 50

# array2 = np.where(array == "Hola", 0, 1)
# print(array2)

from carga_datos import X_votos, y_votos

def particion_entr_prueba(X, y, test=0.20):
    classes = np.unique(y)
    folds_indices = []
    rng = np.random.default_rng()

    n = 5

    for cl in classes:
        cl_array = np.where(y == cl)[0]

        test_examples = round(len(cl_array) * test)

        perm_index = rng.permutation(len(cl_array))

        cl_folds = np.split(perm_index, np.arange(test_examples, len(cl_array), test_examples))

        if len(cl_folds) > 5:

            cl_folds[-2] = np.concatenate((cl_folds[-2], cl_folds[-1]), axis=None)

            cl_folds = cl_folds[:-1]

        if not folds_indices:
            print("Empty")
            folds_indices = cl_folds

        else:
            print(type(folds_indices))
            # IMPORTANTE: Aquí se utiliza un for ya que split devuelve una lista de arrays de numpy, ya que estos pueden no tener el mismo tamaño.
            # Es por ello, que si se intentaba utilizar el método concatenate con axis=1 devolvía error, ya que solo detectaba una dimensión
            for index, element in enumerate(cl_folds):
                folds_indices[index] = np.sort(np.concatenate((folds_indices[index], element), axis=None))

    for fold in folds_indices:
        print(len(fold))
        print(fold)

        # test_indices = np.sort(np.append(test_indices, cl_array[random_indexes]))

    # X_train = np.delete(X, test_indices, axis=0)
    # y_train = np.delete(y, test_indices, axis=0)
    # X_test = np.array(X)[test_indices]
    # y_test = np.array(y)[test_indices]

    # return X_train, X_test, y_train, y_test
    return 0

# X_train = particion_entr_prueba(X_votos, y_votos)


X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Y = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

X_split = np.array_split(X, np.arange(3, len(X), 3))
# pad_with = len(X_split[-2])
# X_split[-1] = np.pad(X_split[-1], (0, 3-len(X_split[-1])), constant_values=(-1))
# print(X_split[-1])
# X_split = np.pad(X_split, ((0, 3),), constant_values=(-1))
print(X_split.shape)

Y_split = np.asarray(X_split, dtype=object)
print(Y_split)
print(Y_split.shape)

# Y_split = np.split(Y, np.arange(4, len(Y), 4))
# print(Y_split)

# array = np.array([X_split], dtype=object)
# print(array)
# print(array.shape)
# print(array[0].shape)

# X_split = np.pad(X_split, pad_with)
# print(X_split)

# x = np.empty((5, 10))
# y = np.empty((3, 10))
# z = np.empty((6, 10))

# test = np.array([x, y, z], dtype=object)
# test[0].shape
