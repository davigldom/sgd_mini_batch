from carga_datos import X_votos, y_votos
import numpy as np

def particion_entr_prueba(X, y, test=0.20):
    classes = np.unique(y)
    folds_indices = []
    sobrante = []
    rng = np.random.default_rng()

    n = 5

    for cl in classes:
        cl_array = np.where(y == cl)[0]
        print(cl_array)

        test_examples = round(len(cl_array) * test)

        perm_index = rng.permutation(cl_array)

        cl_folds = np.split(perm_index, np.arange(test_examples, len(cl_array), test_examples))

        if len(cl_folds) > n:
            sobrante.extend(cl_folds[-1].tolist())

            print(sobrante)
            cl_folds = cl_folds[:-1]

        if len(cl_folds[-1]) != len(cl_folds[-2]):
            cl_folds[-1] = np.pad(cl_folds[-1], (0, len(cl_folds[-2])-len(cl_folds[-1])), constant_values=(-1))

        for fold in cl_folds:
            print(len(fold))

        if not folds_indices:
            folds_indices = cl_folds

        else:
            folds_indices = np.concatenate((folds_indices, cl_folds), axis=1)

        if np.where(folds_indices[-1] == -1)[0].size > 0:

            folds_indices[-1] = np.concatenate((folds_indices[-1][folds_indices[-1] != -1], sobrante), axis=None)
            print(folds_indices[-1])

    for fold in folds_indices:
        print(len(fold))

    folds_indices = np.sort(folds_indices)
    print(folds_indices)

    first_fold = y[folds_indices[0]]
    second_fold = y[folds_indices[1]]
    third_fold = y[folds_indices[2]]
    fourth_fold = y[folds_indices[3]]
    fifth_fold = y[folds_indices[4]]

    print(np.unique(first_fold, return_counts=True)[1])
    print(np.unique(second_fold, return_counts=True)[1])
    print(np.unique(third_fold, return_counts=True)[1])
    print(np.unique(fourth_fold, return_counts=True)[1])
    print(np.unique(fifth_fold, return_counts=True)[1])

    return 0

X_train = particion_entr_prueba(X_votos, y_votos)
