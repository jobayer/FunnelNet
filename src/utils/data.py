from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold


def smote(data, labels, seed):
    data_reshaped = data.reshape(data.shape[0], -1)

    smoteier = SMOTE(random_state=seed)

    x_res, y_res = smoteier.fit_resample(data_reshaped, labels)

    features = x_res.reshape(x_res.shape[0], *data.shape[1:])
    labels = y_res

    print("SMOTEing done.")

    return features, labels


def skf(fold, features, labels, seed):
    cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    folds = cv.split(features, labels)

    return folds
