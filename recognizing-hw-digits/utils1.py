from skimage.transform import rescale
from sklearn.model_selection import train_test_split
from sklearn import metrics

def preprocess(images, rescale_factor):
    resized_images = []
    for d in images:
        resized_images.append(rescale(d, rescale_factor, anti_aliasing=False))
    return resized_images


def create_splits(data, targets, test_size, valid_size , case = False):

    if(case == True):
        n_samples = data.size
        val_ratio = valid_size
        test_ratio = test_size
        train_ratio = 1 - val_ratio - test_ratio
        train_sample = math.ceil(n_samples *train_ratio)
        test_sample = math.ceil(n_samples *test_ratio)
        val_sample = math.ceil(n_samples *val_ratio)
        
        return train_sample , test_sample , val_sample



    X_train, X_test_valid, y_train, y_test_valid = train_test_split(
        data, targets, test_size=test_size + valid_size, shuffle=False
    )

    X_test, X_valid, y_test, y_valid = train_test_split(
        X_test_valid,
        y_test_valid,
        test_size=valid_size / (test_size + valid_size),
        shuffle=False,
    )



    return X_train, X_test, X_valid, y_train, y_test, y_valid


def test_(clf, X, y):
    predicted = clf.predict(X)
    acc = metrics.accuracy_score(y_pred=predicted, y_true=y)
    f1 = metrics.f1_score(y_pred=predicted, y_true=y, average="macro")

    return {"acc": acc, "f1": f1}