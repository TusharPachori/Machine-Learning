import imp
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base import clone, BaseEstimator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

mnist = fetch_mldata('MNIST original')
# print(mnist)

X, y = mnist["data"], mnist["target"]

"""------------------------------------------------------------------------------------------------------------------"""


some_digit = X[36000]
# some_digit_image = some_digit.reshape(28, 28)
#
# plt.imshow(some_digit_image, interpolation="nearest")
# plt.axis("off")
# plt.show()

"""------------------------------------------------------------------------------------------------------------------"""


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)


sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# print(sgd_clf.predict([some_digit]))

"""------------------------------------------------------------------------------------------------------------------"""

"""CROSS VALIDATION"""

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_folds = X_train[test_index]
    y_test_folds = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    n_correct = sum(y_pred == y_test_folds)
    # print(n_correct / len(y_pred))


"""------------------------------------------------------------------------------------------------------------------"""


# x = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# print(x)

"""------------------------------------------------------------------------------------------------------------------"""


# class Never5Classifier(BaseEstimator):
#
#     def fit(self, X, y=None):
#         pass
#
#     def predict(self, X):
#         return np.zeros((len(X), 1), dtype=bool)
#
#
# never_5_clf = Never5Classifier()
# x = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# print(x)

"""------------------------------------------------------------------------------------------------------------------"""


"""CONFUSION MATRIX"""

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
x = confusion_matrix(y_train_5, y_train_pred)


"""------------------------------------------------------------------------------------------------------------------"""


"""PRECISION,  RECALL AND F1_SCORE"""

# print(precision_score(y_train_5, y_train_pred))
#
# print(recall_score(y_train_5, y_train_pred))
#
print(f1_score(y_train_5, y_train_pred))


