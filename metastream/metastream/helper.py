import numpy as np

from imblearn.metrics import geometric_mean_score
from sklearn.metrics import cohen_kappa_score as _cohen
from sklearn.metrics import mean_squared_error, accuracy_score


def cohen_kappa_score(y_true, y_pred):
  return 1.0 if np.array_equal(y_true, y_pred) else _cohen(y_true, y_pred)

