"""All ensemble classifiers reject missing temporal training rows."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
@pytest.mark.parametrize("cls", [EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier])
def test_training_nat_rejected_before_learners_or_pairs(cls, dtype):
    options = dict(M=1, theta=1., sv_selection_ratio=1., random_state=0)
    model = OvOClassifier(SVC(kernel="linear")) if cls is OvOClassifier else cls(**options)
    X = np.tile([[-1.], [1.]], (8, 1))
    y = np.tile(np.array([10, 20], dtype=dtype), 8)
    model.fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)
    y[::2] = "NaT"
    with pytest.raises(ValueError, match="NaT"):
        model.fit(X, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
