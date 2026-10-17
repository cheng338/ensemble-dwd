"""Explicit preprocessing choices for the local MNIST examples."""
import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data


class UnitPixelScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Map a declared pixel range to [0, 1] without estimating a divisor.

    ``input_range`` must be explicitly ``'0-255'`` or ``'0-1'``. Values outside
    that range are errors, never clipped. No range is inferred from the data.
    ``transform`` returns a separate float64 C-order array and never modifies
    the input. Fitting records feature metadata and the declared policy only.
    """

    def __init__(self, *, input_range):
        self.input_range = input_range

    def _validate_options(self):
        if type(self.input_range) is not str or self.input_range not in ('0-255', '0-1'):
            raise ValueError("Declare input_range as '0-255' or '0-1'.")

    def _clear_fitted(self):
        for name in ('n_features_in_', 'feature_names_in_', 'input_range_', 'divisor_'):
            self.__dict__.pop(name, None)

    def _validate_pixels(self, X, *, reset):
        raw = np.asarray(X)
        if raw.dtype.kind not in 'iuf':
            raise ValueError('Pixels must be a dense array of real integer or floating values.')
        array = validate_data(self, X, reset=reset, dtype=None, copy=False,
                              accept_sparse=False, ensure_all_finite=True)
        upper = 255.0 if self.input_range == '0-255' else 1.0
        if array.min() < 0.0 or array.max() > upper:
            raise ValueError('Pixels fall outside declared input_range=' + self.input_range)
        return array

    def fit(self, X, y=None):
        self._clear_fitted()
        try:
            self._validate_options()
            self._validate_pixels(X, reset=True)
            self.input_range_ = self.input_range
            self.divisor_ = 255.0 if self.input_range == '0-255' else 1.0
        except BaseException:
            self._clear_fitted()
            raise
        return self

    def transform(self, X):
        check_is_fitted(self, ('n_features_in_', 'input_range_', 'divisor_'))
        self._validate_options()
        if self.input_range != self.input_range_:
            raise ValueError('input_range changed after fitting; refit the transformer.')
        pixels = self._validate_pixels(X, reset=False)
        result = np.array(pixels, dtype=np.float64, order='C', copy=True)
        result /= self.divisor_
        return result


def add_preprocessing_arguments(parser):
    """Use fixed unit-pixel scaling by default; allow explicit standardization."""
    parser.add_argument('--preprocessing', choices=('standard', 'unit'), default='unit',
                        help='Pixel preprocessing: unit (default) uses the fixed pixel range; '
                             'standard fits mean/variance on training folds.')
    parser.add_argument('--pixel-input-range', choices=('0-255', '0-1'), default=None,
                        help='With unit preprocessing: raw 0-255 pixels (default) or already '
                             'normalized 0-1 pixels. No automatic range inference. '
                             'Not accepted with standard preprocessing.')


def _validate_policy(preprocessing, pixel_input_range):
    if type(preprocessing) is not str or preprocessing not in ('standard', 'unit'):
        raise ValueError("preprocessing must be 'standard' or 'unit'.")
    if preprocessing == 'standard':
        if pixel_input_range is not None:
            raise ValueError('--pixel-input-range requires --preprocessing unit.')
    else:
        if pixel_input_range is None:
            pixel_input_range = '0-255'
        if type(pixel_input_range) is not str or pixel_input_range not in ('0-255', '0-1'):
            raise ValueError("--pixel-input-range must be 0-255 or 0-1.")
    return pixel_input_range


def make_preprocessor(preprocessing='unit', pixel_input_range=None):
    """Default to raw pixels divided by 255; never infer a range from data."""
    pixel_input_range = _validate_policy(preprocessing, pixel_input_range)
    if preprocessing == 'standard':
        return StandardScaler()
    return UnitPixelScaler(input_range=pixel_input_range)


def preprocessing_metadata(preprocessing, pixel_input_range):
    """Record the effective policy, including defaults, without reading data."""
    pixel_input_range = _validate_policy(preprocessing, pixel_input_range)
    if preprocessing == 'standard':
        transform = 'StandardScaler: subtract training mean and divide by training standard deviation'
        learned = 'Training-fold mean and variance; training-constant features use scale 1'
    else:
        transform = 'X / 255' if pixel_input_range == '0-255' else 'X (already in [0, 1])'
        learned = 'None; the pixel range is declared explicitly'
    return {'preprocessing': preprocessing, 'pixel_input_range': pixel_input_range,
            'transform': transform, 'learned_statistics': learned}
