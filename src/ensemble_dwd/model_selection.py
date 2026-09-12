"""Parameter grids for external, whole-estimator cross-validation."""


def mnist_param_grid(prefix="", *, broad=False):
    """Return an RBF starting grid for external MNIST parameter searches.

    This is a starting grid for scaled MNIST features, not a claim of optimal
    parameters for other data. Pass the dictionary to ``GridSearchCV`` around
    the complete ensemble (and any preprocessing pipeline). No fitting or
    cross-validation happens in this function.

    ``prefix`` is prepended literally to each parameter name. For a pipeline
    step named ``ensemble``, use ``prefix="ensemble__"``. The grid fixes neither
    ``M`` nor ``q``; configure those on the estimator. Each call returns fresh
    lists so a caller may refine the grid without affecting another call.
    The sampling rate is fixed at 10% in this preset.

    The default grid has five norm coefficients, five kernel coefficients,
    and five retention fractions (125 candidates). ``broad=True`` returns
    the original nine-by-nine coefficient grid (405 candidates), with the
    same endpoints and fractions. Extend boundary winners and refine the
    grid using training-only CV; this function does not do that automatically.

    ``lambd`` is the underlying DWD package's coefficient of the squared RKHS
    norm. ``gamma`` uses ``exp(-gamma * ||x - z||**2)``.
    """
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")
    if not isinstance(broad, bool):
        raise TypeError("broad must be a boolean")
    lambda_exponents = ((-30, -26, -22, -18, -14, -10, -6, -2, 0)
                        if broad else (-30, -22, -14, -6, 0))
    gamma_exponents = range(-8, 1) if broad else range(-8, 1, 2)
    return {
        prefix + "lambd": [2.0 ** exponent for exponent in lambda_exponents],
        prefix + "gamma": [10.0 ** exponent for exponent in gamma_exponents],
        prefix + "theta": [0.1],
        prefix + "sv_selection_ratio": [0.1, 0.15, 0.2, 0.25, 0.3],
    }


def mnist_svm_param_grid(prefix="", *, broad=False):
    """Return an RBF SVM starting grid with the DWD grid's candidate budget.

    The native SVC ``C`` values span 1e-4 through 1e4 logarithmically;
    larger C penalizes margin violations more strongly. The same numerical C
    is used by every base and final SVC, without sample-size rescaling. C is
    not numerically interchangeable with the DWD norm coefficient ``lambd``.

    Gamma, the fixed 10% sampling rate and retention fractions match
    ``mnist_param_grid``. Both presets contain 125 candidates by default;
    ``broad=True`` restores the original 405-candidate grid. Set M on the
    estimator (default 5).
    Equal candidate counts are comparable search budgets, not guarantees of
    equal runtime or equally optimal hyperparameters. Inspect CV boundary
    winners and refine/extend C and gamma using training data when necessary.

    Use external GridSearchCV with a preprocessing Pipeline. This function
    returns fresh lists and never fits an estimator or starts a search.
    ``prefix`` and ``broad`` have the same semantics as ``mnist_param_grid``.
    """
    grid = mnist_param_grid(prefix, broad=broad)
    grid.pop(prefix + "lambd")
    c_exponents = range(-4, 5) if broad else range(-4, 5, 2)
    return {prefix + "C": [10.0 ** exponent for exponent in c_exponents], **grid}
