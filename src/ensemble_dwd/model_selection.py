"""Parameter grids for external, whole-estimator cross-validation."""


def mnist_param_grid(prefix=""):
    """Return a broad RBF grid used to motivate the MNIST comparison.

    This is a starting grid for scaled MNIST features, not a claim of optimal
    parameters for other data. Pass the dictionary to ``GridSearchCV`` around
    the complete ensemble (and any preprocessing pipeline). No fitting or
    cross-validation happens in this function.

    ``prefix`` is prepended literally to each parameter name. For a pipeline
    step named ``ensemble``, use ``prefix="ensemble__"``. The grid fixes neither
    ``M`` nor ``q``; configure those on the estimator. Each call returns fresh
    lists so a caller may refine the grid without affecting another call.

    ``lambd`` is the underlying DWD package's coefficient of the squared RKHS
    norm. ``gamma`` uses ``exp(-gamma * ||x - z||**2)``.
    """
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")
    return {
        prefix + "lambd": [2.0 ** exponent for exponent in
                           (-30, -26, -22, -18, -14, -10, -6, -2, 0)],
        prefix + "gamma": [10.0 ** exponent for exponent in range(-8, 1)],
        prefix + "theta": [0.1, 0.2],
        prefix + "sv_selection_ratio": [0.1, 0.15, 0.2, 0.25, 0.3],
    }


def mnist_svm_param_grid(prefix=""):
    """Return an RBF SVM starting grid with the DWD grid's candidate budget.

    The nine native SVC ``C`` values span 1e-4 through 1e4 logarithmically;
    larger C penalizes margin violations more strongly. The same numerical C
    is used by every base and final SVC, without sample-size rescaling. C is
    not numerically interchangeable with the DWD norm coefficient ``lambd``.

    Gamma, sampling rates and retention fractions match ``mnist_param_grid``.
    Both presets contain 810 candidates. Set M on the estimator (default 5).
    Equal candidate counts are comparable search budgets, not guarantees of
    equal runtime or equally optimal hyperparameters. Inspect CV boundary
    winners and refine/extend C and gamma using training data when necessary.

    Use external GridSearchCV with a preprocessing Pipeline. This function
    returns fresh lists and never fits an estimator or starts a search.
    ``prefix`` has the same literal-prefix semantics as ``mnist_param_grid``.
    """
    grid = mnist_param_grid(prefix)
    grid.pop(prefix + "lambd")
    return {prefix + "C": [10.0 ** exponent for exponent in range(-4, 5)], **grid}
