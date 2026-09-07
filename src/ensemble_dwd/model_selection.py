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
