"""Binary ensemble kernel DWD with consensus selection and a final refit."""

from copy import deepcopy
from fractions import Fraction
from numbers import Integral, Real
from os import PathLike

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from dwd.gen_kern_dwd import KernGDWD

from ._cache import _CacheIdentityError, _cached_base_stage, _runtime_provenance, _RuntimeVerifier
from ._failure_context import attach_failure_context, failure_context_scope, uses_implementation
from .profiling import _profiled, _record_cache_result, _stage


class IneligibleSelectionError(ValueError):
    """A requested training-row selection cannot train a binary classifier."""


class IneligibleSubsetError(IneligibleSelectionError):
    """The requested or sampled disjoint base subsets are ineligible."""


_SAMPLING_SCHEME = "disjoint-uniform-permutation-v1"


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _finite_real(value, name, *, allow_zero=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero)):
        domain = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {domain} number.")
    return float(value)


def _fraction(value, name):
    value = _finite_real(value, name)
    if value > 1:
        raise ValueError(f"{name} must be in (0, 1].")
    return value


def _private_random_state(seed):
    if seed is None:
        return np.random.RandomState()
    source = check_random_state(seed)
    result = np.random.RandomState(0)
    result.set_state(source.get_state())
    return result


def _sample_subsets(y_encoded, size, M, rng):
    """Partition one uniform permutation into M disjoint base subsets.

    The draw is independent of labels. An ineligible allocation raises before
    any learner is fitted; subsets are never redrawn or repaired using labels.
    """
    y_encoded = np.asarray(y_encoded)
    size = _positive_integer(size, "subset_size")
    M = _positive_integer(M, "M")
    n = len(y_encoded)
    if size < 2:
        raise IneligibleSubsetError("Each base subset must contain at least two observations.")
    if M * size > n:
        raise IneligibleSubsetError(
            f"M={M} disjoint subsets of size {size} require {M * size} distinct "
            f"observations, but only {n} are available.")
    counts = [int(np.count_nonzero(y_encoded == label)) for label in (0, 1)]
    if min(counts) < M:
        raise IneligibleSubsetError(
            f"M={M} disjoint binary subsets require at least {M} observations "
            f"from each class; the class counts are {counts}.")
    permutation = rng.permutation(n)
    subsets = [block.copy() for block in permutation[:M * size].reshape(M, size)]
    invalid = [index + 1 for index, rows in enumerate(subsets)
               if len(np.unique(y_encoded[rows])) != 2]
    if invalid:
        raise IneligibleSubsetError(
            f"The uniform disjoint draw produced single-class base subsets "
            f"at positions {invalid}. This sampling candidate is ineligible.")
    return subsets


def _binary_scores(model, X):
    scores = np.asarray(model.decision_function(X), dtype=np.float64)
    if scores.shape == (X.shape[0], 1):
        scores = scores[:, 0]
    if scores.shape != (X.shape[0],) or not np.isfinite(scores).all():
        raise FloatingPointError("DWD must return one finite decision value per observation.")
    return scores


def _overflow_safe_consensus(models, X, batch_size):
    """Recover a finite mean when an intermediate float64 sum overflowed."""
    result = np.empty(len(X), dtype=np.float64)
    size = min(batch_size or 256, 256)
    for start in range(0, len(X), size):
        stop = min(start + size, len(X))
        scores = [_binary_scores(model, X[start:stop]) for model in models]
        for offset, values in enumerate(zip(*scores)):
            result[start + offset] = float(
                sum((Fraction.from_float(float(value)) for value in values), Fraction())
                / len(models))
    return result


def _contains_callable(value):
    if callable(value):
        return True
    if isinstance(value, dict):
        return any(_contains_callable(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_callable(item) for item in value)
    return False


_TIMING_FIELDS = ("setup_seconds", "optimization_seconds", "total_seconds")
_SOLVER_DIAGNOSTIC_FIELDS = (
    *_TIMING_FIELDS, "requested_backend", "initial_backend", "effective_backend",
    "auto_fallback", "fallback_reason", "prediction_precision",
    "score_reconstruction_max_error", "objective_reconstruction_error",
    "optimality_criterion", "optimality_tolerance", "coefficient_representation",
    "returned_state_uses_certified_columns", "residual_check_order",
    "requested_residual_check_order",
    ("linear_system_diagnostics", (
        "linear_system", "factor_representation", "linear_solves", "linear_recoveries",
        "refinement_steps", "max_linear_residual", "max_constraint_residual",
        "max_estimated_rkhs_solve_error", "added_objective_regularization",
        "positive_eigenvalues_discarded", "residual_checks_are_interval_certificates",
        "native_refinement_trials", "native_refinement_discarded",
        "native_refinement_acceptances", "native_refinement_seconds",
        "compensated_residual_checks", "compensated_residual_seconds",
        "native_residual_checks", "native_residual_seconds",
        "adaptive_residual_probes", "adaptive_residual_probe_errors",
        "last_adaptive_residual_probe_error", "adaptive_residual_probe_seconds",
        "adaptive_residual_probe_invalidations", "adaptive_residual_probe_acceptances",
        "adaptive_native_trials_skipped", "adaptive_residual_probe_fallbacks",
        "adaptive_residual_probe_reuses", "adaptive_residual_history_resets")),
    ("dual_certificate", (
        "feasibility", "slope_quantum", "dual_quadratic_method", "fallback_reason",
        "dual_objective_lower_bound", "dual_objective_is_lower_bound",
        "kernel_assumption", "primal_objective_and_gap_are_estimates",
        "arithmetic_assumptions")),
    ("mm_function_recovery", (
        "attempted", "accepted_actions", "native_small_spectral_actions",
        "rounded_exact_actions", "reduced_coefficient_gauge_checked",
        "true_mm_function_checked", "kernel_approximation_used",
        "positive_directions_discarded", "certificate_status", "exact_recovery",
        "initial_numerical_failure", "total_seconds",
        ("resource_caps", ("max_entries", "max_rank", "max_bits", "max_operations")),
        ("certificate", ("certified", "rank", "n", "seconds", "max_entries",
                         "max_rank", "max_bits", "max_operations")),
        ("preparation", ("status", "n", "rank", "delta", "gauge",
                         "positive_directions_discarded", "original_kernel_unchanged",
                         "preparation_seconds", "budget_scope")),
        ("last_action", ("status", "accepted_candidate", "absolute_offset_checked",
                         "exact_Kalpha_scores_rounded", "reduced_A_gauge_checked",
                         "true_MM_function_checked", "rkhs_norm_squared")))),
    ("spectral_restart", (
        "attempts", "restarts", "discarded_completed_updates", "failed_attempted_iteration",
        "successful_updates", "total_completed_updates", "total_attempted_updates",
        "max_iter_per_attempt", "initialization_reused", "original_kernel_reused",
        "history_scope",
        ("successful_attempt_timing", _TIMING_FIELDS),
        ("failed_attempt", ("completed_iterations", "failed_iteration", "setup_seconds",
                            "optimization_seconds", "elapsed_seconds", "backend",
                            "mm_function_recovery_attempted",
                            "mm_function_recovery_accepted_actions",
                            "mm_function_certificate_status")))),
)


def _compact_diagnostic_fields(source, fields):
    """Copy an explicit, bounded scalar summary; absent/omitted facts stay unknown."""
    result = {}
    if not isinstance(source, dict):
        return result
    for field in fields:
        if isinstance(field, tuple):
            name, children = field
            value = _compact_diagnostic_fields(source.get(name), children)
            if value:
                result[name] = value
        elif field in source:
            value = source[field]
            if isinstance(value, np.generic):
                value = value.item()
            # No arrays, histories or custom objects survive lean storage.
            # Long text is omitted, not clipped into a misleading partial fact.
            if value is None or type(value) in (bool, int, float):
                result[field] = value
            elif type(value) is str and len(value) <= 1024:
                result[field] = value
    return result


def _base_diagnostics(model):
    """Copy reported facts without inferring convergence or retaining solver state."""
    names = ("n_iter_", "returned_iteration_", "converged_", "criterion_reached_",
             "termination_reason_", "backend_", "optimality_met_",
             "objective_tolerance_met_", "gradient_inf_norm_", "rkhs_gradient_norm_",
             "dual_gap_", "dual_equality_residual_", "exact_recovery_",
             "final_objective_", "total_iterations_", "stationarity_residual_",
             "stationarity_checked_", "C_conversion_finite_", "prediction_precision_")
    values = {name[:-1]: getattr(model, name) for name in names if hasattr(model, name)}
    result = _compact_diagnostic_fields(values, tuple(name[:-1] for name in names))
    attempts = getattr(model, "iterations_per_attempt_", None)
    if isinstance(attempts, np.ndarray) and attempts.ndim == 1 and attempts.size <= 16:
        attempts = attempts.tolist()
    if isinstance(attempts, (list, tuple)) and len(attempts) <= 16:
        counts = _compact_diagnostic_fields(dict(enumerate(attempts)), tuple(range(len(attempts))))
        if len(counts) == len(attempts):
            result["iterations_per_attempt"] = [counts[index] for index in range(len(attempts))]
    solver = _compact_diagnostic_fields(getattr(model, "diagnostics_", None),
                                        _SOLVER_DIAGNOSTIC_FIELDS)
    if solver:
        result["solver_diagnostics"] = solver
    return result


class EnsembleKernelDWD(ClassifierMixin, BaseEstimator):
    """Fit Chang Cheng's binary ensemble kernel DWD procedure.

    Split one uniform random permutation into M disjoint subsets; fit optimized
    kernel DWD to each; rank all training rows by
    the absolute mean of their raw decision scores; fit a final optimized DWD
    to the selected rows. Predictions come from that final model.

    Parameters
    ----------
    M : int, default=5
        Number of base learners. A cache miss trains M plus one final model;
        a cache hit reuses the M bases and trains a fresh final model.
        The M subsets must fit within the available rows, and each must contain
        both classes. Ineligible draws raise without resampling.
    theta : float in (0, 1], default=0.1
        Subset fraction. Each subset has max(2, floor(theta * n_samples)) rows.
        M times this size must not exceed n_samples.
    sv_selection_ratio : float in (0, 1], default=0.3
        Retain floor(sv_selection_ratio * n_samples) rows for the final fit.
        The retained rows must contain at least two observations and both classes.
    lambd : positive float, default=1e-6
        Shared coefficient of the squared RKHS norm in the mean-loss objective.
        The dissertation's lambda is twice this package parameter.
    q : positive float, default=1.0
        Generalized DWD power. The dissertation method uses q=1.
    kernel : str or callable, default='rbf'
        Feature-based kernel accepted by the optimized DWD dependency.
        Precomputed kernels are not supported by this row-sampling estimator.
    gamma : {'scale', 'auto'} or nonnegative float, default='scale'
        Shared kernel coefficient. 'scale' means 1 / (n_features * X.var()),
        or 1 for constant data; 'auto' means 1 / n_features. Resolve once
        from the current fit's input, before sampling. For RBF the kernel is
        exp(-gamma * squared_distance). Ignored by kernels without gamma.
    rbf_computation : {'standard', 'direct'}, default='standard'
        DWD RBF arithmetic policy, forwarded to every base and final learner.
        Direct requires named RBF with no extra kernel_kws beyond top-level gamma.
        It is opt-in, may change numerical scores and retained observations,
        and may cost more. Sampling, selection and final-refit rules are unchanged.
    affine_computation : {'standard', 'joint'}, default='standard'
        Decision-score arithmetic, forwarded to every base and final learner.
        Joint includes the intercept in cancellation-aware summation of
        K @ alpha + b. It can change rounded scores, selected rows and labels,
        and can be expensive for many uncertain or exactly zero scores.
    exact_recovery : {'standard', 'extended'}, default='standard'
        Budget for bounded exact-rank recovery in every DWD base and final fit.
        Extended raises the certificate rank and integer-bit limits to 32 and
        8192, retaining the 65536-entry and 1000000-operation limits. It can
        cover additional small difficult kernels but does not guarantee
        convergence or make an indefinite kernel positive semidefinite.
    kernel_kws : dict or None, default=None
        Other kernel keywords, such as degree and coef0. Set gamma using the
        top-level parameter, not this dictionary, for built-in kernels.
    max_iter : positive int, default=100
        Maximum updates for each DWD fit. Reaching this cap is not convergence.
    obj_tol : nonnegative float, default=1e-5
        Absolute successive-objective tolerance for objective stopping.
    stopping : {'objective', 'fixed', 'optimality'}, default='objective'
        Shared stopping policy. Validation stopping needs separate per-learner
        validation provisioning and is not supported by this estimator.
    tol : positive float, default=1e-6
        Shared optimality tolerance, forwarded to optimized DWD.
    acceleration : {None, 'restart'}, default=None
        Optional optimized MM acceleration. Ordinary MM is the default.
    random_state : int, RandomState or None, default=42
        Subset randomness. Integer seeds give repeatable subsets across CV
        candidates. A supplied RandomState is copied; global RNG is not changed.
    prediction_batch_size : positive int or None, default=1024
        Kernel prediction batch size for base scoring and final predictions.
        None permits the dependency to process the whole query at once.
    cache_dir : str, path-like or None, default=None
        Optional trusted directory for sharing deterministic base fits across
        selection ratios. Keys include training data, subsets, base parameters,
        source and numerical runtime. None disables caching. Callable kernels,
        custom estimator subclasses and random_state=None bypass the cache.
    store_base_estimators : bool, default=True
        Retain fitted base models for inspection. False stores only their
        diagnostic arrays and consensus, reducing model and cache storage.
    backend : {'auto', 'spectral', 'cholesky', 'lbfgs'}, default='auto'
        Shared optimized DWD backend for every base and final fit. 'auto'
        retains the dependency's default choice; 'spectral' and 'cholesky'
        select MM representations, and 'lbfgs' selects its alternative optimizer.
        Different backends can produce different finite-budget decision scores
        and selected rows. Backend choices have separate base-cache entries.
    residual_check_order : {'refinement_first', 'adaptive'}, default='refinement_first'
        Residual-check ordering forwarded to every optimized DWD learner.
        The default preserves the existing order; 'adaptive' is an opt-in
        ordering policy with separate base-cache entries. Acceptance criteria
        are unchanged, but finite-budget outputs need not be bitwise identical.
        Optimized Cholesky MM supports the policy with or without restart
        acceleration with the paired DWD 1.3.14 dependency.

    Attributes
    ----------
    estimators_ : list or None
        The M fitted optimized DWD base learners, or None in lean storage mode.
    final_estimator_ : KernGDWD
        The independently fitted final classifier used for prediction.
    base_diagnostics_ : list of dict
        Reported stopping, backend and residual diagnostics of each original
        base fit, including in lean storage mode and when reusing the cache.
        A reached stopping criterion does not imply certified convergence.
    subset_indices_ : list of integer arrays
        Base training positions relative to the data passed to this fit.
    selected_indices_ : integer array
        Final training positions, ordered by absolute consensus score. Exact
        ties retain the current input row order.
    consensus_scores_ : ndarray of shape (n_samples,)
        Mean raw base decision scores on the current training data.
    gamma_ : float or None
        Resolved shared coefficient, or None for kernels without gamma.
    affine_computation_ : {'standard', 'joint'}
        Decision-score policy used by the fitted base and final learners.
        Changing the constructor parameter takes effect on the next fit.
    exact_recovery_ : {'standard', 'extended'}
        Recovery budget requested for the completed ensemble fit.
    n_iter_, converged_ : ndarray of shape (M + 1,)
        Per-learner update counts and checked convergence flags, final model last.
    cache_hit_ : bool
        Whether a previously completed base stage was reused.
    cache_key_ : str or None
        Key of the base stage, or None when the cache was disabled or bypassed.
    cache_status_ : str
        'hit', 'miss', 'disabled', or the reason for bypassing the cache.

    Notes
    -----
    This estimator accepts exactly two classes. Use
    ``OvOClassifier(EnsembleKernelDWD(...))`` for multiclass classification.
    Each class pair fits its own ensemble, and only its final model votes.

    Inputs are dense feature rows converted to float64. Use a StandardScaler
    before this estimator inside a Pipeline to fit preprocessing within CV.
    Each learner has an unregularized intercept and identical kernel, loss,
    regularization and stopping parameters. This estimator performs no CV,
    voting, score normalization, or silent substitution of a different model.
    Base fits run sequentially; parallelize outer CV fits according to memory.
    """

    # Old serialized estimators retain the standard DWD policy.
    rbf_computation = "standard"
    affine_computation = "standard"
    exact_recovery = "standard"

    def __init__(self, M=5, theta=0.1, sv_selection_ratio=0.3, lambd=1e-6,
                 q=1.0, kernel="rbf", gamma="scale", kernel_kws=None,
                 max_iter=100, obj_tol=1e-5, stopping="objective", tol=1e-6,
                 acceleration=None, random_state=42, prediction_batch_size=1024,
                 cache_dir=None, store_base_estimators=True, backend="auto",
                 residual_check_order="refinement_first", rbf_computation="standard",
                 affine_computation="standard", exact_recovery="standard"):
        self.M = M
        self.theta = theta
        self.sv_selection_ratio = sv_selection_ratio
        self.lambd = lambd
        self.q = q
        self.kernel = kernel
        self.gamma = gamma
        self.kernel_kws = kernel_kws
        self.max_iter = max_iter
        self.obj_tol = obj_tol
        self.stopping = stopping
        self.tol = tol
        self.acceleration = acceleration
        self.random_state = random_state
        self.prediction_batch_size = prediction_batch_size
        self.cache_dir = cache_dir
        self.store_base_estimators = store_base_estimators
        self.backend = backend
        self.residual_check_order = residual_check_order
        self.rbf_computation = rbf_computation
        self.affine_computation = affine_computation
        self.exact_recovery = exact_recovery

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.sparse = False
        return tags

    def __sklearn_is_fitted__(self):
        return hasattr(self, "final_estimator_")

    def _clear_fit(self):
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)

    def _validate_options(self):
        if not isinstance(self.exact_recovery, str) or self.exact_recovery not in (
                "standard", "extended"):
            raise ValueError("exact_recovery must be 'standard' or 'extended'.")
        if not isinstance(self.affine_computation, str) or self.affine_computation not in (
                "standard", "joint"):
            raise ValueError("affine_computation must be 'standard' or 'joint'.")
        if not isinstance(self.rbf_computation, str) or self.rbf_computation not in (
                "standard", "direct"):
            raise ValueError("rbf_computation must be 'standard' or 'direct'.")
        if self.rbf_computation == "direct" and (
                not isinstance(self.kernel, str) or self.kernel != "rbf"):
            raise ValueError("rbf_computation='direct' requires the named RBF kernel.")
        _positive_integer(self.M, "M")
        _positive_integer(self.max_iter, "max_iter")
        _fraction(self.theta, "theta")
        _fraction(self.sv_selection_ratio, "sv_selection_ratio")
        _finite_real(self.lambd, "lambd")
        _finite_real(self.q, "q")
        _finite_real(self.obj_tol, "obj_tol", allow_zero=True)
        _finite_real(self.tol, "tol")
        if isinstance(self.gamma, str):
            if self.gamma not in ("scale", "auto"):
                raise ValueError("gamma must be 'scale', 'auto', or a finite nonnegative number.")
        else:
            _finite_real(self.gamma, "gamma", allow_zero=True)
        if not isinstance(self.kernel, str) and not callable(self.kernel):
            raise TypeError("kernel must be a name or a callable.")
        if isinstance(self.kernel, str) and self.kernel == "precomputed":
            raise ValueError("Precomputed kernels are unsupported; supply feature rows.")
        if self.kernel_kws is not None and not isinstance(self.kernel_kws, dict):
            raise TypeError("kernel_kws must be a dictionary or None.")
        if (isinstance(self.kernel, str) and self.kernel_kws is not None
                and "gamma" in self.kernel_kws):
            raise ValueError("Set gamma using the top-level gamma parameter, not kernel_kws.")
        if not isinstance(self.stopping, str) or self.stopping not in (
                "objective", "fixed", "optimality"):
            raise ValueError("stopping must be 'objective', 'fixed', or 'optimality'.")
        if self.acceleration is not None and (
                not isinstance(self.acceleration, str) or self.acceleration != "restart"):
            raise ValueError("acceleration must be None or 'restart'.")
        if not isinstance(self.backend, str) or self.backend not in (
                "auto", "spectral", "cholesky", "lbfgs"):
            raise ValueError("backend must be 'auto', 'spectral', 'cholesky', or 'lbfgs'.")
        if not isinstance(self.residual_check_order, str) or self.residual_check_order not in (
                "refinement_first", "adaptive"):
            raise ValueError("residual_check_order must be 'refinement_first' or 'adaptive'.")
        if self.residual_check_order == "adaptive" and (
                self.backend not in ("auto", "cholesky")):
            raise ValueError("Adaptive residual ordering requires optimized Cholesky MM.")
        if self.prediction_batch_size is not None:
            _positive_integer(self.prediction_batch_size, "prediction_batch_size")
        if self.cache_dir is not None and not isinstance(self.cache_dir, (str, PathLike)):
            raise TypeError("cache_dir must be a directory path or None.")
        if not isinstance(self.store_base_estimators, (bool, np.bool_)):
            raise TypeError("store_base_estimators must be boolean.")

    def _kernel_parameters(self, X):
        kws = deepcopy(self.kernel_kws) if self.kernel_kws is not None else {}
        uses_gamma = isinstance(self.kernel, str) and self.kernel in (
            "rbf", "poly", "polynomial", "sigmoid", "laplacian", "chi2")
        if not uses_gamma:
            return None, kws
        if isinstance(self.gamma, str):
            if self.gamma == "auto":
                gamma = 1.0 / X.shape[1]
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    variance = float(X.var())
                if not np.isfinite(variance):
                    raise ValueError("Feature variance overflowed; rescale X or specify numeric gamma.")
                if variance == 0 and X.min() != X.max():
                    raise ValueError("Feature variance underflowed; rescale X or specify numeric gamma.")
                gamma = (1.0 / X.shape[1]) / variance if variance != 0 else 1.0
                if not np.isfinite(gamma):
                    raise ValueError("Scaled gamma overflowed; rescale X or specify numeric gamma.")
        else:
            gamma = float(self.gamma)
        kws["gamma"] = gamma
        return gamma, kws

    def _make_estimator(self, kernel_kws):
        # Avoid adding a dependency keyword to the unchanged standard route.
        numerical_options = {}
        if self.rbf_computation == "direct":
            numerical_options["rbf_computation"] = "direct"
        if self.affine_computation == "joint":
            numerical_options["affine_computation"] = "joint"
        if self.exact_recovery == "extended":
            numerical_options["exact_recovery"] = "extended"
        try:
            return KernGDWD(
                lambd=self.lambd, q=self.q, kernel=self.kernel,
                kernel_kws=deepcopy(kernel_kws), max_iter=self.max_iter,
                obj_tol=self.obj_tol, stopping=self.stopping, tol=self.tol,
                acceleration=self.acceleration, prediction_batch_size=self.prediction_batch_size,
                implementation="optimized", solver_mode="schur", backend=self.backend,
                initialization="zero", random_state=0,
                residual_check_order=self.residual_check_order, **numerical_options)
        except TypeError as error:
            if numerical_options:
                raise ImportError(
                    "The installed DWD dependency could not construct a learner with "
                    "the requested numerical computation options. Check the chained "
                    "error and install the dependency pinned in this package's "
                    "metadata.") from error
            raise ImportError(
                "EnsembleKernelDWD requires the optimized cheng338/dwd dependency. "
                "Install the dependency pinned in this package's metadata.") from error

    @_profiled("fit")
    def fit(self, X, y):
        """Fit the complete binary ensemble using only the supplied training data."""
        self._clear_fit()
        with failure_context_scope():
            try:
                return self._fit(X, y)
            except BaseException:
                self._clear_fit()
                raise

    @_profiled("base_stage")
    def _base_stage(self, X, y, classes, subsets, kernel_kws):
        context = dict(family="ensemble_dwd", stage="base_setup", base_position=None,
                       base_fit_calls_completed=0, base_score_calls_completed=0,
                       cached_base_learners=0, final_fit_calls_completed=0,
                       base_work_source="current")
        indices = None
        try:
            models = []
            score_sum = np.zeros(len(y), dtype=np.float64)
            for position, indices in enumerate(subsets, 1):
                context.update(stage="base_fit", base_position=position)
                with _stage("base_fit"):
                    model = self._make_estimator(kernel_kws).fit(X[indices], y[indices])
                context["base_fit_calls_completed"] += 1
                context["stage"] = "base_validation"
                with _stage("base_validation"):
                    if not np.array_equal(model.classes_, classes):
                        raise ValueError("All DWD learners must use the same ordered binary labels.")
                context["stage"] = "base_scoring"
                with _stage("base_scoring"), np.errstate(over="ignore", invalid="ignore"):
                    scores = _binary_scores(model, X)
                    context["base_score_calls_completed"] += 1
                    score_sum += scores
                    del scores
                models.append(model)
            context.update(stage="base_consensus", base_position=None)
            indices = None
            with _stage("base_consensus"):
                if np.isfinite(score_sum).all():
                    consensus = score_sum / self.M
                else:
                    with _stage("base_consensus_recovery"):
                        consensus = _overflow_safe_consensus(models, X, self.prediction_batch_size)
            context["stage"] = "base_summary"
            with _stage("selection_sort"):
                order = np.argsort(np.abs(consensus), kind="stable")
            with _stage("base_summary"):
                return dict(consensus=consensus, order=order,
                            models=models if self.store_base_estimators else None,
                            diagnostics=[_base_diagnostics(model) for model in models],
                            n_iter=np.asarray([model.n_iter_ for model in models], dtype=int),
                            converged=np.asarray([model.converged_ for model in models], dtype=bool))
        except BaseException as error:
            try:
                attach_failure_context(error, context, y=y, classes=classes, subset=indices)
            except Exception:
                pass
            raise

    @_profiled("cache_base_stage")
    def _base_result(self, X, y, classes, subsets, kernel_kws):
        context = dict(family="ensemble_dwd", stage="base_cache", base_position=None,
                       base_fit_calls_completed=None, base_score_calls_completed=None,
                       cached_base_learners=None, final_fit_calls_completed=0,
                       base_work_source="unknown")

        def compute():
            result = self._base_stage(X, y, classes, subsets, kernel_kws)
            # Only the known implementation establishes these completed counts.
            if uses_implementation(self, "_base_stage", EnsembleKernelDWD._base_stage):
                context.update(base_fit_calls_completed=len(subsets),
                               base_score_calls_completed=len(subsets),
                               cached_base_learners=0, base_work_source="current")
            return result

        try:
            if self.cache_dir is None:
                reason = "disabled"
            elif self.random_state is None:
                reason = "bypassed: random_state=None"
            elif callable(self.kernel) or _contains_callable(kernel_kws):
                reason = "bypassed: callable kernel or keyword"
            elif type(self) is not EnsembleKernelDWD:
                reason = "bypassed: custom estimator subclass"
            else:
                try:
                    provenance = _runtime_provenance(KernGDWD)
                except _CacheIdentityError as error:
                    return compute(), False, None, "bypassed: " + str(error)
                parameters = self.get_params(deep=False).copy()
                for name in ("sv_selection_ratio", "cache_dir", "random_state", "gamma", "kernel_kws"):
                    parameters.pop(name)
                parameters["kernel_kws"] = kernel_kws
                key_data = dict(sampling=_SAMPLING_SCHEME,
                                X=X, y=y, classes=classes, subsets=subsets, parameters=parameters,
                                provenance=provenance)
                verifier = _RuntimeVerifier(lambda: _runtime_provenance(KernGDWD), provenance)
                result, hit, key = _cached_base_stage(self.cache_dir, key_data, compute, verify=verifier)
                if hit:
                    context.update(base_fit_calls_completed=0, base_score_calls_completed=0,
                                   cached_base_learners=len(subsets), base_work_source="cache")
                status = "bypassed: " + verifier.failure if verifier.failure else ("hit" if hit else "miss")
                return result, hit, key, status
            return compute(), False, None, reason
        except BaseException as error:
            try:
                attach_failure_context(error, context, y=y, classes=classes)
            except Exception:
                pass
            raise

    def _fit(self, X, y):
        context = dict(family="ensemble_dwd", stage="options", base_position=None,
                       base_fit_calls_completed=0, base_score_calls_completed=0,
                       cached_base_learners=0, final_fit_calls_completed=0,
                       base_work_source="current")
        classes = selected = None
        try:
            with _stage("options"):
                self._validate_options()
            context["stage"] = "input_validation"
            with _stage("input_validation"):
                X, y = validate_data(self, X, y, accept_sparse=False, dtype=np.float64)
                check_classification_targets(y)
                classes, encoded = np.unique(y, return_inverse=True)
                if len(classes) < 2:
                    raise ValueError("EnsembleKernelDWD requires two classes; the input contains one class.")
                if len(classes) > 2:
                    raise ValueError("Only binary classification is supported. The input has more than two classes.")
            context["stage"] = "setup"
            n = len(y)
            selected_size = int(n * self.sv_selection_ratio)
            if selected_size < 2:
                raise IneligibleSelectionError(
                    "sv_selection_ratio selects fewer than two observations. "
                    "Increase the ratio or provide more training observations.")
            subset_size = max(2, int(n * self.theta))
            with _stage("kernel_parameters"):
                gamma, kernel_kws = self._kernel_parameters(X)
            context["stage"] = "sampling"
            with _stage("sampling"):
                subsets = _sample_subsets(encoded, subset_size, self.M,
                                          _private_random_state(self.random_state))
            context.update(stage="base_work", base_fit_calls_completed=None,
                           base_score_calls_completed=None, cached_base_learners=None,
                           base_work_source="unknown")
            base, cache_hit, cache_key, cache_status = self._base_result(
                X, y, classes, subsets, kernel_kws)
            if (uses_implementation(self, "_base_stage", EnsembleKernelDWD._base_stage)
                    and uses_implementation(self, "_base_result", EnsembleKernelDWD._base_result)):
                context.update(base_fit_calls_completed=0 if cache_hit else len(subsets),
                               base_score_calls_completed=0 if cache_hit else len(subsets),
                               cached_base_learners=len(subsets) if cache_hit else 0,
                               base_work_source="cache" if cache_hit else "current")
                _record_cache_result(cache_hit, cache_status, len(subsets) if cache_hit else 0)
            context["stage"] = "selection"
            with _stage("selection"):
                consensus = base["consensus"]
                selected = base["order"][:selected_size].copy()
                if len(np.unique(encoded[selected])) != 2:
                    raise IneligibleSelectionError(
                        f"The {selected_size} selected observations contain only one class. "
                        "This candidate cannot fit the final binary DWD. Increase "
                        "sv_selection_ratio or use a different parameter candidate.")
            context["stage"] = "final_fit"
            with _stage("final_fit"):
                final_model = self._make_estimator(kernel_kws).fit(X[selected], y[selected])
            context["final_fit_calls_completed"] = 1
            context["stage"] = "final_validation"
            with _stage("final_validation"):
                if not np.array_equal(final_model.classes_, classes):
                    raise ValueError("The final DWD must use the same ordered binary labels.")
            context["stage"] = "finalization"
            self.classes_ = classes
            self.gamma_ = gamma
            self.affine_computation_ = self.affine_computation
            self.exact_recovery_ = self.exact_recovery
            self.kernel_kws_ = kernel_kws
            self.subset_indices_ = subsets
            self.selected_indices_ = selected
            self.consensus_scores_ = consensus
            self.estimators_ = base["models"]
            context["stage"] = "final_diagnostics"
            with _stage("final_diagnostics"):
                self.base_diagnostics_ = deepcopy(base["diagnostics"])
                self.n_iter_ = np.append(base["n_iter"], int(final_model.n_iter_))
                self.converged_ = np.append(base["converged"], bool(final_model.converged_))
            context["stage"] = "finalization"
            self.final_estimator_ = final_model
            self.cache_hit_ = cache_hit
            self.cache_key_ = cache_key
            self.cache_status_ = cache_status
            return self
        except BaseException as error:
            try:
                attach_failure_context(error, context, y=y, classes=classes, selected=selected)
            except Exception:
                pass
            raise

    def decision_function(self, X):
        """Return final-model scores, positive for classes_[1]."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, accept_sparse=False, dtype=np.float64)
        return _binary_scores(self.final_estimator_, X)

    def predict(self, X):
        """Return original labels using the final DWD's decision boundary."""
        scores = self.decision_function(X)
        return self.classes_[(scores > 0).astype(int)]
