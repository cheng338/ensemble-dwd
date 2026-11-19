"""An unavailable DWD bound remains explicit in lean ensemble diagnostics."""
import json
import numpy as np
from dwd.gen_kern_dwd import KernGDWD


def two_by_two():
    t = 1. + 2.**-45
    return np.array([[1., t], [t, 1.]]), np.array([-1., 1.])


def unavailable(info):
    assert info['dual_objective_is_lower_bound'] is False
    assert info['fallback_reason'] in ('certified_negative_dual_quadratic', 'negative_constant_kernel')
    assert info['kernel_assumption'] == 'symmetric positive-semidefinite represented kernel (contradicted)'


def test_ensemble_lean_summary_preserves_unavailable_bound():
    from ensemble_dwd.estimator import _base_diagnostics
    K,y=two_by_two()
    model=KernGDWD(kernel='precomputed',lambd=1.,max_iter=2,stopping='fixed').fit(K,y)
    summary=_base_diagnostics(model)
    assert summary['dual_gap'] is None
    nested=summary['solver_diagnostics']['dual_certificate']
    unavailable(nested)
    assert nested['dual_objective_lower_bound'] is None
    decoded=json.loads(json.dumps(summary,allow_nan=False))
    assert decoded['dual_gap'] is None
    stored=decoded['solver_diagnostics']['dual_certificate']
    assert stored['dual_objective_lower_bound'] is None
    assert stored['dual_objective_is_lower_bound'] is False
