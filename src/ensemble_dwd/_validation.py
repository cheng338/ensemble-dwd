"""Validation shared by real-valued estimators and their examples."""

import numpy as np
from scipy.sparse import issparse


def _reject_complex(value, name):
    """Reject complex scalars even when an object array hides their dtype.

    Inspect before a float conversion can discard imaginary components. The
    input is not replaced, so feature names and ordinary validation survive.
    """
    def reject():
        raise ValueError(f"Complex data not supported: {name} must be real values.")

    # NumPy may iterate older SciPy sparse arrays instead of treating them as
    # opaque objects. Reject them without conversion; these APIs require dense
    # inputs and decision values, even when their entries are real.
    if issparse(value):
        if value.dtype.kind == "c":
            reject()
        raise TypeError(f"Sparse data not supported: {name} must be dense values.")

    array = np.asarray(value)
    if array.dtype.kind == "c":
        reject()
    if array.dtype.kind != "O":
        return
    # Object arrays can contain zero-dimensional object arrays which float()
    # would unwrap and cast. An explicit stack also handles cyclic containers
    # without recursion or repeated traversal.
    pending, visited = [array], set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        for item in current.flat:
            if (isinstance(item, (complex, np.complexfloating))
                    or (isinstance(item, np.ndarray) and item.dtype.kind == "c")):
                reject()
            if isinstance(item, np.ndarray) and item.dtype.kind == "O":
                pending.append(item)
