"""Validation shared by real-valued estimators and their examples."""

import numpy as np
from scipy.sparse import issparse


def _label_value(value):
    """Return a scalar for reporting, preserving the payload of boxed NumPy text."""
    if isinstance(value, np.str_):
        return str.__getitem__(value, slice(None))
    if isinstance(value, np.bytes_):
        return bytes.__getitem__(value, slice(None))
    return value.item() if isinstance(value, np.generic) else value


def _class_mask(y, label):
    """Compare rows to one representative in the array's label domain."""
    if y.dtype.kind in "OT":
        # A scalar text operand can be inferred as fixed-width and lose
        # its trailing NULs. Assign a single element to avoid invoking
        # array conversion on a custom object representative.
        representative = np.empty(1, dtype=y.dtype)
        representative[0] = label
        return y == representative
    return y == label


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
