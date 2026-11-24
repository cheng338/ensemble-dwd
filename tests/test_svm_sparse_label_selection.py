"""Signed-margin selection must agree across equivalent class encodings."""

import numpy as np
import pandas as pd
import pytest

from ensemble_dwd import EnsembleKernelSVM, IneligibleSelectionError


def test_sparse_text_labels_preserve_signed_margin_selection_and_predictions(tmp_path):
    X = np.array([-3., 1., 2., 2., -1., 3., -1.5, 4.])[:, None]
    codes = np.array([0, 1] * 4)
    labels = np.array(["q", "q\0"], dtype=object)
    y = pd.arrays.SparseArray(labels[codes], fill_value=labels[0])
    options = dict(M=1, theta=1., kernel="linear", sv_selection_ratio=.5, random_state=1)
    control = EnsembleKernelSVM(**options).fit(X, codes)
    distances = np.abs((2 * codes - 1) * control.consensus_scores_ - 1.)
    selected = np.argsort(distances, kind="stable")[:4]
    queries = np.vstack((X, [[0.]]))
    expected_labels = labels[control.predict(queries)]

    for expected_hit in (False, True):
        model = EnsembleKernelSVM(cache_dir=tmp_path, **options).fit(X, y)
        assert model.cache_hit_ is expected_hit, model.cache_status_
        np.testing.assert_array_equal(model.consensus_scores_, control.consensus_scores_)
        np.testing.assert_array_equal(model.selection_distances_, distances)
        np.testing.assert_array_equal(model.selected_indices_, selected)
        assert model.predict(queries).tolist() == expected_labels.tolist()


def test_sparse_text_signed_margin_rejects_one_class_selection(tmp_path):
    X = np.array([-2., 2., -2.5, 2.5, -3., 3., -1.8, 1.8])[:, None]
    codes = np.array([0, 1] * 4)
    y = pd.arrays.SparseArray(np.array(["k", "k\0"] * 4, dtype=object), fill_value="k")
    options = dict(M=1, theta=1., random_state=9)
    full = EnsembleKernelSVM(sv_selection_ratio=1., **options).fit(X, codes)
    distances = np.abs((2 * codes - 1) * full.consensus_scores_ - 1.)
    selected = np.argsort(distances, kind="stable")[:2]
    assert len(np.unique(codes[selected])) == 1

    for cache_dir in (None, tmp_path):
        if cache_dir is not None:
            warmup = EnsembleKernelSVM(cache_dir=cache_dir, sv_selection_ratio=1., **options)
            warmup.fit(X, y)
            assert not warmup.cache_hit_
        model = EnsembleKernelSVM(cache_dir=cache_dir, sv_selection_ratio=.25, **options)
        with pytest.raises(IneligibleSelectionError, match="only one class") as error:
            model.fit(X, y)
        context = error.value.ensemble_failure_context
        assert context["base_work_source"] == ("current" if cache_dir is None else "cache")
        assert context["selected_class_counts"] == np.bincount(codes[selected], minlength=2).tolist()
