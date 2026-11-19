"""Fresh-process, real-fit observation of lazy native imports; no timing claims."""
import argparse
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--family", choices=("dwd", "svm"), default="dwd")
    parser.add_argument("--kernel", choices=("rbf", "linear"), default="rbf")
    parser.add_argument("--backend", default="cholesky")
    parser.add_argument("--preimport", action="store_true")
    args = parser.parse_args()

    import numpy as np
    from threadpoolctl import threadpool_info
    from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM
    from ensemble_dwd import _external_identity as external

    if args.preimport:
        importlib.import_module("scipy.io")
    X = np.column_stack((np.arange(24, dtype=float) - 11.5,
                         np.arange(24, dtype=float) % 3)) / 6.0
    y = np.zeros(24, dtype=int)
    # Both seed-17 disjoint base subsets contain the two classes.
    y[[10, 23]] = 1
    cls = EnsembleKernelDWD if args.family == "dwd" else EnsembleKernelSVM
    settings = dict(M=2, theta=.5, sv_selection_ratio=1., gamma=.2,
                    kernel=args.kernel, random_state=17, cache_dir=args.cache)
    if args.family == "dwd":
        settings.update(lambd=.05, backend=args.backend, max_iter=8, stopping="fixed")
    else:
        settings.update(C=1.0, max_iter=1000)
    original = cls._base_stage
    work = []

    def counted(self, *positional, **keywords):
        work.append(int(self.M))
        return original(self, *positional, **keywords)

    cls._base_stage = counted
    models, records = [], []

    def fit(label):
        before = len(work)
        model = cls(**settings).fit(X, y)
        models.append(model)
        records.append(dict(label=label, hit=bool(model.cache_hit_), key=model.cache_key_,
                            status=model.cache_status_, base_stages=len(work) - before,
                            base_fits=sum(work[before:]),
                            predictions=model.predict(X).tolist(),
                            consensus=model.consensus_scores_.tolist(),
                            selected=model.selected_indices_.tolist(),
                            subsets=[part.tolist() for part in model.subset_indices_],
                            diagnostics=model.base_diagnostics_,
                            final_is_fresh=all(model.final_estimator_ is not prior.final_estimator_
                                               for prior in models[:-1])))
        return model

    fit("initial")
    fit("unchanged")
    before_modules = set(external._EXTERNAL_IDENTITY.modules or {})
    before_pools = threadpool_info()
    importlib.import_module("scipy.io")
    observed = external._observed_artifacts(threadpool_info())
    added = sorted(set(observed[2]) - before_modules)
    fit("after_import")
    fit("stable_after_import")
    cls._base_stage = original
    for model in models[1:]:
        np.testing.assert_array_equal(model.predict(X), models[0].predict(X))
        np.testing.assert_array_equal(model.selected_indices_, models[0].selected_indices_)
        np.testing.assert_allclose(model.consensus_scores_, models[0].consensus_scores_, rtol=0., atol=0.)
    result = dict(family=args.family, kernel=args.kernel, backend=args.backend,
                  preimport=args.preimport, added_modules=added, records=records,
                  pools_before=before_pools, pools_after=threadpool_info(),
                  identity_failure=external._EXTERNAL_IDENTITY.failure,
                  artifact_count=len(external._EXTERNAL_IDENTITY.artifacts or {}),
                  cache_entries=len(list(Path(args.cache).rglob("*.joblib"))))
    Path(args.result).write_text(json.dumps(result, sort_keys=True, default=lambda x: x.tolist()),
                                 encoding="utf-8")


if __name__ == "__main__":
    main()
