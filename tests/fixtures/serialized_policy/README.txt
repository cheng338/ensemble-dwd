Trusted synthetic regression fixture, copied without refitting or repickling.
ensemble-actual.p5 contains the saved estimator. ensemble-actual.input.json
contains the input for the regression check. The same two files were used
on Windows and Linux. tests/test_serialized_policy_ensemble.py pins both
SHA-256 digests and verifies them before deserialization.
The saved object is a prior corrected-baseline Ensemble; it does not assert
compatibility with every historical release. Explicit affine auto remains
historical attribution; new fitting rejects that retired explicit alias.
