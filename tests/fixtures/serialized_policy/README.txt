Trusted synthetic regression fixture, copied without refitting or repickling.
ensemble-actual.p4 and ensemble-actual.p5 are the protocol-4 and protocol-5
copies of the same fitted estimator, saved in the original NumPy 2.5.4 run.
ensemble-actual.input.json contains the input for the regression check.
The test loads the protocol-5 file where supported. NumPy 1.26.4 can read
the protocol-4 file when it lacks the protocol-5 array reconstruction module.
tests/test_serialized_policy_ensemble.py checks all three SHA-256 digests
before deserialization.
The saved object is a prior corrected-baseline Ensemble; it does not assert
compatibility with every historical release. Explicit affine auto remains
historical attribution; new fitting rejects that retired explicit alias.
