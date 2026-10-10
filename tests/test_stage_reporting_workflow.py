"""Supplementary stage plans precede dispatch and cannot change search work."""
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from joblib import hash as parameter_hash
from sklearn.model_selection import ParameterGrid

EXAMPLES = str(Path(__file__).resolve().parents[1] / 'examples')
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)
import adaptive_search as search
import ratio_refinement
import stage_reporting
import test_adaptive_search as search_tests
import test_adaptive_resume as resume_tests
from test_adaptive_search import RecordedCheckpointedCV, ResponseSurfaceClassifier
from test_adaptive_resume import InterruptibleSurface
from test_adaptive_policy import grid, result


class StageReportingWorkflowTests(unittest.TestCase):
    def fixture(self, cls=None):
        fixture = (search_tests.AdaptiveSearchTests if cls is None else cls)()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_plan_exists_before_first_cv_dispatch(self):
        fixture = self.fixture()
        original = RecordedCheckpointedCV.search
        plans = []
        def checked(checkpoint, candidate_grid, *, stage='cv'):
            self.assertFalse(ResponseSurfaceClassifier.events)
            plan = json.loads((fixture.run_dir/(stage+'-plan.json')).read_text())
            plans.append(plan)
            self.assertEqual(plan['workload_plan']['requested_pipeline_evaluations'], 225)
            self.assertEqual(plan['budget']['remaining_before'], 45)
            self.assertEqual(plan['budget']['remaining_after_admission'], 0)
            # The custom test estimator is deliberately not guessed from M.
            self.assertIsNone(plan['workload_plan']['totals']['total_learner_fits'])
            return original(checkpoint, candidate_grid, stage=stage)
        with patch.object(RecordedCheckpointedCV, 'search', checked):
            outcome = fixture.execute(config=search.AdaptiveSearchConfig(max_candidates=45, refine=False))
        self.assertEqual(len(plans), 1)
        fixture.assert_protocol(outcome, final=True)

    def test_build_and_write_failures_preserve_fit_and_prediction_sequence(self):
        fixture = self.fixture()
        config = search.AdaptiveSearchConfig(max_candidates=45, refine=False)
        baseline = fixture.execute(config=config)
        events = list(ResponseSurfaceClassifier.events)
        for module, name in ((search, 'workload_plan'), (stage_reporting, 'write_json_atomic')):
            with self.subTest(failure=name), warnings.catch_warnings(), \
                 patch.object(module, name, side_effect=OSError('controlled supplementary failure')):
                warnings.simplefilter('ignore', RuntimeWarning)
                outcome = fixture.execute(config=config)
            self.assertEqual(events, ResponseSurfaceClassifier.events)
            self.assertEqual(baseline['selection'], outcome['selection'])
            self.assertEqual(baseline['review'], outcome['review'])
            for name in ('params', 'mean_test_score', 'accuracy_evidence'):
                np.testing.assert_equal(baseline['cv_results'][name], outcome['cv_results'][name])
            fixture.assert_protocol(outcome, final=True)

    def test_refused_stage_details_spend_no_budget_and_launch_no_fit(self):
        fixture = self.fixture()
        outcome = fixture.execute(config=search.AdaptiveSearchConfig(max_candidates=1))
        saved = json.loads((fixture.run_dir/'coefficient-refused-stage.json').read_text())
        self.assertEqual(saved['budget']['proposed_count'], 45)
        self.assertEqual(saved['budget']['remaining_before'], 1)
        self.assertEqual(saved['budget']['remaining_after_admission'], 1)
        self.assertEqual(saved['proposal_report']['proposal']['new_candidate_count'], 45)
        self.assertEqual(saved['refused_stage_fits_dispatched'], 0)
        self.assertFalse(ResponseSurfaceClassifier.events)
        self.assertEqual(outcome['review']['state'], 'needs_review')
        self.assertIsNone(outcome['model'])

    def test_reporting_failure_preserves_original_fit_error(self):
        fixture = self.fixture()
        with warnings.catch_warnings(), \
             patch.object(stage_reporting, 'write_json_atomic', side_effect=OSError('report failed')):
            warnings.simplefilter('ignore', RuntimeWarning)
            with self.assertRaisesRegex(FloatingPointError, 'controlled unexpected numerical failure'):
                fixture.execute(failure='numerical')

    def test_interrupted_and_completed_resume_do_not_duplicate_work(self):
        fixture = self.fixture(resume_tests.AdaptiveResumeTests)
        InterruptibleSurface.fail_at = 5
        with self.assertRaises(KeyboardInterrupt):
            fixture.run_search()
        first_plan = fixture.read('coarse-plan.json')
        completed = InterruptibleSurface.attempts[:4]
        InterruptibleSurface.fail_at = None
        resumed = fixture.run_search(resume=True)
        for attempt in completed:
            self.assertEqual(InterruptibleSurface.attempts.count(attempt), 1)
        self.assertEqual(len(InterruptibleSurface.attempts), 47)
        self.assertEqual(first_plan['workload_plan']['requested_pipeline_evaluations'],
                         fixture.read('coarse-plan.json')['workload_plan']['requested_pipeline_evaluations'])
        counts = (len(InterruptibleSurface.attempts), len(InterruptibleSurface.predictions))
        completed_resume = fixture.run_search(resume=True)
        self.assertTrue(completed_resume['resumed_completed'])
        self.assertEqual(resumed['completed'], completed_resume['completed'])
        self.assertEqual(counts, (len(InterruptibleSurface.attempts), len(InterruptibleSurface.predictions)))

    def test_refusal_metadata_fault_does_not_change_authoritative_policy_fields(self):
        args = (result([]), grid(), search.AdaptiveSearchConfig(max_candidates=1))
        expected = search.next_search_stage(*args)
        with patch.object(search, 'grid_proposal', side_effect=RuntimeError('report failure')):
            actual = search.next_search_stage(*args)
        self.assertEqual(actual['proposal_report']['reporting_status'], 'unavailable')
        for key in ('state', 'phase', 'candidates', 'review'):
            self.assertEqual(parameter_hash(expected[key]), parameter_hash(actual[key]))

    def test_ratio_parameter_limit_reports_partial_evidence_without_dispatch(self):
        import test_ratio_policy as ratio_tests
        initial = ratio_tests.grid(ratios=(.5, math.nextafter(.5, 1.)))
        rows = [(p, ratio_tests.surface(p)) for p in ParameterGrid(initial)]
        outcome = ratio_refinement.next_ratio_stage(
            ratio_tests.result(rows), initial, search.AdaptiveSearchConfig(top_k=1),
            ratio_refinement.RatioRefinementConfig(ratio_tol=.1))
        proposal = outcome['proposal_report']['proposal']
        self.assertEqual(outcome['candidates'], [])
        self.assertEqual(outcome['review']['state'], 'needs_review')
        self.assertIsNone(proposal['new_candidate_count'])
        self.assertFalse(proposal['candidate_list_complete'])
        self.assertTrue(proposal['parameter_limits'])


if __name__ == '__main__':
    unittest.main()
