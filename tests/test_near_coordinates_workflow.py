"""Close-coordinate disclosure preserves search and recovery decisions."""
import json
import math
from pathlib import Path
import sys
import tempfile
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
import stage_reporting
from near_coordinates import disclose
import test_adaptive_search as search_tests
import test_adaptive_resume as resume_tests


class NearCoordinatesWorkflowTests(unittest.TestCase):
    def fixture(self, cls=search_tests.AdaptiveSearchTests):
        fixture = cls()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_report_precedes_dispatch_and_preserves_every_exact_candidate(self):
        fixture = self.fixture()
        grid = fixture.grid(reg=(1., 4., math.nextafter(4., math.inf)))
        expected = list(ParameterGrid(grid))
        original = search_tests.RecordedCheckpointedCV.search
        captured = []
        def checked(checkpoint, candidate_grid, *, stage='cv'):
            self.assertFalse(search_tests.ResponseSurfaceClassifier.events)
            plan = json.loads((fixture.run_dir / (stage + '-plan.json')).read_text())
            captured.append(plan['near_coordinates'])
            self.assertEqual(captured[-1]['whole_candidates']['close_distinct_pair_count'], 15)
            self.assertEqual(parameter_hash(list(ParameterGrid(candidate_grid))),
                             parameter_hash(expected))
            return original(checkpoint, candidate_grid, stage=stage)
        with patch.object(search_tests.RecordedCheckpointedCV, 'search', checked):
            outcome = fixture.execute(grid=grid, config=search.AdaptiveSearchConfig(
                max_candidates=45, refine=False))
        self.assertEqual(len(captured), 1)
        self.assertEqual(parameter_hash(outcome['cv_results']['params']), parameter_hash(expected))
        self.assertEqual(len({parameter_hash(row) for row in expected}), 45)

    def test_new_stage_excludes_historical_pairs(self):
        fixture = self.fixture()
        first = {'ensemble__lambd': 4., 'ensemble__gamma': 4.}
        second = {**first, 'ensemble__gamma': math.nextafter(4., math.inf)}
        third = {**first, 'ensemble__gamma': math.nextafter(second['ensemble__gamma'], math.inf)}
        proposals = [
            {'state': {}, 'review': None, 'phase': 'coarse', 'candidates': [first, second]},
            {'state': {}, 'review': None, 'phase': 'refine-1', 'candidates': [third]},
            {'state': {}, 'review': {'state': 'needs_review', 'stopping_reason': 'controlled_hold',
                                    'reasons': ['controlled hold']}, 'phase': 'review', 'candidates': []},
        ]
        expected = parameter_hash([first, second, third])
        with patch.object(search, 'next_search_stage', side_effect=proposals):
            result = fixture.execute(config=search.AdaptiveSearchConfig(refine=False))
        counts = [json.loads((fixture.run_dir / (name + '-plan.json')).read_text())[
            'near_coordinates']['whole_candidates']['close_distinct_pair_count']
                  for name in ('coarse', 'refine-1')]
        self.assertEqual(counts, [1, 2])
        self.assertEqual(parameter_hash(result['cv_results']['params']), expected)
        fits = [event for event in search_tests.ResponseSurfaceClassifier.events if event[0] == 'fit']
        self.assertEqual(len(fits), 15)

    def test_disclosure_failure_preserves_workload_and_search(self):
        fixture = self.fixture()
        config = search.AdaptiveSearchConfig(max_candidates=45, refine=False)
        baseline = fixture.execute(config=config)
        events = list(search_tests.ResponseSurfaceClassifier.events)
        with patch.object(search, 'disclose', side_effect=ValueError('controlled reporting error')), \
                warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            actual = fixture.execute(config=config)
        plan = json.loads((fixture.run_dir / 'coarse-plan.json').read_text())
        self.assertEqual(plan['near_coordinates']['reporting_status'], 'unavailable')
        self.assertEqual(plan['workload_plan']['candidate_count'], 45)
        self.assertEqual(plan['budget']['proposed_count'], 45)
        self.assertEqual(events, search_tests.ResponseSurfaceClassifier.events)
        for field in ('selection', 'review', 'policy_state'):
            self.assertEqual(baseline[field], actual[field])
        for field in ('params', 'mean_test_score', 'accuracy_evidence'):
            np.testing.assert_equal(baseline['cv_results'][field], actual['cv_results'][field])

    def test_reporting_does_not_hide_interrupt(self):
        fixture = self.fixture()
        with patch.object(search, 'disclose', side_effect=KeyboardInterrupt('report interrupted')):
            with self.assertRaisesRegex(KeyboardInterrupt, 'report interrupted'):
                fixture.execute()
        self.assertFalse(search_tests.ResponseSurfaceClassifier.events)

    def test_interrupted_resume_suppresses_same_stage_notice_and_preserves_folds(self):
        fixture = self.fixture(resume_tests.AdaptiveResumeTests)
        fixture.grid['ensemble__gamma'] = [1., 4., math.nextafter(4., math.inf)]
        resume_tests.InterruptibleSurface.fail_at = 5
        with patch.object(stage_reporting, 'print', create=True) as announce:
            with self.assertRaises(KeyboardInterrupt):
                fixture.run_search()
            first_plan = fixture.read('coarse-plan.json')
            completed = list(resume_tests.InterruptibleSurface.attempts[:4])
            initial_calls = list(announce.call_args_list)
            notice_calls = [call for call in initial_calls if 'Close coordinates:' in call.args[0]]
            self.assertEqual(len(notice_calls), 1)
            announce.reset_mock()
            resume_tests.InterruptibleSurface.fail_at = None
            result = fixture.run_search(resume=True)
            self.assertFalse(any('Close coordinates:' in call.args[0] for call in announce.call_args_list))
        self.assertEqual(first_plan['near_coordinates'], fixture.read('coarse-plan.json')['near_coordinates'])
        for attempt in completed:
            self.assertEqual(resume_tests.InterruptibleSurface.attempts.count(attempt), 1)
        before_replay = (len(resume_tests.InterruptibleSurface.attempts),
                         len(resume_tests.InterruptibleSurface.predictions))
        replay = fixture.run_search(resume=True)
        self.assertEqual(result['selection'], replay['selection'])
        self.assertEqual(before_replay, (len(resume_tests.InterruptibleSurface.attempts),
                                        len(resume_tests.InterruptibleSurface.predictions)))

    def test_missing_or_malformed_supplementary_report_is_non_authoritative(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'stage-plan.json'
            report = {'near_coordinates': disclose(
                [{'gamma': 1.}, {'gamma': math.nextafter(1., math.inf)}], ('gamma',))}
            with patch.object(stage_reporting, 'print', create=True) as announce:
                stage_reporting.publish_report(path, lambda: report, announce=True)
                self.assertEqual(announce.call_count, 1)
                stage_reporting.publish_report(path, lambda: report, announce=True)
                self.assertEqual(announce.call_count, 1)
                path.write_text('invalid JSON', encoding='utf-8')
                stage_reporting.publish_report(path, lambda: report, announce=True)
                self.assertEqual(announce.call_count, 2)
            self.assertEqual(json.loads(path.read_text()), report)

    def test_notice_console_failure_and_warning_filter_do_not_fail_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'stage-plan.json'
            report = {'near_coordinates': disclose(
                [{'gamma': 1.}, {'gamma': math.nextafter(1., math.inf)}], ('gamma',))}
            with patch.object(stage_reporting, 'print', create=True, side_effect=OSError('console closed')), \
                    warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                self.assertEqual(stage_reporting.publish_report(path, lambda: report, announce=True), report)
            self.assertEqual(json.loads(path.read_text()), report)


if __name__ == '__main__':
    unittest.main()
