"""Recovery reporting preserves current, historical and failed-load accounting."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from work_reporting import aggregate_work_summaries, summarize_records


def record(failed_load):
    return {'state': 'complete', 'score': .75, 'fit_time': 2., 'score_time': .25,
            'diagnostics': {}, 'pair_recovery': {
                'schema': 1, 'fitted_pairs': 1, 'recovered_pairs': 1,
                'fitted_unsaved_pairs': 1, 'current_pair_fit_seconds': .5,
                'pair_load_seconds': .02, 'pair_save_seconds': 0.,
                'pair_failed_load_seconds': failed_load,
                'historical_reused_pair_fit_seconds': 7., 'cleanup_status': 'cleaned'}}


@pytest.mark.parametrize('replayed', [False, True])
def test_failed_load_belongs_to_its_current_or_historical_invocation(replayed):
    row = record(.125)
    before = deepcopy(row)
    value = summarize_records([(row, replayed)])
    work = value['pair_recovery']
    current = work['current_worker_seconds']
    historical = work['replayed_receipts']['worker_seconds']
    assert current.get('pair_failed_load', 0.) == (0. if replayed else .125)
    assert historical.get('pair_failed_load', 0.) == (.125 if replayed else 0.)
    assert current['pair_load'] == (0. if replayed else .02)
    assert historical['pair_load'] == (.02 if replayed else 0.)
    assert value['current_worker_seconds']['fit'] == (0. if replayed else 2.)
    assert row == before


def test_aggregation_keeps_recovery_cost_separate_without_changing_total_fit():
    first = summarize_records([(record(.125), False)])
    replay = summarize_records([(record(.25), True)])
    before = deepcopy([first, replay])
    value = aggregate_work_summaries([first, replay])
    assert value['pair_recovery']['current_worker_seconds']['pair_failed_load'] == .125
    assert value['pair_recovery']['replayed_receipts']['worker_seconds']['pair_failed_load'] == .25
    assert value['current_worker_seconds']['fit'] == 2.
    assert [first, replay] == before


def test_ordinary_and_older_summaries_do_not_acquire_spurious_recovery_work():
    ordinary = record(0.)
    older = deepcopy(ordinary)
    del older['pair_recovery']['pair_failed_load_seconds']
    a = summarize_records([(ordinary, False)])
    b = summarize_records([(older, False)])
    assert a == b
    combined = aggregate_work_summaries([a, b])
    assert 'pair_failed_load' not in combined['pair_recovery']['current_worker_seconds']
    assert 'pair_failed_load' not in combined['pair_recovery']['replayed_receipts']['worker_seconds']
