"""Verify the post-hoc criterion against every possible four-answer rule table."""
import importlib.util
import itertools
from pathlib import Path

import pytest

from master_thesis_modules.risk_core.aggregators.fuzzy import LegacyLikeFuzzyAggregator
from master_thesis_modules.risk_core.schema import node_ids as ids

spec = importlib.util.spec_from_file_location(
    'fuzzy_sensitivity', Path(__file__).resolve().parents[2] /
    'studies/REQ-004/fuzzy_consistency_sensitivity.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize('scores', ([5, 2, 4, 1], [5, 4, 2, 1], [3, 3, 3, 3], [5, 5, 5, 1]))
def test_incomparable_preferences_and_equal_values_are_valid(scores):
    assert audit.audit_rules(scores)['violation_count'] == 0


@pytest.mark.parametrize('scores', ([5, 1, 4, 2], [5, 2, 5, 5], [5, 1, 5, 3]))
def test_actual_nonmonotone_patterns(scores):
    result = audit.audit_rules(scores)
    assert result['violation_count'] == 1
    assert next(e for e in result['edges'] if e['delta_score'] < 0)['edge'] == 'LL_to_HL'


@pytest.mark.parametrize('scores', ([True, 2, 3, 4], [5, 1, 2], [5, 0, 2, 1], [5., 2, 3, 1]))
def test_invalid_rule_tables_are_not_silently_repaired(scores):
    with pytest.raises(ValueError):
        audit.audit_rules(scores)


def test_all_625_rule_tables_match_all_four_implemented_fuzzy_nodes():
    stages = [
        (ids.INTERNAL_RISK, ids.INTERNAL_STATIC_RISK, ids.INTERNAL_DYNAMIC_RISK),
        (ids.EXTERNAL_DYNAMIC_RISK, ids.STAFF_DISTANCE_RISK, ids.STAFF_NOT_WATCHING_RISK),
        (ids.EXTERNAL_RISK, ids.EXTERNAL_STATIC_RISK, ids.EXTERNAL_DYNAMIC_RISK),
        (ids.TOTAL_RISK, ids.INTERNAL_RISK, ids.EXTERNAL_RISK)]
    for scores in itertools.product(range(1, 6), repeat=4):
        result = audit.audit_rules(list(scores))
        for output, first, second in stages:
            values = tuple((s - 1)/4 for s in scores)
            if output == ids.EXTERNAL_DYNAMIC_RISK:
                values = tuple(values[i] for i in (0, 2, 1, 3))
            aggregator = LegacyLikeFuzzyAggregator(output, values)
            def f(x, y):
                return aggregator.aggregate({first: x, second: y})
            edges = [f(1, 1)-f(0, 1), f(1, 0)-f(0, 0),
                     f(1, 1)-f(1, 0), f(0, 1)-f(0, 0)]
            assert sum(d < 0 for d in edges) == result['violation_count']
            for x, y in ((.2, .7), (.73, .19)):
                dx, dy = audit.derivatives(scores, x, y)
                assert dx == pytest.approx((f(x+.01, y)-f(x-.01, y))/.02, abs=1e-12)
                assert dy == pytest.approx((f(x, y+.01)-f(x, y-.01))/.02, abs=1e-12)


def test_unasked_not_counted_as_acceptance_and_axis_denominators_differ():
    cases = [
        {'exact_rank_match': True, 'followup_status': 'not_asked_agreement',
         'helpfulness': None, 'acceptance': None, 'pairs': []},
        {'exact_rank_match': False, 'followup_status': 'answered',
         'helpfulness': 4, 'acceptance': 'partial', 'pairs': [{'acceptance': 'reject'}]}]
    result = audit.aggregate(cases)
    assert result['case_count'] == 2 and result['exact_matches'] == 1
    assert result['followup_case_count'] == 1 and result['helpfulness_4_or_5'] == 1
    assert result['acceptance_counts'] == {'partial': 1}
    assert result['high_helpfulness_acceptance_counts'] == {'partial': 1}
    assert result['pair_acceptance_counts'] == {'reject': 1}
