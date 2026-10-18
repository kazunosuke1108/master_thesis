"""Post-hoc monotonicity audit; preserve the original REQ-004 analysis.

Risk inputs and consequents increase toward danger. With bilinear interpolation,
coordinate-wise monotonicity is equivalent to four nonnegative edge differences.
This audit is not an AHP CI or a validated psychological consistency scale.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path

GROUPS = ('internal', 'watching', 'external', 'total')
EDGES = (('LH_to_HH', 0, 2), ('LL_to_HL', 1, 3),
         ('HL_to_HH', 0, 1), ('LL_to_LH', 2, 3))
FLOAT_TOLERANCE = 1e-12  # Saved continuous values only; ordinal edge audit is exact.


def audit_rules(scores):
    if (not isinstance(scores, list) or len(scores) != 4 or
            any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 5 for v in scores)):
        raise ValueError('four ordinal integer scores 1..5 in HH/HL/LH/LL order required')
    edges = [{'edge': name, 'delta_score': scores[hi] - scores[lo],
              'deficit_score': max(0, scores[lo] - scores[hi])}
             for name, hi, lo in EDGES]
    return {'scores': scores, 'edges': edges,
            'violation_count': sum(e['delta_score'] < 0 for e in edges),
            'max_normalized_deficit': max(e['deficit_score'] for e in edges) / 4}


def derivatives(scores, x, y):
    hh, hl, lh, ll = [(v - 1) / 4 for v in scores]
    return y * (hh-lh) + (1-y) * (hl-ll), x * (hh-hl) + (1-x) * (lh-ll)


def aggregate(cases):
    asked = [r for r in cases if r['followup_status'] == 'answered']
    return {'case_count': len(cases),
            'exact_matches': sum(r['exact_rank_match'] is True for r in cases),
            'mismatches': sum(r['exact_rank_match'] is False for r in cases),
            'followup_case_count': len(asked),
            'helpfulness_4_or_5': sum(r['helpfulness'] in (4, 5) for r in asked),
            'acceptance_counts': dict(Counter(r['acceptance'] for r in asked)),
            'high_helpfulness_acceptance_counts': dict(Counter(r['acceptance'] for r in asked if r['helpfulness'] in (4, 5))),
            'pair_count': sum(len(r['pairs']) for r in asked),
            'pair_acceptance_counts': dict(Counter(p['acceptance'] for r in asked for p in r['pairs']))}


def analyze(root, destination):
    root, destination = Path(root).resolve(), Path(destination)
    if destination.exists():
        raise FileExistsError('Use a new output directory; original results stay unchanged')
    hashes = {}
    def read(path):
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    original = read(root/'final-summary/results.json')
    profiles, excluded = [], []
    for profile in original['profiles']:
        identity = profile['respondent_id']
        knowledge = read(root/'participants'/identity/'knowledge.json')
        groups = {g: audit_rules(knowledge['fuzzy_answers'][g]) for g in GROUPS}
        assert knowledge['fuzzy_answers'] == profile['fuzzy_answers']
        count = sum(v['violation_count'] for v in groups.values())
        if count:
            excluded.append(identity)
        negative, pairs = [], []
        for aggregation in ('weighted_sum', 'weighted_max'):
            snapshots = read(root/'analysis-v2'/aggregation/identity/'snapshots.json')
            for snap in snapshots:
                for group in GROUPS:
                    trace = snap['rule_trace'][group]
                    assert trace['raw_answers_1to5'] == knowledge['fuzzy_answers'][group]
                    x, y = trace['inputs']
                    dx, dy = derivatives(trace['raw_answers_1to5'], x, y)
                    if min(dx, dy) < -FLOAT_TOLERANCE:
                        negative.append({'aggregation': aggregation, 'case_id': snap['case_id'],
                                         'patient_id': snap['patient_id'], 'group': group,
                                         'inputs': [x, y], 'output': trace['output'],
                                         'd_output_dx': dx, 'd_output_dy': dy})
            for case in sorted({s['case_id'] for s in snapshots}):
                for a, b in itertools.combinations([s for s in snapshots if s['case_id'] == case], 2):
                    for group in GROUPS:
                        ta, tb = a['rule_trace'][group], b['rule_trace'][group]
                        for high, low, ht, lt in ((a, b, ta, tb), (b, a, tb, ta)):
                            delta = [u-v for u, v in zip(ht['inputs'], lt['inputs'])]
                            if (min(delta) >= -FLOAT_TOLERANCE and max(delta) > FLOAT_TOLERANCE and
                                    ht['output'] < lt['output'] - FLOAT_TOLERANCE):
                                pairs.append({'aggregation': aggregation, 'case_id': case, 'group': group,
                                              'higher_input_patient': high['patient_id'],
                                              'lower_input_patient': low['patient_id'],
                                              'higher_inputs': ht['inputs'], 'lower_inputs': lt['inputs'],
                                              'higher_input_output': ht['output'],
                                              'lower_input_output': lt['output']})
        profiles.append({'respondent_id': identity, 'cohort': profile['cohort'], 'groups': groups,
                         'violation_count_of_16': count,
                         'max_normalized_deficit': max(g['max_normalized_deficit'] for g in groups.values()),
                         'exclude_from_sensitivity': bool(count),
                         'negative_derivative_observations': negative, 'dominated_input_reversals': pairs})
    summaries = []
    for cohort in ('pilot2', 'derived14', 'all16'):
        for aggregation in ('weighted_sum', 'weighted_max'):
            rows = [r for r in original['cases'] if r['aggregation'] == aggregation and
                    (cohort == 'all16' or r['cohort'] == cohort)]
            summaries.append({'cohort': cohort, 'aggregation': aggregation,
                              'before': aggregate(rows),
                              'after': aggregate([r for r in rows if r['respondent_id'] not in excluded])})
    payload = {'request': 'REQ-004 revision3, post-hoc sensitivity only',
               'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
               'criterion': 'Any negative ordinal edge among four comparable edges of any of four groups; exclude the entire person in both aggregations and all scenes',
               'criterion_threshold': 0, 'ordinal_tolerance': 0, 'continuous_tolerance': FLOAT_TOLERANCE,
               'not_AHP_CI': True, 'not_published_MI_formula': True,
               'primary_analysis_replaced': False, 'answers_or_scores_modified': False,
               'excluded_respondents': excluded, 'profiles': profiles, 'summaries': summaries,
               'source_sha256': hashes,
               'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination.mkdir(parents=True)
    (destination/'sensitivity.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (destination/'analysis-script.py').write_bytes(Path(__file__).read_bytes())
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    result = analyze(args.root, args.destination)
    print(json.dumps({'excluded': result['excluded_respondents'], 'summaries': result['summaries']}, ensure_ascii=False, indent=2))
