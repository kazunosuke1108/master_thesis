"""REQ-004: immutable initial comparison and separately bound follow-up answers.

Pair decomposition describes the implemented calculation, not the respondent's
psychology. All 32 hybrids are diagnostic inputs, never additional observations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import sys

from .study import (_csv, _identifier, _json, _provenance, _scene, _sha,
                    _write_new, load_cases, validate_responses)
from master_thesis_modules.risk_core.aggregators.fuzzy import LegacyLikeFuzzyAggregator
from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine
from master_thesis_modules.risk_core.schema import node_ids as ids

GROUPS = ('patient_attributes', 'patient_action', 'objects', 'staff_distance', 'outside_view')
LABELS = ('患者属性', '患者行動', '周囲物体', 'スタッフとの距離', '視野外である程度')
RULE_NODES = {'internal': ids.INTERNAL_RISK, 'watching': ids.EXTERNAL_DYNAMIC_RISK,
              'external': ids.EXTERNAL_RISK, 'total': ids.TOTAL_RISK}
CONDITIONS = ('HH', 'HL', 'LH', 'LL')
ACCEPTANCE = ('accept', 'partial', 'reject', 'uncertain')
QUESTION = ('アンケートで重視したことと、この場面で順位を決めたときに重視したことに'
            '違いはありましたか。違いがない場合も含め、算出された順をどう考えるか、理由を教えてください。')
RULE_DESCRIPTIONS = {
    'internal': ('患者属性のリスク', '患者行動のリスク'),
    'watching': ('スタッフとの距離によるリスク', 'スタッフの視野外であることによるリスク'),
    'external': ('周囲物体のリスク', '見守り状況のリスク'),
    'total': ('患者文脈のリスク', '空間的文脈のリスク'),
}


def risk_groups(result):
    """Actual five inputs immediately below the personalized Fuzzy hierarchy."""
    return (result.upper_risks[ids.INTERNAL_STATIC_RISK],
            result.upper_risks[ids.INTERNAL_DYNAMIC_RISK],
            result.upper_risks[ids.EXTERNAL_STATIC_RISK],
            result.factor_risks[ids.STAFF_DISTANCE_RISK],
            result.factor_risks[ids.STAFF_NOT_WATCHING_RISK])


def score_groups(config, values):
    """Replay spatial-context RiskEngine with unchanged lower-stage values."""
    if len(values) != 5 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError('five finite lower risks in [0,1] required')
    if config.model_type != 'spatial_context' or not config.use_legacy_like_fuzzy:
        raise ValueError('personalized spatial_context Fuzzy model required')
    s, a, o, d, v = values
    def f(node, inputs):
        if node not in config.fuzzy_rule_results:
            raise ValueError('all four personalized rule sets required')
        return LegacyLikeFuzzyAggregator(node, config.fuzzy_rule_results[node]).aggregate(inputs)
    i = f(ids.INTERNAL_RISK, {ids.INTERNAL_STATIC_RISK: s, ids.INTERNAL_DYNAMIC_RISK: a})
    w = f(ids.EXTERNAL_DYNAMIC_RISK, {ids.STAFF_DISTANCE_RISK: d, ids.STAFF_NOT_WATCHING_RISK: v})
    e = f(ids.EXTERNAL_RISK, {ids.EXTERNAL_STATIC_RISK: o, ids.EXTERNAL_DYNAMIC_RISK: w})
    return f(ids.TOTAL_RISK, {ids.INTERNAL_RISK: i, ids.EXTERNAL_RISK: e})


def decompose_pair(config, first, second):
    """Exact baseline-second -> first Shapley decomposition (five groups)."""
    if len(first) != 5 or len(second) != 5:
        raise ValueError('five groups required')
    hybrids = {mask: score_groups(config, tuple(first[j] if mask & (1 << j) else second[j]
                                               for j in range(5))) for mask in range(32)}
    contributions = {}
    for j, name in enumerate(GROUPS):
        value = 0.0
        for mask in range(32):
            if mask & (1 << j):
                continue
            n = mask.bit_count()
            weight = math.factorial(n) * math.factorial(4 - n) / math.factorial(5)
            value += weight * (hybrids[mask | (1 << j)] - hybrids[mask])
        contributions[name] = value
    delta = hybrids[31] - hybrids[0]
    if not math.isclose(sum(contributions.values()), delta, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('decomposition additivity failure')
    return {'baseline': 'second_patient', 'delta_first_minus_second': delta,
            'contributions': contributions, 'hybrid_scores': hybrids,
            'diagnostic_only': True, 'human_causal_inference': False}



def trace_rules(config, knowledge, result):
    """Actual four personalized stages with canonical answer and local rule order."""
    upper, lower = result.upper_risks, result.factor_risks
    inputs = {'internal': (upper[ids.INTERNAL_STATIC_RISK], upper[ids.INTERNAL_DYNAMIC_RISK]),
              'watching': (lower[ids.STAFF_DISTANCE_RISK], lower[ids.STAFF_NOT_WATCHING_RISK]),
              'external': (upper[ids.EXTERNAL_STATIC_RISK], upper[ids.EXTERNAL_DYNAMIC_RISK]),
              'total': (upper[ids.INTERNAL_RISK], upper[ids.EXTERNAL_RISK])}
    trace = {}
    for group, node in RULE_NODES.items():
        x, y = inputs[group]
        activations = (x*y, x*(1-y), (1-x)*y, (1-x)*(1-y))
        local_order = ('HH', 'LH', 'HL', 'LL') if group == 'watching' else CONDITIONS
        answers = knowledge['fuzzy_answers'][group]
        trace[group] = {'input_labels': RULE_DESCRIPTIONS[group], 'inputs': [x, y],
                        'output': upper[node], 'canonical_order': CONDITIONS,
                        'raw_answers_1to5': answers, 'canonical_activations': activations,
                        'canonical_representatives': [(a-1)/4 for a in answers],
                        'local_rule_order': local_order,
                        'local_representatives': config.fuzzy_rule_results[node]}
    return trace


def _relation(a, b):
    return 'higher' if a < b else 'lower' if a > b else 'tied'


def _rank_text(ranks):
    return '、'.join('＝'.join(sorted(p for p in ranks if ranks[p] == n))
                     for n in sorted(set(ranks.values())))


def _referenced_answers(config, knowledge, first, second):
    """Select actually active answers on changed calculation paths for readers.

    Full numerical activations and all source answers remain in researcher trace.
    Selection uses actual per-patient rule weights, not hypothetical scores.
    """
    left, right = trace_rules(config, knowledge, first), trace_rules(config, knowledge, second)
    changed = [a != b for a, b in zip(risk_groups(first), risk_groups(second))]
    relevant = []
    if changed[0] or changed[1]:
        relevant.append('internal')
    if changed[3] or changed[4]:
        relevant.append('watching')
    if changed[2] or changed[3] or changed[4]:
        relevant.append('external')
    if any(changed):
        relevant.append('total')
    lines, references = [], []
    conditions = (("高い", "高い"), ("高い", "低い"), ("低い", "高い"), ("低い", "低い"))
    for group in relevant:
        a, b = RULE_DESCRIPTIONS[group]
        # Cite the strongest actual applications whose weights differ between
        # patients; zero-activation and equal-weight rules do not distinguish them.
        weights_a = left[group]['canonical_activations']
        weights_b = right[group]['canonical_activations']
        indices = [j for j in range(4) if weights_a[j] != weights_b[j]]
        indices.sort(key=lambda j: abs(weights_a[j] - weights_b[j]), reverse=True)
        selected = indices[:2]
        if not selected:
            continue
        examples = []
        for j in selected:
            x, y = conditions[j]
            if group == 'watching':
                first_condition = 'スタッフが遠い' if x == '高い' else 'スタッフが近い'
                second_condition = 'スタッフの視野から外れている' if y == '高い' else 'スタッフの視野に入っている'
                condition_text = first_condition + '・' + second_condition
            else:
                condition_text = f'{a}が{x}・{b}が{y}'
            value = knowledge['fuzzy_answers'][group][j]
            examples.append(f'「{condition_text}」ときは危険性を5段階の{value}')
            references.append({'group': group, 'canonical_condition': CONDITIONS[j],
                               'raw_answer': value, 'knowledge_pointer': f'/fuzzy_answers/{group}/{j}',
                               'actual_activation_first': weights_a[j], 'actual_activation_second': weights_b[j],
                               'source_metadata': {k: knowledge[k] for k in ('source_hashes', 'knowledge_origin', 'source_question_ids') if k in knowledge}})
        lines.append('アンケートでは、' + '、'.join(examples) + 'と回答していました。')
    return lines, references


def _actual_conditions(case, first_id, second_id):
    scene = _scene(case)
    patient_info = {p['patient_id']: p for p in scene['patients']}
    frames = {case['mapping'][f.person_id]: f for f in case['frames']}
    descriptions = []
    for patient_id in (first_id, second_id):
        frame = frames[patient_id]
        parts = [f"{patient_id}は{patient_info[patient_id]['age_group']}で、{patient_info[patient_id]['action']}"]
        for label, position in (('車椅子', frame.nearest_wheelchair_position),
                                ('点滴台', frame.nearest_iv_position),
                                ('手すり', frame.nearest_handrail_position)):
            if position is not None:
                parts.append(f'{label}まで{frame.patient_position.distance_to(position):.1f}m')
        if frame.nearest_staff_position is None:
            parts.append('場面にスタッフはいません')
        else:
            distance = frame.patient_position.distance_to(frame.nearest_staff_position)
            parts.append(f'最も近いスタッフまで{distance:.1f}m')
            velocity = frame.nearest_staff_velocity
            if velocity is None or velocity.norm == 0 or distance == 0:
                parts.append('見守る方向が定まらない状態です')
            else:
                relative_x = frame.patient_position.x - frame.nearest_staff_position.x
                relative_y = frame.patient_position.y - frame.nearest_staff_position.y
                dot = relative_x * velocity.vx + relative_y * velocity.vy
                location = '正面側' if dot > 0 else '背後側' if dot < 0 else '横方向'
                parts.append(f'場面に示されたスタッフの向きを基準に{location}にあります')
        descriptions.append('、'.join(parts) + '。')
    return ' '.join(descriptions)


def _object_comparison(config, knowledge, first, second):
    """Cite an actual comparison between object types distinguishing this pair."""
    objects = ((ids.IV_POLE_RISK, '点滴台'), (ids.WHEELCHAIR_RISK, '車椅子'),
               (ids.HANDRAIL_DISTANCE_RISK, '手すり'))
    differences = [config.object_weights[node] * (first.factor_risks[node] - second.factor_risks[node])
                   for node, _ in objects]
    positive = [j for j, value in enumerate(differences) if value > 0]
    negative = [j for j, value in enumerate(differences) if value < 0]
    if positive and negative:
        i = max(positive, key=lambda j: abs(differences[j]))
        j = max(negative, key=lambda j: abs(differences[j]))
    else:
        different = sorted(range(3), key=lambda j: abs(differences[j]), reverse=True)
        if not differences[different[0]]:
            return '', []
        i, j = different[:2]
    value = knowledge['object_pairwise'][i][j]
    a, b = objects[i][1], objects[j][1]
    if value == 1:
        wording = f'{a}と{b}を同程度に重視する'
    elif value > 1:
        wording = f'{b}より{a}を重視する'
    else:
        wording = f'{a}より{b}を重視する'
    text = (f'物体の比較設問では「{wording}」側の回答でした（比較値{value:g}）。'
            'システムはこの比較回答も使って、物体との距離の違いをまとめています。')
    reference = {'knowledge_pointer': f'/object_pairwise/{i}/{j}', 'raw_answer': value,
                 'compared_objects': [a, b], 'meaning': 'pairwise preference, not risk ratio'}
    return text, [reference]


def _direction_bridge(first_id, second_id, decomposition, first_rank, second_rank):
    """Translate patient-B-baseline decomposition signs without claiming human causes."""
    directions = []
    terms = ('患者属性', '患者行動', '周囲物体', 'スタッフとの距離', 'スタッフの見守る方向')
    for name, term in zip(GROUPS, terms):
        contribution = decomposition['contributions'][name]
        if abs(contribution) > 1e-12:
            patient = first_id if contribution > 0 else second_id
            directions.append(f'{term}の違いは{patient}の危険度を相対的に高める向き')
    statement = ('この2人の比較の計算内訳では、' + '、'.join(directions) + 'に働いていました。') if directions else 'この2人には計算上の危険度差がありませんでした。'
    if first_rank < second_rank:
        final = f'{first_id}を{second_id}より優先する'
    elif first_rank > second_rank:
        final = f'{second_id}を{first_id}より優先する'
    else:
        final = f'{first_id}と{second_id}を同じ優先度とする'
    values = decomposition['contributions'].values()
    opposing = any(v > 1e-12 for v in values) and any(v < -1e-12 for v in values)
    balance = ''
    if opposing and first_rank != second_rank:
        winner = first_id if first_rank < second_rank else second_id
        balance = f'両側の影響を比べると、計算全体では{winner}側の押上げが上回りました。'
    return statement + balance + f'これらの影響を組み合わせた結果、システムは{final}順としました。'


def _case_bridge(pairs, system, human):
    comparisons = []
    terms = ('患者属性', '患者行動', '周囲物体', 'スタッフとの距離', '見守る方向')
    for pair in pairs:
        a, b = pair['patient_ids']
        directions = []
        for name, term in zip(GROUPS, terms):
            contribution = pair['decomposition']['contributions'][name]
            if abs(contribution) > 1e-12:
                directions.append(f'{term}は{a if contribution > 0 else b}の危険度を高める向き')
        final = (f'{a}を優先' if system[a] < system[b] else f'{b}を優先'
                 if system[b] < system[a] else '同じ優先度')
        details = '・'.join(directions) if directions else '計算上の危険度に差がない状態'
        comparisons.append(f'{a}と{b}の比較は' + details + f'で、組み合わせると「{final}」')
    return ('計算内訳では、' + '；'.join(comparisons) + 'という結果になりました。'
            f'各比較を含めて{len(system)}人を並べた算出順位は{_rank_text(system)}で、'
            f'最初の回答{_rank_text(human)}とは上記の患者対の優先関係が異なっています。')


def analyze_person(manifest_path, knowledge_path, responses_path, output_dir,
                   action_aggregation='weighted_sum', command=()):
    from master_thesis_modules.risk_core.engine.personalized_config import make_personalized_risk_config
    knowledge_path, responses_path = Path(knowledge_path).resolve(), Path(responses_path).resolve()
    knowledge = json.loads(knowledge_path.read_text(encoding='utf-8'))
    respondent = _identifier(knowledge['respondent_id'], 'respondent_id')
    config = make_personalized_risk_config(knowledge, action_aggregation=action_aggregation)
    manifest_path, manifest, cases = load_cases(manifest_path)
    answers = validate_responses(manifest_path, responses_path)
    own = [r for r in answers if r['respondent_id'] == respondent]
    if not own:
        raise ValueError('knowledge respondent missing from initial responses')
    if any(not r['reason'] for r in own):
        raise ValueError('initial reasons required before explanation')
    engine = RiskEngine(config)
    snapshots, ranks_rows, comparison, packet_cases = [], [], [], []
    summary_cases = []
    for case in cases:
        case_id = case['id']
        evaluated = {case['mapping'][frame.person_id]: engine.evaluate(frame) for frame in case['frames']}
        for result in evaluated.values():
            if not math.isclose(score_groups(config, risk_groups(result)), result.total_risk,
                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('five-input replay disagrees with RiskEngine')
        system = {p: 1 + sum(other.total_risk > r.total_risk for other in evaluated.values())
                  for p, r in evaluated.items()}
        original = {r['patient_id']: r for r in own if r['case_id'] == case_id}
        ranked = all(r['status'] == 'ranked' for r in original.values())
        human = {p: int(r['rank']) for p, r in original.items()} if ranked else None
        matched = human == system if ranked else None
        summary_cases.append({'case_id': case_id, 'exact_rank_match': matched,
                              'initial_status': 'ranked' if ranked else 'unable',
                              'followup_status': 'not_asked_agreement' if matched else
                                  'not_asked_unable' if not ranked else 'pending'})
        for p, result in evaluated.items():
            row = {'respondent_id': respondent, 'case_id': case_id, 'patient_id': p,
                   'system_score': result.total_risk, 'system_rank': system[p]}
            ranks_rows.append(row)
            comparison.append({**row, 'response_rank': original[p]['rank'],
                               'response_status': original[p]['status'], 'reason': original[p]['reason']})
            snapshots.append({'case_id': case_id, 'patient_id': p,
                              'feature_frame': asdict(next(f for f in case['frames'] if case['mapping'][f.person_id] == p)),
                              'result': asdict(result), 'five_groups': dict(zip(GROUPS, risk_groups(result))),
                              'rule_trace': trace_rules(config, knowledge, result)})
        if not ranked or matched:
            continue
        pairs = []
        for first, second in itertools.combinations(sorted(evaluated), 2):
            if _relation(human[first], human[second]) == _relation(system[first], system[second]):
                continue
            decomposition = decompose_pair(config, risk_groups(evaluated[first]), risk_groups(evaluated[second]))
            answer_lines, answer_refs = _referenced_answers(config, knowledge, evaluated[first], evaluated[second])
            object_text, object_refs = ('', [])
            if abs(decomposition['contributions']['objects']) > 1e-12:
                object_text, object_refs = _object_comparison(config, knowledge, evaluated[first], evaluated[second])
            bridge = _direction_bridge(first, second, decomposition, system[first], system[second])
            statement = (' '.join(answer_lines) + (' ' + object_text if object_text else '') + '\n' + _actual_conditions(case, first, second) + '\n'
                         + bridge + '\n' + f'システムは、これらの回答をこの場面の条件に当てはめ、{first}を{system[first]}位、'
                         f'{second}を{system[second]}位と算出しました。最初の回答は{first}が{human[first]}位、'
                         f'{second}が{human[second]}位だったため、この2人の順が異なりました。'
                         '引用した回答だけでなく、動作・物体・見守りについての回答を組み合わせた結果です。'
                         '一つの回答だけが順位差の原因だと断定するものではありません。')
            pairs.append({'pair_id': first + '__' + second, 'patient_ids': [first, second],
                          'human_relation': _relation(human[first], human[second]),
                          'system_relation': _relation(system[first], system[second]),
                          'explanation': statement, 'direction_explanation': bridge, 'answer_references': answer_refs,
                          'object_answer_references': object_refs, 'decomposition': decomposition})
        packet_cases.append({'case_id': case_id, 'scene': _scene(case), 'initial_ranks': human,
                             'initial_reasons': {p: r['reason'] for p, r in original.items()},
                             'system_ranks': system, 'answer_explanations': list(dict.fromkeys(
                                 line for pair in pairs for line in pair['explanation'].split('\n')[:1])),
                             'case_explanation': _case_bridge(pairs, system, human),
                             'pairs': pairs, 'raw_fuzzy_answers': knowledge['fuzzy_answers'],
                             'actual_calculation': {p: {'five_groups': dict(zip(GROUPS, risk_groups(r))),
                                 'rules': trace_rules(config, knowledge, r), 'total_risk': r.total_risk}
                                 for p, r in evaluated.items()}, 'question': QUESTION})
    packet = {'schema_version': 1, 'request': 'REQ-004 revision 2', 'respondent_id': respondent,
              'explanation_version': 2, 'initial_responses_sha256': _sha(responses_path), 'knowledge_sha256': _sha(knowledge_path),
              'cases': packet_cases, 'summary_cases': summary_cases,
              'response_contract': {'helpfulness': {'1': '全く参考にならない', '2': 'あまり参考にならない',
                '3': 'どちらともいえない', '4': 'やや参考になる', '5': 'とても参考になる', 'null': '判断できない'},
                'acceptance': {'accept': '受け入れる', 'partial': '一部を受け入れる',
                               'reject': '受け入れない', 'uncertain': '判断を保留する'},
                'reason': '自由回答を必須とする。患者対と全体順位への受容を別々に回答する。'}}
    summary = {'respondent_id': respondent, 'cases': summary_cases,
               'exact_matches': sum(c['exact_rank_match'] is True for c in summary_cases),
               'ranked_cases': sum(c['initial_status'] == 'ranked' for c in summary_cases),
               'followup_cases': len(packet_cases), 'primary_metric': None}
    provenance = _provenance([manifest_path, knowledge_path, responses_path, *(c['source'] for c in cases)], command)
    provenance.update({'request': 'REQ-004 revision 2', 'action_aggregation': action_aggregation,
                       'explanation_version': 2, 'knowledge_metadata': {k: v for k, v in knowledge.items() if k not in {'action_pairwise', 'object_pairwise', 'fuzzy_answers'}},
                       'initial_response_version': 'immutable source bytes + SHA256',
                       'prompt_provenance': 'Only supplied knowledge metadata retained; absent prompt/model/seed not reconstructed',
                       'watching_assumption': 'Staff movement direction is a proxy for gaze; continuous angle-based risk, not observed gaze or binary field-of-view classification',
                       'rank_rule': 'competition; exact float equality; no tolerance'})
    fields = ['respondent_id', 'case_id', 'patient_id', 'system_score', 'system_rank']
    template = {'schema_version': 1, 'respondent_id': respondent, 'packet_sha256': None,
                'cases': [{'case_id': c['case_id'], 'helpfulness': None, 'acceptance': None, 'reason': '',
                           'pairs': [{'pair_id': p['pair_id'], 'helpfulness': None, 'acceptance': None, 'reason': ''}
                                     for p in c['pairs']]} for c in packet_cases]}
    import hashlib
    template['packet_sha256'] = hashlib.sha256(_json(packet).encode()).hexdigest()
    files = {'knowledge.json': knowledge_path.read_text(encoding='utf-8'),
             'initial-responses.csv': responses_path.read_text(encoding='utf-8-sig'),
             'manifest.json': _json(manifest), 'snapshots.json': _json(snapshots),
             'config.json': _json(asdict(config)), 'system-ranks.csv': _csv(ranks_rows, fields),
             'comparison.csv': _csv(comparison, fields + ['response_rank', 'response_status', 'reason']),
             'followup-packet.json': _json(packet), 'followup-template.json': _json(template),
             'summary.json': _json(summary), 'provenance.json': _json(provenance)}
    output = _write_new(output_dir, files)
    # Preserve original response bytes, including BOM/newlines, for auditing.
    (output / 'initial-responses.csv').write_bytes(responses_path.read_bytes())
    return {'output_dir': str(output), 'packet_path': str(output / 'followup-packet.json'), 'summary': summary}


def validate_followup(packet_path, answers_path, output_dir, command=()):
    packet_path, answers_path = Path(packet_path).resolve(), Path(answers_path).resolve()
    packet = json.loads(packet_path.read_text(encoding='utf-8'))
    answer = json.loads(answers_path.read_text(encoding='utf-8'))
    if not isinstance(packet, dict) or not isinstance(answer, dict):
        raise ValueError('packet and followup must be JSON objects')
    if packet.get('schema_version') != 1 or answer.get('schema_version') != 1:
        raise ValueError('schema_version must be 1')
    if answer.get('respondent_id') != packet['respondent_id'] or answer.get('packet_sha256') != _sha(packet_path):
        raise ValueError('followup respondent/packet binding mismatch')
    expected = {c['case_id']: c for c in packet['cases']}
    actual = answer.get('cases')
    if (not isinstance(actual, list) or len(actual) != len(expected)
            or any(not isinstance(c, dict) for c in actual)
            or {c.get('case_id') for c in actual} != set(expected)):
        raise ValueError('followup must cover exactly requested mismatched cases')
    def validate_record(record):
        if 'helpfulness' not in record or (record['helpfulness'] is not None and (
                type(record['helpfulness']) is not int or record['helpfulness'] not in range(1, 6))):
            raise ValueError('helpfulness must be integer 1..5 or null (unable)')
        if record.get('acceptance') not in ACCEPTANCE:
            raise ValueError('invalid acceptance')
        if not isinstance(record.get('reason'), str) or not record['reason'].strip():
            raise ValueError('free reason required for every case and pair')
    for case in actual:
        validate_record(case)
        pairs = case.get('pairs')
        expected_pairs = {p['pair_id'] for p in expected[case['case_id']]['pairs']}
        if (not isinstance(pairs, list) or len(pairs) != len(expected_pairs)
                or any(not isinstance(p, dict) for p in pairs)
                or {p.get('pair_id') for p in pairs} != expected_pairs):
            raise ValueError('followup pairs must exactly match packet case')
        for pair in pairs:
            validate_record(pair)
    summary = {'respondent_id': packet['respondent_id'], 'answered_cases': len(actual),
               'cases': actual, 'initial_comparison': packet['summary_cases'],
               'interpretation': 'acceptance is self-report, not a changed initial rank',
               'combined_primary_metric': None}
    provenance = _provenance([packet_path, answers_path], command)
    provenance['request'] = 'REQ-004 revision 2'
    output = _write_new(output_dir, {'answers.json': answers_path.read_text(encoding='utf-8'),
                                    'packet.json': packet_path.read_text(encoding='utf-8'),
                                    'summary.json': _json(summary), 'provenance.json': _json(provenance)})
    return {'output_dir': str(output), 'summary': summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    analyze = commands.add_parser('analyze', help='初回順位と算出順位を比較し不一致の説明を作成')
    for name in ('manifest', 'knowledge', 'responses', 'output_dir'):
        analyze.add_argument(name)
    analyze.add_argument('--action-aggregation', choices=('weighted_sum', 'weighted_max'), default='weighted_sum')
    followup = commands.add_parser('followup', help='元順位を保持して追加回答を検証・保存')
    for name in ('packet', 'answers', 'output_dir'):
        followup.add_argument(name)
    args = parser.parse_args(argv)
    command = [sys.executable, *sys.argv] if argv is None else list(argv)
    try:
        if args.command == 'analyze':
            result = analyze_person(args.manifest, args.knowledge, args.responses, args.output_dir, args.action_aggregation, command)
        else:
            result = validate_followup(args.packet, args.answers, args.output_dir, command)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'error: {exc}\n')
    print(_json(result), end='')
