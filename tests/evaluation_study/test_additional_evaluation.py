import csv
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from master_thesis_modules.evaluation_study.additional_evaluation import (
    analyze_person, decompose_pair, risk_groups, score_groups, validate_followup)
from master_thesis_modules.evaluation_study.study import FIELDS, load_cases
from master_thesis_modules.risk_core.engine.personalized_config import make_personalized_risk_config
from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine


@pytest.fixture
def inputs(tmp_path):
    scene = {'scenario_name': 'unit', 'time_range': {'duration_s': 1, 'step_s': 1},
             'patients': [{'id': 'a', 'position': {'x': 0, 'y': 0}, 'age_group_label': 'old', 'action_label': 'standing_up'},
                          {'id': 'b', 'position': {'x': 4, 'y': 0}, 'age_group_label': 'old', 'action_label': 'neutral_sitting'}],
             'staff': [{'id': 's', 'position': {'x': 1, 'y': 2}, 'velocity': {'vx': 0, 'vy': -1}}],
             'objects': [{'id': 'o', 'type': 'wheelchair', 'position': {'x': 0, 'y': 0}}]}
    (tmp_path / 'scene.yaml').write_text(yaml.safe_dump(scene))
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(yaml.safe_dump({'schema_version': 1, 'cases': [{'case_id': 'C01', 'scenario': 'scene.yaml',
        'time_s': 0, 'description': '同じ場面', 'patient_ids': {'a': 'P01', 'b': 'P02'}}]}))
    knowledge = {'respondent_id': 'S01', 'synthetic': True,
                 'action_pairwise': [[1]*7 for _ in range(7)], 'object_pairwise': [[1]*3 for _ in range(3)],
                 'fuzzy_answers': {'internal': [5, 3, 4, 1], 'watching': [5, 2, 4, 1],
                                   'external': [5, 4, 3, 1], 'total': [5, 2, 4, 1]}}
    kp = tmp_path / 'knowledge.json'
    kp.write_text(json.dumps(knowledge))
    rp = tmp_path / 'responses.csv'
    with rp.open('w', newline='') as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows([{'case_id': 'C01', 'respondent_id': 'S01', 'patient_id': p,
                          'rank': '1', 'status': 'ranked', 'reason': '同じ優先度と判断した'} for p in ['P01', 'P02']])
    return manifest, kp, rp, knowledge


@pytest.mark.parametrize('aggregation', ['weighted_sum', 'weighted_max'])
def test_exact_replay_and_shapley(inputs, aggregation):
    manifest, _, _, knowledge = inputs
    config = make_personalized_risk_config(knowledge, action_aggregation=aggregation)
    results = [RiskEngine(config).evaluate(f) for f in load_cases(manifest)[2][0]['frames']]
    for result in results:
        assert score_groups(config, risk_groups(result)) == pytest.approx(result.total_risk, abs=1e-14)
    decomposition = decompose_pair(config, risk_groups(results[0]), risk_groups(results[1]))
    assert len(decomposition['hybrid_scores']) == 32
    assert sum(decomposition['contributions'].values()) == pytest.approx(results[0].total_risk - results[1].total_risk, abs=1e-14)
    reverse = decompose_pair(config, risk_groups(results[1]), risk_groups(results[0]))
    for group, value in decomposition['contributions'].items():
        assert reverse['contributions'][group] == pytest.approx(-value, abs=1e-14)


def test_watching_canonical_order(inputs):
    config = make_personalized_risk_config(inputs[3])
    # Isolate watching: external and total simply propagate their second input.
    from master_thesis_modules.risk_core.schema import node_ids as ids
    rules = dict(config.fuzzy_rule_results)
    rules[ids.EXTERNAL_RISK] = (1., 0., 1., 0.)
    rules[ids.TOTAL_RISK] = (1., 0., 1., 0.)
    config = replace(config, fuzzy_rule_results=rules)
    assert score_groups(config, (0, 0, 0, 1, 0)) == .25  # HL score2
    assert score_groups(config, (0, 0, 0, 0, 1)) == .75  # LH score4


def test_ties_packet_preserves_sources_and_no_overwrite(inputs, tmp_path):
    manifest, kp, rp, _ = inputs
    before = rp.read_bytes()
    result = analyze_person(manifest, kp, rp, tmp_path / 'analysis')
    packet = json.loads(Path(result['packet_path']).read_text())
    assert packet['cases'][0]['pairs'][0]['human_relation'] == 'tied'
    assert len(packet['cases'][0]['pairs']) == 1
    assert packet['cases'][0]['pairs'][0]['system_relation'] != 'tied'
    assert rp.read_bytes() == before == (tmp_path / 'analysis/initial-responses.csv').read_bytes()
    assert result['summary']['followup_cases'] == 1
    with pytest.raises(FileExistsError):
        analyze_person(manifest, kp, rp, tmp_path / 'analysis')
    assert rp.read_bytes() == before


def test_agreement_not_positive_acceptance(inputs, tmp_path):
    manifest, kp, rp, knowledge = inputs
    case = load_cases(manifest)[2][0]
    scores = {case['mapping'][f.person_id]: RiskEngine(make_personalized_risk_config(knowledge)).evaluate(f).total_risk for f in case['frames']}
    rows = list(csv.DictReader(rp.open()))
    for row in rows:
        row['rank'] = str(1 + sum(s > scores[row['patient_id']] for s in scores.values()))
    with rp.open('w', newline='') as f:
        writer = csv.DictWriter(f, FIELDS); writer.writeheader(); writer.writerows(rows)
    result = analyze_person(manifest, kp, rp, tmp_path / 'analysis')
    packet = json.loads(Path(result['packet_path']).read_text())
    assert packet['cases'] == []
    assert packet['summary_cases'][0]['followup_status'] == 'not_asked_agreement'
    assert 'acceptance' not in packet['summary_cases'][0]


def _followup(inputs, tmp_path):
    result = analyze_person(*inputs[:3], tmp_path / 'analysis')
    packet = Path(result['packet_path'])
    answer = json.loads((packet.parent / 'followup-template.json').read_text())
    for case in answer['cases']:
        for record in [case, *case['pairs']]:
            record.update(helpfulness=4, acceptance='reject', reason='参考になるが順位には同意しない')
    ap = tmp_path / 'followup.json'
    ap.write_text(json.dumps(answer, ensure_ascii=False))
    return packet, ap, answer


def test_followup_separate_axes_and_pair_case(inputs, tmp_path):
    packet, ap, answer = _followup(inputs, tmp_path)
    answer['cases'][0]['pairs'][0]['acceptance'] = 'partial'
    ap.write_text(json.dumps(answer))
    result = validate_followup(packet, ap, tmp_path / 'validated')
    assert result['summary']['cases'][0]['acceptance'] == 'reject'
    assert result['summary']['cases'][0]['pairs'][0]['acceptance'] == 'partial'
    with pytest.raises(FileExistsError):
        validate_followup(packet, ap, tmp_path / 'validated')


@pytest.mark.parametrize('mutation', ['person', 'packet', 'case', 'pair', 'reason', 'helpfulness', 'acceptance'])
def test_invalid_followup_does_not_create_output(inputs, tmp_path, mutation):
    packet, ap, answer = _followup(inputs, tmp_path)
    if mutation == 'person': answer['respondent_id'] = 'S02'
    if mutation == 'packet': answer['packet_sha256'] = 'bad'
    if mutation == 'case': answer['cases'][0]['case_id'] = 'C02'
    if mutation == 'pair': answer['cases'][0]['pairs'][0]['pair_id'] = 'P02__P03'
    if mutation == 'reason': answer['cases'][0]['reason'] = ''
    if mutation == 'helpfulness': answer['cases'][0]['helpfulness'] = True
    if mutation == 'acceptance': answer['cases'][0]['acceptance'] = 'yes'
    ap.write_text(json.dumps(answer))
    with pytest.raises(ValueError):
        validate_followup(packet, ap, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()


def test_missing_watching_rejected(inputs, tmp_path):
    manifest, kp, rp, knowledge = inputs
    del knowledge['fuzzy_answers']['watching']
    kp.write_text(json.dumps(knowledge))
    with pytest.raises(ValueError):
        analyze_person(manifest, kp, rp, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()


def test_rule_trace_reconstructs_actual_stages(inputs, tmp_path):
    result = analyze_person(*inputs[:3], tmp_path / 'analysis')
    snapshots = json.loads((Path(result['output_dir']) / 'snapshots.json').read_text())
    for patient in snapshots:
        for group, stage in patient['rule_trace'].items():
            output = sum(a*b for a, b in zip(stage['canonical_activations'], stage['canonical_representatives']))
            assert output == pytest.approx(stage['output'], abs=1e-14)
            assert stage['raw_answers_1to5'] == inputs[3]['fuzzy_answers'][group]
            if group == 'watching':
                assert stage['local_rule_order'] == ['HH', 'LH', 'HL', 'LL']


def test_unable_is_not_rank_mismatch_or_acceptance(inputs, tmp_path):
    manifest, kp, rp, _ = inputs
    rows = list(csv.DictReader(rp.open()))
    for row in rows:
        row.update(rank='', status='unable', reason='判断材料が足りない')
    with rp.open('w', newline='') as f:
        writer = csv.DictWriter(f, FIELDS); writer.writeheader(); writer.writerows(rows)
    result = analyze_person(manifest, kp, rp, tmp_path / 'analysis')
    assert result['summary']['ranked_cases'] == 0
    assert result['summary']['followup_cases'] == 0
    case = result['summary']['cases'][0]
    assert case['exact_rank_match'] is None
    assert case['followup_status'] == 'not_asked_unable'


def test_initial_reason_required_before_explanation(inputs, tmp_path):
    manifest, kp, rp, _ = inputs
    rows = list(csv.DictReader(rp.open()))
    rows[0]['reason'] = ''
    with rp.open('w', newline='') as f:
        writer = csv.DictWriter(f, FIELDS); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match='initial reasons'):
        analyze_person(manifest, kp, rp, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()


def test_staff_explanation_actual_conditions_without_calculation_jargon(inputs, tmp_path):
    manifest, kp, rp, knowledge = inputs
    knowledge['source_question_ids'] = {'watching': ['W01', 'W02', 'W03', 'W04']}
    kp.write_text(json.dumps(knowledge))
    result = analyze_person(manifest, kp, rp, tmp_path / 'analysis')
    packet = json.loads(Path(result['packet_path']).read_text())
    case = packet['cases'][0]
    text = ' '.join([*case['answer_explanations'], *(p['explanation'] for p in case['pairs'])])
    for forbidden in ['Shapley', 'HH', 'HL', 'LH', 'LL', '後件', '共通ルール', '所属度', '視線代理', '移動方向']:
        assert forbidden not in text
    assert '立ち上がる' in text and '座っている' in text
    assert '車椅子まで' in text and 'スタッフまで' in text
    assert '最初の回答' in text and '5段階の' in text
    refs = case['pairs'][0]['answer_references']
    assert refs
    for ref in refs:
        group, index = ref['knowledge_pointer'].split('/')[-2:]
        assert ref['raw_answer'] == knowledge['fuzzy_answers'][group][int(index)]
        assert ref['source_metadata']['source_question_ids'] == knowledge['source_question_ids']
    assert len(refs) < 16


def test_helpfulness_unable_is_separate_from_acceptance(inputs, tmp_path):
    packet, ap, answer = _followup(inputs, tmp_path)
    answer['cases'][0]['helpfulness'] = None
    answer['cases'][0]['acceptance'] = 'reject'
    answer['cases'][0]['reason'] = '説明の参考度は判断できないが算出順位は受け入れない'
    ap.write_text(json.dumps(answer))
    result = validate_followup(packet, ap, tmp_path / 'validated')
    assert result['summary']['cases'][0]['helpfulness'] is None
    assert result['summary']['cases'][0]['acceptance'] == 'reject'


def test_missing_helpfulness_is_not_explicit_unable(inputs, tmp_path):
    packet, ap, answer = _followup(inputs, tmp_path)
    del answer['cases'][0]['helpfulness']
    ap.write_text(json.dumps(answer))
    with pytest.raises(ValueError, match='helpfulness'):
        validate_followup(packet, ap, tmp_path / 'invalid')


def test_direction_bridge_follows_all_decomposition_signs(inputs, tmp_path):
    result = analyze_person(*inputs[:3], tmp_path / 'analysis')
    packet = json.loads(Path(result['packet_path']).read_text())
    assert packet['explanation_version'] == 2
    case = packet['cases'][0]
    for pair in case['pairs']:
        a, b = pair['patient_ids']
        for key, term in zip(('patient_attributes', 'patient_action', 'objects', 'staff_distance', 'outside_view'),
                             ('患者属性', '患者行動', '周囲物体', 'スタッフとの距離', 'スタッフの見守る方向')):
            value = pair['decomposition']['contributions'][key]
            if abs(value) > 1e-12:
                patient = a if value > 0 else b
                assert f'{term}の違いは{patient}の危険度を相対的に高める向き' in pair['explanation']
        assert pair['direction_explanation'] in pair['explanation']
        assert '組み合わせた結果' in pair['explanation']
        assert '計算内訳' in case['case_explanation']
        for ref in pair['object_answer_references']:
            _, matrix, i, j = ref['knowledge_pointer'].split('/')
            assert inputs[3][matrix][int(i)][int(j)] == ref['raw_answer']
            assert ref['meaning'] == 'pairwise preference, not risk ratio'
    assert case['case_explanation'].count('。') == 2
    assert '危険性が' not in case['case_explanation']  # calculation direction, not risk ratio claim


def test_zero_difference_case_summary_has_complete_sentence():
    from master_thesis_modules.evaluation_study.additional_evaluation import _case_bridge, GROUPS
    pair = {'patient_ids': ['P01', 'P02'], 'decomposition': {'contributions': {g: 0.0 for g in GROUPS}}}
    text = _case_bridge([pair], {'P01': 1, 'P02': 1}, {'P01': 1, 'P02': 2})
    assert '計算上の危険度に差がない状態' in text
    assert '比較はで' not in text
