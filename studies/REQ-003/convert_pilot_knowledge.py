"""Validate synthetic pilot answers and write the existing AHP/TFN CSV contract.

This preserves the supplied values; it never repairs inconsistent judgements.
Only the central value is used by the current engine, so l=c=r is a deliberate
CSV encoding for this pilot, not an estimate of judgement uncertainty.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


def convert(source: Path, output: Path) -> None:
    source = Path(source)
    data = json.loads(source.read_text())
    identity = data['respondent_id']
    if data.get('synthetic') is not True or not re.fullmatch(r'S\d{2}', identity):
        raise ValueError('Only explicitly synthetic Sxx pilot profiles are supported')
    matrices = []
    for key, size in [('action_pairwise', 7), ('object_pairwise', 3)]:
        matrix = data[key]
        if len(matrix) != size or any(len(row) != size for row in matrix):
            raise ValueError(f'{key} must be {size}x{size}')
        for i in range(size):
            for j in range(size):
                x = matrix[i][j]
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 1/9 <= x <= 9:
                    raise ValueError(f'{key}: invalid pairwise value')
                if i == j and x != 1:
                    raise ValueError(f'{key}: diagonal must be one')
                if not math.isclose(x * matrix[j][i], 1, abs_tol=1e-9):
                    raise ValueError(f'{key}: answers must be reciprocal; no automatic repair')
        matrices.append(matrix)
    groups = data['fuzzy_answers']
    for key in ['internal', 'external', 'total']:
        if len(groups[key]) != 4 or any(type(x) is not int or x not in range(1, 6) for x in groups[key]):
            raise ValueError(f'{key}: four 1..5 integer answers required')
    # Existing TFN row contract: internal HH HL LH LL; external HL HH LL LH;
    # total HH HL LH LL. Factory reorders external rows [5,4,7,6].
    answers = groups['internal'] + [groups['external'][i] for i in [1,0,3,2]] + groups['total']
    centers = [(x - 1)/4 for x in answers]
    output = Path(output)
    paths = [output / f'comparison_mtx_30000001_{identity}.csv',
             output / f'comparison_mtx_30000010_{identity}.csv',
             output / f'TFN_{identity}.csv', output / f'{identity}_knowledge_provenance.json']
    if any(p.exists() for p in paths):
        raise FileExistsError('Existing pilot knowledge files are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(paths[:3], matrices + [[[x,x,x] for x in centers]]):
        with path.open('x', newline='') as handle:
            csv.writer(handle).writerows(rows)
    paths[3].write_text(json.dumps({'synthetic': True, 'respondent_id':identity,
        'converted_at':datetime.now(timezone.utc).isoformat(),
        'converter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source':str(source.resolve()), 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'score_conversion':'(score-1)/4', 'input_group_order':['HH','HL','LH','LL'],
        'external_csv_row_order':['HL','HH','LL','LH'],
        'tfn_width':'l=c=r; central values only; uncertainty not estimated'}, ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    convert(args.source, args.output)
