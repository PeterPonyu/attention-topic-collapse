"""Exact two-sided Clopper--Pearson intervals for saved strict-collapse flags.

Uses only the paper-local matched training results. Seeds, not cells, are trials.
No external statistical package is required.
"""
import json
from collections import defaultdict
from pathlib import Path


def binom_cdf(k, n, p):
    from math import comb

    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def inverse_decreasing(target, k, n):
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def interval(x, n, alpha=0.05):
    # P_p(X >= x) = alpha/2 and P_p(X <= x) = alpha/2.
    lower = 0.0 if x == 0 else inverse_decreasing(1 - alpha / 2, x - 1, n)
    upper = 1.0 if x == n else inverse_decreasing(alpha / 2, x, n)
    return [lower, upper]


def main():
    source = Path(__file__).with_name('training_results.json')
    rows = json.loads(source.read_text())
    groups = defaultdict(list)
    for row in rows:
        expected = row['argmax_topics'] == 1 and row['max_coordinate_range'] <= 1e-8
        if row['collapsed'] != expected:
            raise ValueError(f"inconsistent strict-collapse flag: {row['dataset']} {row['seed']} {row['alpha']}")
        groups[(row['dataset'], row['alpha'])].append(row)
    out = []
    for (dataset, concentration), arm in sorted(groups.items()):
        seeds = [r['seed'] for r in arm]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f'duplicate seed: {dataset} {concentration}')
        x = sum(r['collapsed'] for r in arm)
        out.append(dict(dataset=dataset, alpha=concentration, seeds=sorted(seeds),
                        collapsed=x, total=len(arm), rate=x / len(arm),
                        ci95_clopper_pearson=interval(x, len(arm))))
    destination = Path(__file__).with_name('collapse_ci.json')
    destination.write_text(json.dumps(dict(source='training_results.json',
        unit='training seed within a fixed public data object',
        method='two-sided exact Clopper-Pearson, alpha=0.05',
        groups=out), indent=2) + '\n')
    print(destination.read_text())


if __name__ == '__main__':
    main()
