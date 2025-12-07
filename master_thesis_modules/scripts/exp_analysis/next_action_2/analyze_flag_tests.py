import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(p_values):
    """Return FDR-adjusted p-values (Benjamini–Hochberg)."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    adjusted = p_values * m / ranks
    adjusted = np.minimum.accumulate(adjusted[order][::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out


def main():
    base = Path(__file__).resolve().parent
    src = base / "output/results_fix_answer_evaluate.csv"
    df = pd.read_csv(src)

    flag_cols = df.columns[7:]
    df["flag_sum"] = df[flag_cols].sum(axis=1)

    scenarios = sorted(df["シナリオ"].unique())
    print(f"Evaluating {len(scenarios)} scenarios from {src.name}")
    for scen in scenarios:
        sub = df[df["シナリオ"] == scen]
        pivot = (
            sub.pivot_table(
                index="被験者ID", columns="条件", values="flag_sum", aggfunc="first"
            )
            .reindex(columns=[1, 2, 3])
            .dropna()
        )
        n = pivot.shape[0]
        print(f"\nシナリオ {scen}: subjects with all 3 conditions = {n}")
        if n == 0:
            print("  Skipped (no complete trios).")
            continue

        # Friedman test across the three conditions
        friedman = stats.friedmanchisquare(pivot[1], pivot[2], pivot[3])
        print(
            f"  Friedman chi2={friedman.statistic:.3f}, p={friedman.pvalue:.4f} "
            f"(df=2, n={n})"
        )

        # Pairwise Wilcoxon with Bonferroni and FDR corrections
        pairs = [(1, 2), (1, 3), (2, 3)]
        p_raw = []
        stats_out = []
        for a, b in pairs:
            data = pivot[[a, b]].dropna()
            if data.empty:
                stats_out.append(((a, b), np.nan, np.nan))
                p_raw.append(np.nan)
                continue
            res = stats.wilcoxon(data[a], data[b], zero_method="wilcox", alternative="two-sided")
            stats_out.append(((a, b), res.statistic, res.pvalue))
            p_raw.append(res.pvalue)

        p_bonf = [p * len(pairs) if not np.isnan(p) else np.nan for p in p_raw]
        p_fdr = benjamini_hochberg(np.array([p for p in p_raw if not np.isnan(p)]))

        # map FDR back to pairs (skip NaNs)
        fdr_iter = iter(p_fdr)
        p_fdr_full = [next(fdr_iter) if not np.isnan(p) else np.nan for p in p_raw]

        print("  Wilcoxon pairwise (stat, p_raw, p_bonf, p_fdr):")
        for (a, b), statv, p in stats_out:
            idx = pairs.index((a, b))
            pb = p_bonf[idx]
            pf = p_fdr_full[idx]
            if np.isnan(p):
                print(f"    条件{a} vs 条件{b}: insufficient data")
            else:
                print(
                    f"    条件{a} vs 条件{b}: W={statv:.3f}, "
                    f"p={p:.4f}, p_bonf={min(pb,1):.4f}, p_fdr={pf:.4f}"
                )


if __name__ == "__main__":
    main()
