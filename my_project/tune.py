"""ハイパーパラメータ探索。

各候補を`model.cross_validate`（3fold）で評価し、mean/stdで比較する。
std（0.00565、docs/experiments.md参照）より小さい差は誤差とみなし、改善と判断しない。
"""

from __future__ import annotations

from my_project.io import load_train
from my_project.model import LGB_PARAMS, cross_validate

# 1軸ずつベースラインから振る（coordinate descent）。全部同時に組み合わせると候補数が爆発するため。
CANDIDATES = [
    ("baseline", {}),
    ("num_leaves=31", {"num_leaves": 31}),
    ("num_leaves=95", {"num_leaves": 95}),
    ("num_leaves=127", {"num_leaves": 127}),
    ("min_child_samples=10", {"min_child_samples": 10}),
    ("min_child_samples=50", {"min_child_samples": 50}),
    ("feature_fraction=0.7", {"feature_fraction": 0.7}),
    ("feature_fraction=0.9", {"feature_fraction": 0.9}),
    ("bagging_fraction=0.7", {"bagging_fraction": 0.7}),
    ("bagging_fraction=0.9", {"bagging_fraction": 0.9}),
    ("learning_rate=0.03", {"learning_rate": 0.03}),
    ("lambda_l2=1.0", {"lambda_l2": 1.0}),
]


def run_search() -> list[dict]:
    train_df = load_train()
    results = []
    for name, overrides in CANDIDATES:
        params = {**LGB_PARAMS, **overrides}
        cv = cross_validate(train_df, params=params, verbose=False)
        print(f"{name:<24} mean={cv['mean']:.5f} ± {cv['std']:.5f}  (scores={[round(s, 5) for s in cv['scores']]})")
        results.append({"name": name, "overrides": overrides, **cv})
    return results


if __name__ == "__main__":
    results = run_search()
    baseline_mean = next(r["mean"] for r in results if r["name"] == "baseline")
    print(f"\nbaseline mean = {baseline_mean:.5f}")
    best = min(results, key=lambda r: r["mean"])
    print(f"best: {best['name']}  mean={best['mean']:.5f} (baseline比 {best['mean'] - baseline_mean:+.5f})")
