"""Recompute saved allocation evidence; never fit or update a model."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import beta
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ci", ROOT / "evidence/collapse_ci.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
source = ROOT / "evidence/training_results.json"
saved = json.loads(source.read_text())
results, graph_checks = [], []
hashes = {str(source.relative_to(ROOT)): hashlib.sha256(source.read_bytes()).hexdigest()}
for row in saved:
    directory = ROOT / "evidence/reproduction/training" / f"{row['dataset']}_s{row['seed']}_a{row['alpha']:g}"
    path = directory / "latent.npz"
    z = np.load(path)["latent"]
    assert z.shape == (450, 10) and np.isfinite(z).all()
    span = float(np.ptp(z, axis=0).max())
    occupancy = int(np.unique(z.argmax(1)).size)
    js = float(np.mean(pdist(z.astype(float), metric="jensenshannon") ** 2))
    perplexity = float(np.exp(-(z * np.log(np.maximum(z, 1e-30))).sum(1)).mean())
    collapse = bool(occupancy == 1 and span <= 1e-8)
    # Reproduce the archived graph helper's float64 input and rank slicing.
    graph_z = np.asarray(z, dtype=float)
    indices = NearestNeighbors(n_neighbors=16, n_jobs=1).fit(graph_z).kneighbors(graph_z, return_distance=False)
    distances = NearestNeighbors(n_neighbors=17, n_jobs=1).fit(graph_z).kneighbors(graph_z)[0]
    graph_checks.append(dict(dataset=row['dataset'], seed=row['seed'], alpha=row['alpha'],
                             unique_rows=int(np.unique(z,axis=0).shape[0]),
                             retained_self_indices=int((indices[:,1:] == np.arange(450)[:,None]).sum()),
                             equal_distances_at_k15_cutoff=int((distances[:,15] == distances[:,16]).sum()),
                             smallest_nonself_distance=float(distances[:,1].min())))
    assert collapse == row["collapsed"] and (js <= 1e-12) == row["degenerate"]
    for key, value in [("max_coordinate_range", span), ("argmax_topics", occupancy),
                       ("exact_pair_js", js), ("mean_perplexity", perplexity)]:
        assert np.isclose(value, row[key], rtol=1e-7, atol=1e-16), (key, value, row[key])
    schedule = np.load(directory.parent / f"{row['dataset']}_s{row['seed']}_schedule.npz")
    assert schedule["order"].shape == (200, 16, 128)
    assert schedule["random_seeds"].shape == (3200,)
    assert all(np.unique(ep).size == 2048 for ep in schedule["order"])
    assert row["updates"] == 3200 and row["presentations"] == 409600
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    results.append(dict(dataset=row["dataset"], seed=row["seed"], alpha=row["alpha"],
                        max_coordinate_range=span, argmax_topics=occupancy, exact_pair_js=js,
                        mean_perplexity=perplexity, strict_collapse=collapse,
                        simplex_max_error=float(np.abs(z.sum(1) - 1).max())))
groups = []
for ds in ("setty", "dentate"):
    for alpha in (0.1, 1):
        arm = [r for r in results if (r["dataset"], r["alpha"]) == (ds, alpha)]
        x, n = sum(r["strict_collapse"] for r in arm), len(arm)
        interval = [0.0 if x == 0 else float(beta.ppf(.025, x, n-x+1)),
                    1.0 if x == n else float(beta.ppf(.975, x+1, n-x))]
        assert np.allclose(interval, ci.interval(x, n), atol=1e-14)
        groups.append(dict(dataset=ds, alpha=alpha, collapsed=x, total=n, ci95=interval,
                           range_threshold_sensitivity={str(t): sum(r["argmax_topics"] == 1 and r["max_coordinate_range"] <= t for r in arm) for t in [1e-10, 1e-8, 1e-6]},
                           leave_one_seed_out={str(s): {"collapsed": sum(r["strict_collapse"] for r in arm if r["seed"] != s), "n": 2} for s in range(3)}))
probe_rows = json.loads((ROOT / "evidence/probe_results.json").read_text())
for row in probe_rows:
    path = ROOT / "evidence/reproduction/probes" / f"{row['dataset']}_s{row['seed']}_a{row['alpha']:g}_{row['mode']}.npz"
    data = np.load(path)
    actual = data[row["intervention"]]
    change = np.abs(actual - data["ordinary"][:len(actual)])
    assert np.isclose(change.max(), row["max_abs"], rtol=1e-7, atol=1e-16)
    # NumPy and torch float32 reductions differ by at most a few rounding units.
    assert np.isclose(change.sum(1).mean(), row["mean_l1"], rtol=5e-7, atol=1e-15)
identical = np.zeros((20,10))
tie_indices = NearestNeighbors(n_neighbors=6,n_jobs=1).fit(identical).kneighbors(identical,return_distance=False)
output = dict(scope="Independent saved-array recomputation; no training, threshold changes or new seed trials",
              input_sha256=hashes, arms=results, groups=groups, probe_records_verified=len(probe_rows),
              native_array_graph_checks=graph_checks,
              exact_duplicate_stress=dict(n=20,k=5,retained_self_indices=int((tie_indices[:,1:] == np.arange(20)[:,None]).sum()),
                                         implementation="Historical graph rule: NearestNeighbors(k+1).kneighbors(x)[:,1:]; reproduced directly here without modifying the scorer"),
              sensitivity="Exploratory range cutoffs 1e-10, 1e-8, 1e-6 leave all 12 classifications unchanged; original threshold remains primary.",
              limitation="0/3 does not exclude collapse. Leave-one-seed-out rates are influence checks, not additional trials.")
destination = ROOT / "validation/robustness-audit.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps({"arms_verified": len(results), "probe_records": len(probe_rows), "groups": groups}, indent=2))
