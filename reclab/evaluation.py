"""Validation-only parameter selection and untouched final test metrics."""

import hashlib
import json
import math
from dataclasses import asdict

from . import __version__
from .data import split_ratings
from .model import Recommender


def fingerprint(rows):
    values = [asdict(row) for row in sorted(rows, key=lambda r: (r.user, r.item))]
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def measure(model, rows):
    if not rows:
        raise ValueError("metrics require nonempty observations")
    errors, fallbacks = [], 0
    for row in rows:
        prediction = model.predict(row.user, row.item)
        errors.append(prediction.score - row.score)
        fallbacks += prediction.fallback
    return {"rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "mae": sum(abs(e) for e in errors) / len(errors),
            "fallback_rate": fallbacks / len(errors), "count": len(errors)}


def experiment(rows, candidate_k=(3, 5, 10, 20), seed=42):
    candidates = list(candidate_k)
    if not candidates or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in candidates):
        raise ValueError("candidate K values must be positive integers")
    candidates = sorted(set(candidates))
    train, validation, test = split_ratings(rows, seed)
    final_train = train + validation
    report = {"version": __version__, "seed": seed,
              "protocol": "per-user 70/15/15; validation selects K; refit train+validation; test once",
              "parameters": {"shrinkage": 5.0, "regularization": 10.0, "bias_iterations": 12},
              "splits": {name: {"count": len(group), "sha256": fingerprint(group)}
                         for name, group in (("train", train), ("validation", validation), ("test", test))},
              "baseline": measure(Recommender("baseline").fit(final_train), test), "algorithms": {}}
    for mode in ("user", "item"):
        model = Recommender(mode).fit(train)
        curve = []
        for k in candidates:
            model.k = k
            curve.append({"k": k, **measure(model, validation)})
        best = min(curve, key=lambda entry: (entry["rmse"], entry["k"]))
        final_model = Recommender(mode, k=best["k"]).fit(final_train)
        report["algorithms"][mode] = {"best_k": best["k"], "validation": curve,
                                       "test": measure(final_model, test)}
    return report
