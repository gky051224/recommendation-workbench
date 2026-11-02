"""Regularized bias baseline and overlap-shrunk residual cosine neighbors."""

import math
from dataclasses import dataclass

from .data import validate


@dataclass(frozen=True)
class Prediction:
    score: float
    baseline: float
    neighbors: tuple
    fallback: bool


class Recommender:
    def __init__(self, mode="item", k=10, shrinkage=5.0, regularization=10.0):
        if mode not in {"user", "item", "baseline"}:
            raise ValueError("mode must be user, item, or baseline")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not math.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError("shrinkage must be finite and nonnegative")
        if not math.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        self.mode, self.k = mode, k
        self.shrinkage, self.regularization = shrinkage, regularization
        self._fitted = False

    def fit(self, ratings):
        rows = validate(ratings)
        self.users, self.items = {}, {}
        for row in rows:
            self.users.setdefault(row.user, {})[row.item] = row.score
            self.items.setdefault(row.item, {})[row.user] = row.score
        self.mean = sum(row.score for row in rows) / len(rows)
        self.user_bias = dict.fromkeys(self.users, 0.0)
        self.item_bias = dict.fromkeys(self.items, 0.0)
        # Alternating ridge updates fit user generosity and item popularity.
        for _ in range(12):
            for user, scores in self.users.items():
                self.user_bias[user] = sum(
                    score - self.mean - self.item_bias[item] for item, score in scores.items()
                ) / (self.regularization + len(scores))
            for item, scores in self.items.items():
                self.item_bias[item] = sum(
                    score - self.mean - self.user_bias[user] for user, score in scores.items()
                ) / (self.regularization + len(scores))
        self.vectors = {}
        for row in rows:
            entity, coordinate = (row.user, row.item) if self.mode == "user" else (row.item, row.user)
            self.vectors.setdefault(entity, {})[coordinate] = row.score - self._baseline(row.user, row.item)
        self._similarities = {}
        self._fitted = True
        return self

    def _baseline(self, user, item):
        return self.mean + self.user_bias.get(user, 0.0) + self.item_bias.get(item, 0.0)

    def _similarity(self, a, b):
        key = tuple(sorted((a, b)))
        if key not in self._similarities:
            left, right = self.vectors.get(a, {}), self.vectors.get(b, {})
            overlap = sorted(left.keys() & right.keys())
            numerator = sum(left[x] * right[x] for x in overlap)
            denominator = math.sqrt(sum(left[x] ** 2 for x in overlap) * sum(right[x] ** 2 for x in overlap))
            self._similarities[key] = (
                numerator / denominator * len(overlap) / (len(overlap) + self.shrinkage)
                if denominator else 0.0
            )
        return self._similarities[key]

    def predict(self, user, item):
        if not self._fitted:
            raise ValueError("fit the model before predicting")
        base = self._baseline(user, item)
        neighbors = []
        if self.mode == "user":
            for other, score in self.items.get(item, {}).items():
                if other != user:
                    similarity = self._similarity(user, other)
                    if similarity > 0:
                        neighbors.append((other, similarity, score - self._baseline(other, item)))
        elif self.mode == "item":
            for other, score in self.users.get(user, {}).items():
                if other != item:
                    similarity = self._similarity(item, other)
                    if similarity > 0:
                        neighbors.append((other, similarity, score - self._baseline(user, other)))
        neighbors.sort(key=lambda row: (-row[1], row[0]))
        selected = neighbors[:self.k]
        weight = sum(row[1] for row in selected)
        adjustment = sum(row[1] * row[2] for row in selected) / weight if weight else 0.0
        return Prediction(min(5.0, max(1.0, base + adjustment)), base, tuple(selected), not bool(selected))

    def recommend(self, user, top_n=10, candidates=None):
        if not self._fitted:
            raise ValueError("fit the model before recommending")
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        seen = self.users.get(user, {})
        pool = set(self.items if candidates is None else candidates) - seen.keys()
        ranked = [(item, self.predict(user, item)) for item in sorted(pool)]
        ranked.sort(key=lambda row: (-row[1].score, row[0]))
        return ranked[:top_n]
