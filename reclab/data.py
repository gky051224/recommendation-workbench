"""Validated CSV input, synthetic fixtures, and reproducible holdouts."""

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rating:
    user: str
    item: str
    score: float


def validate(ratings):
    ratings = list(ratings)
    if not ratings:
        raise ValueError("ratings must not be empty")
    pairs = set()
    for row in ratings:
        if not isinstance(row.user, str) or not row.user.strip():
            raise ValueError("user IDs must be nonempty strings")
        if not isinstance(row.item, str) or not row.item.strip():
            raise ValueError("item IDs must be nonempty strings")
        if not math.isfinite(row.score) or not 1 <= row.score <= 5:
            raise ValueError("ratings must be finite numbers between 1 and 5")
        pair = (row.user, row.item)
        if pair in pairs:
            raise ValueError(f"duplicate user/item pair: {pair}")
        pairs.add(pair)
    return ratings


def load_ratings(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"user_id", "item_id", "rating"} <= set(reader.fieldnames or []):
            raise ValueError("CSV requires user_id,item_id,rating columns")
        rows = []
        for line, row in enumerate(reader, 2):
            try:
                rows.append(Rating(row["user_id"].strip(), row["item_id"].strip(), float(row["rating"])))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"invalid CSV row {line}") from exc
    return validate(rows)


def split_ratings(ratings, seed=42, validation_ratio=0.15, test_ratio=0.15):
    """Per-user holdout. Users with <3 ratings remain entirely in training.

    Sparse users still retain at least one training observation. Actual ratios
    vary with rounding and are recorded in the experiment report.
    """
    rows = validate(ratings)
    if not (0 < validation_ratio < 1 and 0 < test_ratio < 1
            and validation_ratio + test_ratio < 1):
        raise ValueError("holdout ratios must be positive and sum to less than 1")
    by_user = {}
    for row in rows:
        by_user.setdefault(row.user, []).append(row)
    rng = random.Random(seed)
    train, validation, test = [], [], []
    for user in sorted(by_user):
        group = sorted(by_user[user], key=lambda row: row.item)
        if len(group) < 3:
            train.extend(group)
            continue
        rng.shuffle(group)
        n_test = min(max(1, int(len(group) * test_ratio)), len(group) - 2)
        n_validation = min(max(1, int(len(group) * validation_ratio)), len(group) - n_test - 1)
        test.extend(group[:n_test])
        validation.extend(group[n_test:n_test + n_validation])
        train.extend(group[n_test + n_validation:])
    if not validation or not test:
        raise ValueError("evaluation requires at least one user with three ratings")
    return train, validation, test


def create_demo(directory, seed=42):
    """Generate fictional catalog and preference data; never overwrite files."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    ratings_path, items_path = target / "ratings.csv", target / "items.csv"
    if ratings_path.exists() or items_path.exists():
        raise ValueError("demo files already exist; choose a new directory")
    rng = random.Random(seed)
    genres = ["adventure", "comedy", "mystery", "science-fiction"]
    with items_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_id", "title", "genre"])
        for item in range(36):
            writer.writerow([f"film-{item:02d}", f"Fictional Film {item + 1:02d}", genres[item % 4]])
    with ratings_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_id", "item_id", "rating"])
        for user in range(80):
            bias = rng.uniform(-0.4, 0.4)
            for item in sorted(rng.sample(range(36), 18)):
                affinity = 1.3 if item % 4 == user % 4 else -0.35
                score = min(5, max(1, 3.1 + bias + affinity + rng.gauss(0, 0.35)))
                writer.writerow([f"viewer-{user:03d}", f"film-{item:02d}", round(score, 2)])
    return ratings_path, items_path
