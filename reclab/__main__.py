"""Run with python -m reclab; errors return nonzero without a traceback."""

import argparse
import csv
import json
from pathlib import Path

from .data import create_demo, load_ratings
from .evaluation import experiment
from .model import Recommender


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible recommendation workbench")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="generate fictional data, refusing existing files")
    demo.add_argument("directory", type=Path)
    demo.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser("evaluate", help="select K on validation, score held-out test")
    evaluate.add_argument("ratings", type=Path)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--k", type=int, nargs="+", default=[3, 5, 10, 20])
    evaluate.add_argument("--report", type=Path)
    recommend = commands.add_parser("recommend", help="rank unseen items with neighbor evidence")
    recommend.add_argument("ratings", type=Path)
    recommend.add_argument("--user", required=True)
    recommend.add_argument("--mode", choices=["user", "item", "baseline"], default="item")
    recommend.add_argument("--k", type=int, default=10)
    recommend.add_argument("--top", type=int, default=10)
    recommend.add_argument("--catalog", type=Path, help="optional item_id,title,genre CSV")
    recommend.add_argument("--genre", help="restrict candidates to this catalog genre")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            ratings, catalog = create_demo(args.directory, args.seed)
            output = {"ratings": str(ratings), "catalog": str(catalog), "synthetic": True}
        elif args.command == "evaluate":
            output = experiment(load_ratings(args.ratings), args.k, args.seed)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            catalog = {}
            if args.catalog:
                with args.catalog.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    if not {"item_id", "title", "genre"} <= set(reader.fieldnames or []):
                        raise ValueError("catalog requires item_id,title,genre columns")
                    for row in reader:
                        item = row.get("item_id")
                        if not item or item in catalog or any(row.get(key) is None for key in ("title", "genre")):
                            raise ValueError("catalog IDs must be unique and rows complete")
                        catalog[item] = row
            if args.genre and not args.catalog:
                raise ValueError("--genre requires --catalog")
            candidates = [item for item, row in catalog.items() if row["genre"] == args.genre] if args.genre else None
            model = Recommender(args.mode, args.k).fit(load_ratings(args.ratings))
            recs = model.recommend(args.user, args.top, candidates)
            output = {"user": args.user, "mode": args.mode, "cold_start": args.user not in model.users,
                      "recommendations": [{"item_id": item, "title": catalog.get(item, {}).get("title", item),
                                           "score": prediction.score, "baseline": prediction.baseline,
                                           "fallback": prediction.fallback,
                                           "neighbors": [{"id": key, "similarity": sim, "residual": residual}
                                                         for key, sim, residual in prediction.neighbors]}
                                          for item, prediction in recs]}
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError, csv.Error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
