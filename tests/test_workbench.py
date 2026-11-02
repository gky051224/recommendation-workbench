import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reclab.__main__ import main
from reclab.data import Rating, create_demo, load_ratings, split_ratings, validate
from reclab.evaluation import experiment, measure
from reclab.model import Recommender


class DataTests(unittest.TestCase):
    def test_arbitrary_ids_and_bom_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.csv"
            path.write_text("user_id,item_id,rating\n观众-甲,电影-99,4.5\n", encoding="utf-8-sig")
            self.assertEqual(load_ratings(path), [Rating("观众-甲", "电影-99", 4.5)])

    def test_reject_bad_scores(self):
        for score in (0, 6, math.nan, math.inf, -math.inf):
            with self.subTest(score=score), self.assertRaises(ValueError):
                validate([Rating("a", "b", score)])

    def test_reject_blank_ids(self):
        for user, item in (("", "x"), (" ", "x"), ("x", ""), ("x", "  ")):
            with self.subTest(user=user, item=item), self.assertRaises(ValueError):
                validate([Rating(user, item, 3)])

    def test_reject_duplicate_pair(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate([Rating("a", "x", 2), Rating("a", "x", 5)])

    def test_empty_ratings(self):
        with self.assertRaises(ValueError):
            validate([])

    def test_bad_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.csv"
            for body in ("wrong,columns\n1,2", "user_id,item_id,rating\na,b,nope", "user_id,item_id,rating\na,b"):
                path.write_text(body, encoding="utf-8")
                with self.subTest(body=body), self.assertRaises(ValueError):
                    load_ratings(path)

    def test_disjoint_complete_reproducible_split(self):
        rows = [Rating(user, str(item), 3) for user in ("a", "b") for item in range(20)]
        groups = split_ratings(rows, seed=13)
        self.assertEqual(groups, split_ratings(list(reversed(rows)), seed=13))
        sets = [set(group) for group in groups]
        self.assertEqual(set.union(*sets), set(rows))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertNotEqual(groups, split_ratings(rows, seed=14))

    def test_sparse_users_retained_in_training(self):
        rows = [Rating("dense", str(i), 3) for i in range(3)] + [Rating("sparse", "x", 4)]
        train, validation, test = split_ratings(rows)
        self.assertIn(rows[-1], train)
        self.assertEqual((len(train), len(validation), len(test)), (2, 1, 1))

    def test_too_small_for_holdout(self):
        with self.assertRaisesRegex(ValueError, "three"):
            split_ratings([Rating("a", "x", 3)])

    def test_invalid_split_ratios(self):
        for validation, test in ((0, .1), (.5, .5), (math.nan, .1), (.1, -1)):
            with self.subTest(validation=validation), self.assertRaises(ValueError):
                split_ratings([Rating("a", str(i), 3) for i in range(3)], validation_ratio=validation, test_ratio=test)

    def test_demo_is_reproducible_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            a, _ = create_demo(Path(directory) / "a")
            b, _ = create_demo(Path(directory) / "b")
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(len(load_ratings(a)), 1440)
            with self.assertRaises(ValueError):
                create_demo(a.parent)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.rows = [Rating("a", "x", 5), Rating("a", "y", 1),
                     Rating("b", "x", 1), Rating("b", "y", 5)]

    def test_baseline_hand_computed_balanced_data(self):
        model = Recommender("baseline").fit(self.rows)
        self.assertAlmostEqual(model.predict("new", "new").score, 3)
        self.assertAlmostEqual(model.predict("a", "x").score, 3)

    def test_unknown_user_and_item_fallback(self):
        for mode in ("user", "item"):
            model = Recommender(mode).fit(self.rows)
            for user, item in (("new", "x"), ("a", "new"), ("new", "new")):
                with self.subTest(mode=mode, user=user, item=item):
                    prediction = model.predict(user, item)
                    self.assertTrue(prediction.fallback)
                    self.assertAlmostEqual(prediction.score, 3)

    def test_no_shared_observations(self):
        rows = [Rating("a", "x", 5), Rating("b", "y", 1)]
        for mode in ("user", "item"):
            result = Recommender(mode).fit(rows).predict("a", "y")
            self.assertTrue(result.fallback)
            self.assertTrue(1 <= result.score <= 5)

    def test_zero_residuals_do_not_divide_by_zero(self):
        rows = [Rating("a", "x", 3), Rating("a", "y", 3), Rating("b", "x", 3)]
        self.assertEqual(Recommender("item").fit(rows).predict("b", "y").score, 3)

    def test_exclude_seen_items_and_deterministic_ties(self):
        rows = self.rows + [Rating("c", "z", 3)]
        model = Recommender("baseline").fit(rows)
        self.assertEqual([item for item, _ in model.recommend("a")], ["z"])
        self.assertEqual([item for item, _ in model.recommend("new")], ["x", "y", "z"])

    def test_candidate_filter_deduplicates(self):
        model = Recommender().fit(self.rows)
        self.assertEqual([item for item, _ in model.recommend("a", candidates=["x", "z", "z"])], ["z"])
        self.assertEqual(model.recommend("a", candidates=[]), [])

    def test_refit_clears_similarity_cache(self):
        model = Recommender("user").fit(self.rows)
        model.predict("a", "y")
        self.assertTrue(model._similarities)
        model.fit([Rating("new", "new", 4)])
        self.assertFalse(model._similarities)
        self.assertEqual(set(model.users), {"new"})

    def test_predict_requires_fit(self):
        with self.assertRaises(ValueError):
            Recommender().predict("a", "x")
        with self.assertRaises(ValueError):
            Recommender().recommend("a")

    def test_invalid_model_options(self):
        for options in ({"k": 0}, {"k": -1}, {"k": True}, {"k": 1.5}, {"mode": "bad"},
                        {"shrinkage": -1}, {"shrinkage": math.nan}, {"regularization": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Recommender(**options)

    def test_invalid_top_n(self):
        model = Recommender().fit(self.rows)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                model.recommend("a", value)

    def test_real_neighbor_evidence_reconstructs_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = create_demo(directory)
            for mode in ("user", "item"):
                model = Recommender(mode, k=3).fit(load_ratings(path))
                results = model.recommend("viewer-000")
                self.assertTrue(any(not prediction.fallback for _, prediction in results))
                for item, prediction in results:
                    with self.subTest(mode=mode, item=item):
                        self.assertLessEqual(len(prediction.neighbors), 3)
                        denominator = sum(row[1] for row in prediction.neighbors)
                        delta = sum(row[1] * row[2] for row in prediction.neighbors) / denominator if denominator else 0
                        self.assertAlmostEqual(prediction.score, min(5, max(1, prediction.baseline + delta)))
                        self.assertTrue(all(row[1] > 0 for row in prediction.neighbors))


class EvaluationTests(unittest.TestCase):
    def test_metric_hand_calculation(self):
        model = Recommender("baseline").fit([Rating("a", "x", 3)])
        result = measure(model, [Rating("new", "x", 5), Rating("new", "x", 2)])
        self.assertAlmostEqual(result["rmse"], math.sqrt(2.5))
        self.assertEqual(result["mae"], 1.5)
        self.assertEqual(result["fallback_rate"], 1)

    def test_metrics_require_observations(self):
        with self.assertRaises(ValueError):
            measure(None, [])

    def test_no_test_labels_used_in_fit_or_selection(self):
        train = [Rating("a", "x", 4), Rating("b", "x", 4), Rating("b", "y", 2)]
        validation = [Rating("a", "y", 3)]
        test = [Rating("a", "z", 1)]
        fitted = []

        class AuditedModel(Recommender):
            def fit(self, rows):
                fitted.append(list(rows))
                return super().fit(rows)

        with patch("reclab.evaluation.Recommender", AuditedModel):
            with patch("reclab.evaluation.split_ratings", return_value=(train, validation, test)):
                first = experiment(train, [1, 2])
            self.assertEqual(fitted, [train + validation, train, train + validation, train, train + validation])
            with patch("reclab.evaluation.split_ratings", return_value=(train, validation, [Rating("a", "z", 5)])):
                second = experiment(train, [1, 2])
        for mode in ("user", "item"):
            self.assertEqual(first["algorithms"][mode]["validation"], second["algorithms"][mode]["validation"])
            self.assertEqual(first["algorithms"][mode]["best_k"], second["algorithms"][mode]["best_k"])

    def test_bad_candidates(self):
        for candidates in ([], [0], [-1], [True], [1.2]):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                experiment([], candidates)

    def test_reproducible_report_and_tie_prefers_small_k(self):
        rows = [Rating(user, str(i), 3) for user in ("a", "b") for i in range(6)]
        a, b = experiment(rows, [5, 1, 1]), experiment(rows, [1, 5])
        self.assertEqual(a, b)
        self.assertEqual(a["algorithms"]["user"]["best_k"], 1)
        self.assertEqual(a["algorithms"]["item"]["test"]["rmse"], 0)


class CommandTests(unittest.TestCase):
    def test_full_cli_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["demo", str(target / "data")]), 0)
                report = target / "results" / "report.json"
                self.assertEqual(main(["evaluate", str(target / "data" / "ratings.csv"), "--k", "3", "5", "--report", str(report)]), 0)
            self.assertIn("baseline", json.loads(report.read_text(encoding="utf-8")))
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                main(["recommend", str(target / "data" / "ratings.csv"), "--user", "new",
                      "--catalog", str(target / "data" / "items.csv"), "--genre", "mystery", "--top", "3"])
            output = json.loads(buffer.getvalue())
            self.assertTrue(output["cold_start"])
            self.assertEqual(len(output["recommendations"]), 3)
            self.assertTrue(all(int(row["item_id"].split("-")[1]) % 4 == 2 for row in output["recommendations"]))

    def test_bad_input_has_nonzero_exit(self):
        result = subprocess.run([sys.executable, "-m", "reclab", "evaluate", "nonexistent.csv"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_genre_requires_catalog(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["recommend", "ignored.csv", "--user", "a", "--genre", "mystery"])
        self.assertEqual(caught.exception.code, 2)

    def test_bad_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "items.csv"
            for body in ("bad,columns\n", "item_id,title,genre\nx,Name,mystery\nx,Again,mystery\n"):
                catalog.write_text(body, encoding="utf-8")
                with self.subTest(body=body), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(["recommend", "ignored.csv", "--user", "a", "--catalog", str(catalog)])


if __name__ == "__main__":
    unittest.main()
