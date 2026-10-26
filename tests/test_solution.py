"""Tests for the HanuAi ML assessment solution."""

from __future__ import annotations

import re
import unittest

import pandas as pd

from hanuai.graded import load_graded_reviews
from hanuai.pipeline import task1_export_frame, task2_export_frame
from hanuai.sentiment import analyze_reviews, classify_review, polarity_from_compound
from hanuai.task2.clustering import cluster_texts
from hanuai.task2.eda import run_eda
from hanuai.task2.text_mining import (
    assign_issue_categories,
    find_redacted_contexts,
    mine_dataframe,
    normalise_for_matching,
)


class GradedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_graded_reviews()

    def test_graded_set_has_21_unique_reviews(self) -> None:
        self.assertEqual(len(self.frame), 21)
        self.assertEqual(self.frame["ReviewID"].nunique(), 21)
        self.assertEqual(self.frame["PrimaryKey"].nunique(), 21)

    def test_dates_are_iso(self) -> None:
        self.assertTrue(self.frame["ReviewDate"].map(lambda value: bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))).all())

    def test_graded_rows_are_labelled(self) -> None:
        self.assertEqual(self.frame["Dataset"].nunique(), 1)
        self.assertIn("Graded", self.frame["Dataset"].iloc[0])


class SentimentTests(unittest.TestCase):
    def test_empty_review_is_neutral_uncategorized(self) -> None:
        result = classify_review("", "")
        self.assertEqual(result["Polarity"], "Neutral")
        self.assertEqual(result["AspectTags"], ["Uncategorized"])

    def test_neutral_band(self) -> None:
        self.assertEqual(polarity_from_compound(0.0), "Neutral")
        self.assertEqual(polarity_from_compound(0.049), "Neutral")
        self.assertEqual(polarity_from_compound(0.05), "Positive")
        self.assertEqual(polarity_from_compound(-0.05), "Negative")

    def test_aspect_tags_have_shape(self) -> None:
        result = classify_review(
            "Comfortable headphones",
            "The sound quality is excellent and the battery lasts for days. Great value.",
        )
        self.assertEqual(result["Polarity"], "Positive")
        self.assertGreaterEqual(len(result["AspectTags"]), 2)
        for tag in result["AspectTags"]:
            self.assertRegex(tag, r"^[A-Za-z &/]+ \((Pos|Neg|Neu)\)$")

    def test_classification_is_deterministic(self) -> None:
        first = classify_review("Too tight", "These are painful to wear and clamp too hard.")
        second = classify_review("Too tight", "These are painful to wear and clamp too hard.")
        self.assertEqual(first, second)
        self.assertEqual(first["Polarity"], "Negative")

    def test_graded_analysis_is_reproducible(self) -> None:
        frame = load_graded_reviews()
        first = analyze_reviews(frame)[["Polarity", "CompoundScore", "Aspect_Tag"]]
        second = analyze_reviews(frame)[["Polarity", "CompoundScore", "Aspect_Tag"]]
        pd.testing.assert_frame_equal(first, second)

    def test_task1_export_labels_sources(self) -> None:
        graded = analyze_reviews(load_graded_reviews())
        export = task1_export_frame(graded)
        self.assertIn("Polarity", export.columns)
        self.assertIn("Aspect_Tag", export.columns)
        self.assertEqual(export["Dataset"].nunique(), 1)
        self.assertTrue((export["CollectionMethod"] != "").all())
        self.assertTrue((export["SortFilter"] != "").all())


class Task2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.eda = run_eda(make_figures=False)
        cls.mined = mine_dataframe(cls.eda["frame"])

    def test_dataset_shape_and_duplicates(self) -> None:
        self.assertEqual(self.eda["shape"], (1000, 20))
        self.assertEqual(self.eda["duplicates"], 0)
        self.assertEqual(len(self.eda["profile"]), 20)

    def test_sparse_fields_reported_not_plotted(self) -> None:
        self.assertIn("Trigger", self.eda["sparse_columns"])
        profile = self.eda["profile"].set_index("Column")
        self.assertAlmostEqual(float(profile.loc["Trigger", "NonNullFraction"]), 0.005, places=3)

    def test_verbatim_columns_are_text(self) -> None:
        profile = self.eda["profile"].set_index("Column")
        for column in ("CAUSAL_VERBATIM", "CORRECTION_VERBATIM", "CUSTOMER_VERBATIM"):
            self.assertEqual(profile.loc[column, "Role"], "text")

    def test_spanish_normalisation(self) -> None:
        normalised = normalise_for_matching("PANTALLA SE PONE NEGRA, el modulo no funciona")
        self.assertIn("black screen", normalised)
        self.assertIn("module", normalised)
        self.assertIn("inoperative", normalised)
        self.assertIn("display", normalise_for_matching("la pantalla esta oscura"))

    def test_redacted_code_context_is_preserved(self) -> None:
        contexts = find_redacted_contexts("performed SPS programming but error code   appears on screen")
        self.assertTrue(contexts)
        self.assertTrue(any("[REDACTED]" in context for context in contexts))

    def test_issue_categories_are_valid(self) -> None:
        valid = {
            "Component Failure",
            "Electrical Issue",
            "Software Glitch",
            "Assembly Quality",
        }
        categories = assign_issue_categories("radio screen blank and wiring shorted", "Module/Component-Shorted")
        self.assertGreaterEqual(len(categories), 1)
        self.assertTrue(set(categories).issubset(valid))

    def test_every_record_is_categorised(self) -> None:
        self.assertTrue(self.mined["IssueCategories"].map(len).ge(1).all())
        self.assertTrue(self.mined["PrimaryIssueCategory"].isin(
            ["Component Failure", "Electrical Issue", "Software Glitch", "Assembly Quality"]
        ).all())

    def test_redaction_and_spanish_flags_exist(self) -> None:
        self.assertGreater(int(self.mined["RedactedCodeFlag"].sum()), 0)
        self.assertGreater(int(self.mined["SpanishFlag"].sum()), 0)

    def test_task2_export_contains_tags_and_cluster(self) -> None:
        export = task2_export_frame(self.mined)
        for column in (
            "IssueCategory",
            "PrimaryIssueCategory",
            "ComponentsText",
            "SymptomsText",
            "ErrorCodesText",
            "EntityTags",
        ):
            self.assertIn(column, export.columns)
        self.assertEqual(len(export), 1000)


class ClusteringTests(unittest.TestCase):
    TEXTS = [
        "radio screen blank and replaced radio",
        "sps programming error unable to program the radio",
        "wiring shorted and lost communication on the bus",
        "rear camera shows black screen after reverse",
        "sd card missing navigation error",
        "battery drain and no power to the module",
        "bluetooth pairing fails and no sound from speakers",
        "screen flickers intermittently while driving",
        "amplifier replaced due to internal fault",
        "usb update failed and system reboots",
        "cluster gauges inoperative after battery replacement",
        "onstar cannot connect to network",
    ] * 3

    def test_manual_fallback_is_deterministic(self) -> None:
        first = cluster_texts(self.TEXTS, n_clusters=3, prefer_sklearn=False)
        second = cluster_texts(self.TEXTS, n_clusters=3, prefer_sklearn=False)
        self.assertEqual(first.backend, "numpy fallback")
        self.assertEqual(first.labels.tolist(), second.labels.tolist())
        self.assertEqual(first.inertia, second.inertia)
        self.assertEqual(len(set(first.labels.tolist())), 3)

    def test_top_terms_are_returned(self) -> None:
        result = cluster_texts(self.TEXTS, n_clusters=2, prefer_sklearn=False)
        self.assertEqual(len(result.top_terms), 2)
        for terms in result.top_terms.values():
            self.assertTrue(terms)


if __name__ == "__main__":
    unittest.main()
