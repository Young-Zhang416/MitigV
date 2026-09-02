"""Tests for :mod:`evaluators.pope`."""

import pytest

from evaluators.pope import (
    METRIC_NAMES,
    REFERENCE,
    compare_to_reference,
    compute_metrics,
)


class TestComputeMetrics:
    def test_perfect_predictions(self):
        gt = ["yes", "no", "yes", "no"]
        gen = ["yes", "no", "yes", "no"]
        m = compute_metrics(gt, gen)
        assert m["accuracy"] == 100.0
        assert m["precision"] == 100.0
        assert m["recall"] == 100.0
        assert m["f1"] == 100.0

    def test_substring_matching_like_official(self):
        # Official eval uses substring: "yes" in answer / "no" in answer.
        gt = ["yes", "no"]
        gen = ["Yes, there is.", "No."]
        m = compute_metrics(gt, gen)
        assert m["accuracy"] == 100.0

    def test_false_positives_and_negatives(self):
        gt = ["yes", "yes", "no", "no"]
        gen = ["yes", "no", "yes", "no"]  # item1 FN, item2 FP
        m = compute_metrics(gt, gen)
        # TP=1, FN=1, TN=1, FP=1
        assert m["accuracy"] == 50.0
        assert m["precision"] == 50.0
        assert m["recall"] == 50.0
        assert m["f1"] == 50.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            compute_metrics(["yes"], ["yes", "no"])

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError, match="unexpected ground-truth"):
            compute_metrics(["maybe"], ["yes"])


class TestCompareToReference:
    def test_ignorable_when_within_tolerance(self):
        metrics = {"accuracy": 85.0, "precision": 90.0, "recall": 80.0, "f1": 85.0}
        cmp = compare_to_reference(metrics, "random", "VCD", tolerance=5.0)
        assert cmp["ignorable"] is True

    def test_not_ignorable_when_outside_tolerance(self):
        metrics = {"accuracy": 50.0, "precision": 50.0, "recall": 50.0, "f1": 50.0}
        cmp = compare_to_reference(metrics, "random", "VCD", tolerance=2.0)
        assert cmp["ignorable"] is False

    def test_reference_has_all_splits_and_methods(self):
        for split in ("random", "popular", "adversarial"):
            for method in ("Regular", "VCD"):
                assert set(METRIC_NAMES) <= set(REFERENCE[split][method])
