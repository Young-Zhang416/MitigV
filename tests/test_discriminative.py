from mitigv.evaluation.discriminative import evaluate_discriminative, parse_yes_no


def test_parse_yes_no_punctuation_variants():
    assert parse_yes_no("Yes,") == "yes"
    assert parse_yes_no("No.") == "no"
    assert parse_yes_no("yesterday") is None


def test_discriminative_metrics_and_details():
    result = evaluate_discriminative(
        [{"id": 1, "question": "q", "label": "yes"}, {"id": 2, "question": "q2", "label": "no"}],
        [{"answer": "Yes, definitely"}, {"answer": "No."}],
    )
    assert result["accuracy"] == 1.0
    assert result["precision"] == result["recall"] == result["f1"] == 1.0
    assert all(row["correct"] for row in result["details"])
