from mitigv.evaluation.length_analysis import analyze_length_control


def test_length_analysis_outputs_adjusted_metrics_and_scatter():
    rows = [
        {"word_count": 5, "mentioned_objects": ["cat"], "hallucinated": []},
        {"word_count": 10, "mentioned_objects": ["cat", "dog"], "hallucinated": ["dog"]},
    ]
    result = analyze_length_control({"baseline": rows, "method": rows}, baseline="baseline")
    assert "length_adjusted_chairi" in result["configurations"]["method"]
    assert result["configurations"]["baseline"]["residual_gain"] == 0
    assert len(result["length_chairi_scatter"]) == 2
