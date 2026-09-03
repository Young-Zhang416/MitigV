from mitigv.evaluation.chair import ChairEvaluator, load_chair_synonyms


def test_official_synonym_table_has_80_coco_groups():
    synonyms = load_chair_synonyms()
    assert synonyms["automobile"] == "car"
    assert synonyms["traffic signal"] == "traffic light"


def test_chair_details_and_bootstrap_metrics():
    result = ChairEvaluator({1: {"person", "car"}, 2: {"dog"}}).evaluate(
        [
            {"image_id": 1, "caption": "A man in an automobile."},
            {"image_id": 2, "caption": "A dog beside a bicycle."},
        ],
        bootstrap_samples=1000,
        seed=7,
    )
    first, second = result["details"]
    assert first["mentioned_objects"] == ["person", "car"]
    assert first["hallucinated"] == []
    assert first["recalled"] == ["car", "person"]
    assert first["missed"] == []
    assert second["hallucinated"] == ["bicycle"]
    assert second["recalled"] == ["dog"]
    assert second["missed"] == []
    for metric in result["summary"].values():
        assert len(metric["ci95"]) == 2
        assert metric["ci95"][0] <= metric["ci95"][1]
