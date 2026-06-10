from src.infra.io import save_json, load_json


def test_save_json_creates_parent(tmp_path):
    nested = tmp_path / "a" / "b" / "data.json"
    save_json({"x": 1}, nested)
    assert load_json(nested)["x"] == 1
