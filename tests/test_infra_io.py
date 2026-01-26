from src.infra.io import load_json_safe, save_json, load_json


def test_load_json_safe_missing_returns_default(tmp_path):
    missing = tmp_path / "missing.json"
    assert load_json_safe(missing, default={"ok": True}) == {"ok": True}


def test_save_json_creates_parent(tmp_path):
    nested = tmp_path / "a" / "b" / "data.json"
    save_json({"x": 1}, nested)
    assert load_json(nested)["x"] == 1
