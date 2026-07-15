import json
from pathlib import Path

from building_simplify.infer import load_fid_filter


def test_fid_filter_loads_unmatched_jsonl(tmp_path: Path):
    path = tmp_path / "unmatched.jsonl"
    path.write_text(
        json.dumps({"raw_fid": 7, "osm_id": "a"}) + "\n" + json.dumps({"raw_fid": 11}) + "\n",
        encoding="utf-8",
    )

    assert load_fid_filter(path) == {7, 11}
    assert load_fid_filter(None) is None
