from pathlib import Path

import pytest

from src.data.loader import find_raw_json


def test_find_raw_json_explicit_path(tmp_path):
    test_file = tmp_path / "test.json"
    test_file.write_text("{}")

    result = find_raw_json(str(test_file))
    assert result == test_file


def test_find_raw_json_nonexistent(tmp_path):
    with pytest.raises(FileNotFoundError, match="Raw JSON not found"):
        find_raw_json(str(tmp_path / "nonexistent.json"))
