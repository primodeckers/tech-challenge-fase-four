"""Invariantes de dados; fixtures sinteticas, sem rede nem dados pessoais."""

import hashlib
import io
import json
from zipfile import ZipFile

import pandas as pd
import pytest

from credit_monitor import data
from credit_monitor.data import acquire_data, normalize_data, profile_data, split_data, write_partitions


@pytest.fixture
def frame():
    return pd.DataFrame({
        "ID": range(1, 5001),
        "LIMIT_BAL": [10000] * 5000,
        "default_next_month": [0, 0, 0, 0, 1] * 1000,
    })


def test_partitions_preserve_rows_and_stratification_without_id_overlap(frame):
    original = frame.copy(deep=True)
    parts = split_data(frame)
    assert {name: len(part) for name, part in parts.items()} == {
        "reference": 3000, "calibration": 1000, "production": 1000,
    }
    combined = pd.concat(parts.values())
    assert combined.ID.is_unique
    pd.testing.assert_frame_equal(
        combined.sort_values("ID").reset_index(drop=True), original
    )
    for part in parts.values():
        assert part.default_next_month.mean() == pytest.approx(0.2)
    pd.testing.assert_frame_equal(frame, original)


def test_seed_reproduces_membership_even_if_input_order_changes(frame):
    first = split_data(frame, seed=42)
    again = split_data(frame.sample(frac=1, random_state=7), seed=42)
    changed = split_data(frame, seed=43)
    for name in first:
        pd.testing.assert_frame_equal(first[name], again[name])
    assert set(first["reference"].ID) != set(changed["reference"].ID)


@pytest.mark.parametrize("invalid", ["small", "duplicate_id", "null_id", "bad_target", "null_target", "one_class", "missing_id", "missing_target"])
def test_invalid_structural_inputs_are_rejected(frame, invalid):
    if invalid == "small":
        frame = frame.iloc[:4999]
    elif invalid == "duplicate_id":
        frame.loc[1, "ID"] = 1
    elif invalid == "null_id":
        frame.loc[1, "ID"] = float("nan")
    elif invalid == "bad_target":
        frame.loc[1, "default_next_month"] = 2
    elif invalid == "null_target":
        frame.loc[1, "default_next_month"] = float("nan")
    elif invalid == "one_class":
        frame["default_next_month"] = 0
    elif invalid == "missing_id":
        frame = frame.drop(columns="ID")
    else:
        frame = frame.drop(columns="default_next_month")
    with pytest.raises(ValueError):
        split_data(frame)


def test_manifest_describes_actual_files_and_repeated_run_is_identical(frame, tmp_path):
    manifest = write_partitions(frame, tmp_path)
    for name, expected_rows in [("reference", 3000), ("calibration", 1000), ("production", 1000)]:
        info = manifest["partitions"][name]
        path = tmp_path / info["file"]
        assert info["rows"] == expected_rows == len(pd.read_csv(path))
        assert info["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert info["target_counts"] == {"0": expected_rows * 4 // 5, "1": expected_rows // 5}
    saved = (tmp_path / "manifest.json").read_bytes()
    assert json.loads(saved) == manifest
    write_partitions(frame, tmp_path)
    assert (tmp_path / "manifest.json").read_bytes() == saved


def test_corrupted_cached_download_is_rejected_without_overwriting(tmp_path):
    cached = tmp_path / "uci-credit.zip"
    cached.write_bytes(b"corrupt download")
    with pytest.raises(ValueError, match="SHA-256"):
        acquire_data(tmp_path)
    assert cached.read_bytes() == b"corrupt download"


def test_download_and_cache_preserve_exact_source_bytes(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("default of credit card clients.xls", b"synthetic workbook")
        archive.writestr("../unwanted.txt", b"must not extract")
    payload = buffer.getvalue()
    monkeypatch.setattr(data, "ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(data, "urlopen", lambda url, timeout: io.BytesIO(payload))
    downloaded = acquire_data(tmp_path)
    assert downloaded.read_bytes() == b"synthetic workbook"
    assert not (tmp_path.parent / "unwanted.txt").exists()

    def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(data, "urlopen", offline)
    assert acquire_data(tmp_path).read_bytes() == b"synthetic workbook"


def test_normalization_only_renames_target_preserving_unusual_values():
    raw = pd.DataFrame({"ID": [1, 2], "PAY_0": [-2, 0], "EDUCATION": [0, 6],
                        "BILL_AMT1": [-10, 0], "default payment next month": [0, 1]})
    normalized = normalize_data(raw)
    assert normalized.columns.tolist() == ["ID", "PAY_0", "EDUCATION", "BILL_AMT1", "default_next_month"]
    assert normalized.to_numpy().tolist() == [[1, -2, 0, -10, 0], [2, 0, 6, 0, 1]]
    assert "default payment next month" in raw.columns


def test_profile_reports_nulls_and_duplicates_without_returning_individual_rows():
    sample = pd.DataFrame({"ID": [1, 2, 3], "AGE": [21, 21, 79],
                           "EDUCATION": [0, 0, 6], "PAY_0": [-2, -2, 0],
                           "BILL_AMT1": [-10, -10, None], "default_next_month": [0, 0, 1]})
    profile = profile_data(sample)
    assert profile["rows"] == 3
    assert profile["duplicate_rows_without_id"] == 1
    assert profile["duplicate_ids"] == 0
    assert profile["nulls"]["BILL_AMT1"] == 1
    assert profile["numeric_ranges"]["AGE"] == {"min": 21, "max": 79}
    assert profile["categorical_codes"]["EDUCATION"] == [0, 6]
    assert profile["negative_counts"]["BILL_AMT1"] == 2
    assert "ID" not in profile["categorical_codes"]


def test_failed_regeneration_does_not_leave_a_success_manifest(frame, tmp_path):
    write_partitions(frame, tmp_path)
    with pytest.raises(ValueError):
        write_partitions(frame.iloc[:10], tmp_path)
    assert not (tmp_path / "manifest.json").exists()
