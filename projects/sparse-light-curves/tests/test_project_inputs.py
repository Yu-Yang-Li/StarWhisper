import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("external", [False, True])
def test_training_paths_do_not_depend_on_working_directory(tmp_path, external):
    env = os.environ.copy()
    env.pop("STARWHISPER_SPARSE_ROOT", None)
    if external:
        env["STARWHISPER_SPARSE_ROOT"] = str(tmp_path / "external data")
    code = '''
import json, sys
sys.path.insert(0, sys.argv[1])
import experiment_registry as registry
import split_utils
print(json.dumps([str(registry.BASE), str(split_utils.BASE_DIR)]))
'''
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code, str(PROJECT / "train_models")],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    expected = tmp_path / "external data" if external else PROJECT
    assert [Path(path) for path in json.loads(completed.stdout)] == [expected, expected]


def test_lightcurve_paths_relocate_without_changing_the_sample(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT / "data"))
    helper = importlib.import_module("e2e_utils")
    expected = tmp_path / "train2" / "RR" / "star.csv"
    expected.parent.mkdir(parents=True)
    expected.write_text("mjd,mag\n1,17\n", encoding="utf-8")
    for recorded in ["train2/RR/star.csv", r"train2\RR\star.csv",
                     "/root/shared-nvme/train2/RR/star.csv",
                     r"E:\ZTF_variables\train2\RR\star.csv"]:
        assert helper.resolve_csv_path(recorded, tmp_path) == expected
    assert helper.resolve_csv_path("train2/RR/missing.csv", tmp_path) is None
    fixed_length = tmp_path / "train4" / "missing.csv"
    fixed_length.parent.mkdir()
    fixed_length.write_text("mjd,mag\n1,17\n", encoding="utf-8")
    assert helper.resolve_csv_path("train2/RR/missing.csv", tmp_path) is None


def test_external_data_does_not_relocate_the_feature_extractor_source(tmp_path, monkeypatch):
    monkeypatch.setenv("STARWHISPER_SPARSE_ROOT", str(tmp_path))
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.syspath_prepend(str(PROJECT / "data"))
    module = importlib.import_module("build_handcrafted_features")
    for name in ("1117", "1121"):
        extractor = module._get_extractor(name)
        assert Path(extractor.__file__).parent == PROJECT / "features"
        assert callable(extractor.extract_features_from_file)
    assert (tmp_path / "extract_features_train4.log").is_file()
    assert (tmp_path / "extract_features_reduced.log").is_file()
    assert list(caller.iterdir()) == []


def test_variable_length_input_keeps_time_magnitude_and_band(tmp_path, monkeypatch):
    monkeypatch.setenv("STARWHISPER_SPARSE_ROOT", str(tmp_path))
    monkeypatch.syspath_prepend(str(PROJECT / "data"))
    module = importlib.import_module("prepare_e2e_varlen")
    path = tmp_path / "curve.csv"
    pd.DataFrame({"mjd": [4, 1, 2], "mag": [17.4, 17.1, 17.2],
                  "filtercode": ["zr"] * 3}).to_csv(path, index=False)
    data, band = module.load_varlen_curve(path)
    assert band == 2
    np.testing.assert_allclose(data, [[0, 1, 3], [17.1, 17.2, 17.4], [2, 2, 2]])


@pytest.mark.parametrize("columns,name,expected", [
    ({"filtercode": ["zg"]}, "curve.csv", 1),
    ({"filtercode": ["zr"]}, "curve.csv", 2),
    ({"fid": [3]}, "curve.csv", 3),
    ({}, "curve_zg.csv", 1),
    ({}, "curve.csv", 4),
])
def test_band_codes_match_fixed_length_preprocessing(monkeypatch, columns, name, expected):
    monkeypatch.syspath_prepend(str(PROJECT / "data"))
    helper = importlib.import_module("e2e_utils")
    assert helper.encode_band_scalar(pd.DataFrame(columns), Path(name)) == expected
