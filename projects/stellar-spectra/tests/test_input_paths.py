import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arguments,required", [
    ([], "--data_dir"),
    (["--make_groups"], "--fits_root"),
])
def test_preprocessing_requires_the_actual_input_directory(tmp_path, arguments, required):
    completed = subprocess.run(
        [sys.executable, "-B", str(PROJECT / "scripts/preprocess_data.py"), *arguments],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8",
    )
    assert completed.returncode == 2
    assert required in completed.stderr
    assert list(tmp_path.iterdir()) == []


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, PROJECT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preprocessor_finds_the_project_vocabulary_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FIXED_VOCAB_PATH", raising=False)
    module = load_script("spectra_preprocess", "scripts/preprocess_data.py")
    assert Path(module.resolve_fixed_vocab_csv_path()) == PROJECT / "vocab/vocabulary.csv"


def test_finetuning_requires_checkpoint_and_separate_step_evaluation_input(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "finetune.log"))
    module = load_script("spectra_finetune", "scripts/finetune.py")
    for name in ("TRAIN_CSV", "VAL_CSV", "PRETRAIN_CKPT_PATH", "STEP_EVAL_CSV", "STEP_EVAL_FREQ_STEPS"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(FileNotFoundError, match="PRETRAIN_CKPT_PATH"):
        module.load_pretrained_model()
    for name in ("TRAIN_CSV", "VAL_CSV", "PRETRAIN_CKPT_PATH"):
        monkeypatch.setenv(name, str(tmp_path / name))
    with pytest.raises(FileNotFoundError, match="STEP_EVAL_CSV"):
        module.main()


def test_importing_interpolation_does_not_start_spectrum_generation(monkeypatch):
    import concurrent.futures

    def no_workers(*args, **kwargs):
        raise AssertionError("Importing a helper must not start interpolation jobs")

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", no_workers)
    module = load_script("spectra_interpolation", "code/interpolation&decrease_resolution/ShowOut/interpolation0415.py")
    assert callable(module.main)
