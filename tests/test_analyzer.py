import shutil

import pytest

from chatter import Analyzer


def test_preprocess_directory_smoke(tiny_config, random_audio_file, tmp_path):
    # Skip if ffprobe/ffmpeg are unavailable (pydub needs them for probing).
    if not shutil.which("ffprobe") and not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg/ffprobe not available in this environment")

    # Skip if optional audiocomplib is not installed (preprocessing uses it if present).
    try:
        import audiocomplib  # noqa: F401
    except ImportError:
        pytest.skip("audiocomplib not available in this environment")

    # Smoke test: ensure preprocessing writes an output file
    analyzer = Analyzer(tiny_config, n_jobs=1)
    out_dir = tmp_path / "processed"

    analyzer.preprocess_directory(
        input_dir=random_audio_file.parent,
        processed_dir=out_dir,
        batch_size=1,
    )

    expected = out_dir / random_audio_file.name
    assert expected.exists()
    assert expected.stat().st_size > 0
