from pathlib import Path

import numpy as np
from scipy.io import wavfile

from Code.audio import AudioOrchestrator, AudioFeat, AudioPreproc, AudioPreprocCfg


def _write_sine(path: Path, freq: float, sr: int = 16_000, duration: float = 0.5) -> None:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    y = 0.5 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    wavfile.write(path, sr, (y * 32767).astype(np.int16))

    # -------------------------------------------------------------------------------------------------  #
    #                                 --------- Unit Test  ---------                                     #
    # -------------------------------------------------------------------------------------------------  #

def test_build_and_identify(tmp_path):
    paths = []
    labels = []
    for freq, label in zip((220.0, 440.0, 880.0), ("low", "mid", "high")):
        file_path = tmp_path / f"{label}.wav"
        _write_sine(file_path, freq)
        paths.append(file_path)
        labels.append(label)

    orchestrator = AudioOrchestrator()
    orchestrator.build_reference_from_paths(paths, labels)

    prediction = orchestrator.identify_path(paths[0])
    assert prediction in labels

    batch_predictions = orchestrator.identify_batch(paths)
    assert len(batch_predictions) == len(paths)
    assert all(pred in labels for pred in batch_predictions)