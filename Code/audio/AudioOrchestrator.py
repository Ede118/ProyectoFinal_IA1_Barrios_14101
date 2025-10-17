from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from Code.audio.AudioFeat import AudioFeat
from Code.audio.AudioPreproc import AudioPreproc
from Code.audio.Standardizer import Standardizer
from Code.audio.KnnModel import KnnModel

if TYPE_CHECKING:
    from Code.adapters.Repositorio import Repo

AudioPath = str | Path


@dataclass(slots=True)
class AudioOrchestrator:
    """Coordinate audio preprocessing, feature extraction, and KNN inference."""

    preproc: AudioPreproc = field(default_factory=AudioPreproc)
    feat: AudioFeat = field(default_factory=AudioFeat)
    stats: Standardizer = field(default_factory=Standardizer)
    knn: KnnModel = field(default_factory=KnnModel)
    _X_store: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _y_store: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def build_reference_from_paths(self, paths: Sequence[AudioPath], labels: Sequence[str]) -> None:
        """Build the reference embedding table from file paths and labels."""
        path_list = list(paths)
        label_list = [str(lbl) for lbl in labels]
        if not path_list:
            raise ValueError("paths no puede estar vacío.")
        if len(path_list) != len(label_list):
            raise ValueError("paths y labels deben tener la misma longitud.")

        feats: list[np.ndarray] = []
        for path in path_list:
            y_proc, sr = self.preproc.process_path(path)
            vec = self.feat.extract(y_proc, sr).astype(np.float32, copy=False)
            feats.append(vec)

        X = np.stack(feats, axis=0).astype(np.float32, copy=False)
        self.stats.calculate_statistics(X)
        X_std = self.stats.transform(X)

        self.knn.upload_batch(X_std, label_list)
        self._X_store = X_std
        self._y_store = np.asarray(label_list, dtype=np.str_)

    def load_reference_from_repo(self, repo: "Repo", name: str) -> None:
        """Load reference embeddings and statistics from a repository."""
        data = repo.load_knn(name)
        stats = repo.load_model("audio", f"{name}_stats")
        mu = stats["mu"].astype(np.float32, copy=False)
        sigma = stats["sigma"].astype(np.float32, copy=False)
        X = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.str_, copy=False)

        self.stats.mu = mu
        self.stats.sigma = sigma
        self.knn.upload_batch(X, y.tolist())
        self._X_store = X
        self._y_store = y

    def save_reference_to_repo(self, repo: "Repo", name: str) -> None:
        """Persist the current reference table and standardizer to a repository."""
        self._ensure_ready()
        repo.save_knn(name, self._X_store, self._y_store)  # type: ignore[arg-type]
        repo.save_model("audio", f"{name}_stats", mu=self.stats.mu, sigma=self.stats.sigma)

    def identify_path(self, path: AudioPath) -> str:
        """Identify a command label from an audio file path."""
        self._ensure_ready()
        y_proc, sr = self.preproc.process_path(path)
        vec = self.feat.extract(y_proc, sr)
        vec_std = self.stats.transform_one(vec)
        return self.knn.predict(vec_std)

    def identify_batch(self, paths: Sequence[AudioPath]) -> list[str]:
        """Identify multiple audio files in sequence."""
        return [self.identify_path(p) for p in paths]

    def _ensure_ready(self) -> None:
        if self._X_store is None or self._y_store is None or self.stats.mu is None or self.stats.sigma is None:
            raise RuntimeError("Referencia no construida. Llamá a build_reference_from_paths o load_reference_from_repo.")


_DEFAULT_AUDIO_ORCHESTRATOR = AudioOrchestrator()


def build_reference_from_paths(paths: Sequence[AudioPath], labels: Sequence[str]) -> None:
    """Fit the shared audio orchestrator from disk paths."""
    _DEFAULT_AUDIO_ORCHESTRATOR.build_reference_from_paths(paths, labels)


def load_reference_from_repo(repo: "Repo", name: str) -> None:
    """Load audio references into the shared orchestrator from a repository."""
    _DEFAULT_AUDIO_ORCHESTRATOR.load_reference_from_repo(repo, name)


def save_reference_to_repo(repo: "Repo", name: str) -> None:
    """Persist the shared audio reference in the repository."""
    _DEFAULT_AUDIO_ORCHESTRATOR.save_reference_to_repo(repo, name)


def identify_path(path: AudioPath) -> str:
    """Identify the command label for a WAV file stored on disk."""
    return _DEFAULT_AUDIO_ORCHESTRATOR.identify_path(path)


def identify_batch(paths: Sequence[AudioPath]) -> list[str]:
    """Identify a batch of audio file paths."""
    return _DEFAULT_AUDIO_ORCHESTRATOR.identify_batch(paths)


__all__ = [
    "AudioOrchestrator",
    "build_reference_from_paths",
    "load_reference_from_repo",
    "save_reference_to_repo",
    "identify_path",
    "identify_batch",
]
