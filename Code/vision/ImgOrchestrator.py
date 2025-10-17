from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from Code.types import ColorImageU8, MatF, VecF
from Code.vision.ImgPreproc import ImgPreproc
from Code.vision.ImgFeat import ImgFeat
from Code.vision.KmeansModel import KMeans

if TYPE_CHECKING:
    from Code.adapters.Repositorio import Repo

ImagePath = Union[str, Path]


@dataclass(slots=True)
class OrchestratorCfg:
    """Configuration knobs for the image pipeline."""

    dim: str = "5D"
    usar_gradiente_en_3D: bool = True
    return_intermediate: bool = False


@dataclass(slots=True)
class ImgOrchestrator:
    """High-level coordinator for the vision pipeline."""

    pre: ImgPreproc = field(default_factory=ImgPreproc)
    feat: ImgFeat = field(default_factory=ImgFeat)
    model: KMeans = field(default_factory=lambda: KMeans(k=4, random_state=42))
    class_names: list[str] = field(default_factory=list)
    class_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    oni_tau_global: float = 0.15
    cfg: OrchestratorCfg = field(default_factory=OrchestratorCfg)
    oni_tau_por_cluster: Optional[Dict[int, float]] = None
    cluster_to_label: Dict[int, str] = field(default_factory=dict, init=False)
    _last_fit_features: Optional[MatF] = field(default=None, init=False, repr=False)

    # --- fitting -----------------------------------------------------------------
    def fit_from_paths(
        self,
        paths: Sequence[ImagePath],
        *,
        k: int = 4,
        seed: int = 42,
        seeds: Optional[MatF] = None,
    ) -> "ImgOrchestrator":
        """Fit K-Means using image paths and update the cluster→label mapping."""
        
        dataset = list(paths)
        
        if not dataset:
            raise ValueError("paths no puede estar vacío.")
        
        images, labels = self._prepare_dataset(dataset)
        
        if images.shape[0] < k:
            raise ValueError("k no puede exceder la cantidad de imágenes disponibles.")

        self.model = KMeans(k=k, random_state=seed)
        fit_seeds = np.asarray(seeds, dtype=np.float32) if seeds is not None else None
        self.model.fit(images, seeds=fit_seeds)
        assignments = self.model.predict(images)

        self.cluster_to_label = self._build_mapping(assignments, labels)
        self.class_names = [self.cluster_to_label.get(i, f"cluster_{i}") for i in range(self.model.k)]
        self.oni_tau_por_cluster = None  # legacy fields no longer computed
        self._last_fit_features = images
        return self

    def train(self, matParametros: MatF, seeds: Optional[MatF] = None) -> "ImgOrchestrator":
        """Backward compatible alias that accepts pre-computed features."""
        X = np.asarray(matParametros, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("matParametros debe tener shape (N, D)")
        fit_seeds = np.asarray(seeds, dtype=np.float32) if seeds is not None else None
        self.model.fit(X, seeds=fit_seeds)
        self.cluster_to_label = {i: f"cluster_{i}" for i in range(self.model.k)}
        self.class_names = [self.cluster_to_label[i] for i in range(self.model.k)]
        self.oni_tau_por_cluster = None
        self._last_fit_features = X
        return self

    def train_from_path(
        self,
        paths: Sequence[ImagePath],
        seeds: Optional[MatF] = None,
    ) -> "ImgOrchestrator":
        """Backward compatible alias of `fit_from_paths`."""
        return self.fit_from_paths(paths, k=self.model.k, seed=self.model.random_state or 42, seeds=seeds)

    def train_from_patron(
        self,
        pattern: str,
        start: int,
        end: int,
        seeds: Optional[MatF] = None,
    ) -> "ImgOrchestrator":
        """Expand printf-style pattern into paths and fit the model."""
        if "%d" not in pattern and "%i" not in pattern:
            raise ValueError("El patrón debe contener %d o %i (o con padding).")
        if start > end:
            raise ValueError("start debe ser <= end")
        paths = [pattern % i for i in range(start, end + 1)]
        valid = [p for p in paths if Path(p).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}]
        if not valid:
            raise RuntimeError("El patrón no generó rutas válidas.")
        return self.fit_from_paths(valid, k=self.model.k, seed=self.model.random_state or 42, seeds=seeds)

    # --- prediction --------------------------------------------------------------
    def identify(self, img: np.ndarray) -> str:
        """Infer the class name for an image array."""
        vec = self._extract_feature_from_array(img)
        cluster = int(self.model.predict(vec[np.newaxis, :])[0])
        return self.cluster_to_label.get(cluster, f"cluster_{cluster}")

    def identify_path(self, path: ImagePath) -> str:
        """Infer the class name for an image loaded from disk."""
        img = self._read_image(path)
        return self.identify(img)

    def identify_batch(self, paths: Sequence[ImagePath]) -> list[str]:
        """Infer class names for a batch of image paths."""
        return [self.identify_path(path) for path in paths]

    def load_centroids_from_file(self, path: ImagePath) -> "ImgOrchestrator":
        """Load pre-computed centroids from a `.npz` file with key `C`."""
        npz_path = Path(path)
        if not npz_path.is_file():
            raise FileNotFoundError(f"No se encontró el archivo de centroides: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as data:
            if "C" not in data:
                raise KeyError("El archivo no contiene la clave 'C'.")
            centroids = data["C"].astype(np.float32, copy=False)
        self.model = KMeans(k=int(centroids.shape[0]), random_state=self.model.random_state)
        self.model.matCentroides_ = centroids
        self.model.vecLabels_ = None
        self.model.inercia_ = None
        self.cluster_to_label = {i: f"cluster_{i}" for i in range(self.model.k)}
        self.class_names = [self.cluster_to_label[i] for i in range(self.model.k)]
        self.oni_tau_por_cluster = None
        self._last_fit_features = None
        return self

    def predict_batch(self, n: int) -> Tuple[MatF, list[str]]:
        """Generate pseudo-batch predictions using the most recent fit dataset."""
        if self._last_fit_features is None:
            raise RuntimeError("No hay features recientes. Llamá a fit_from_paths primero.")
        if n <= 0:
            raise ValueError("n debe ser > 0")
        idx = np.arange(min(n, self._last_fit_features.shape[0]))
        feats = self._last_fit_features[idx]
        labels_idx = self.model.predict(feats)
        labels = [self.cluster_to_label.get(int(i), f"cluster_{int(i)}") for i in labels_idx]
        return feats, labels

    def run(self, input_: Union[ImagePath, ColorImageU8]) -> Dict[str, Any]:
        """Legacy helper used by the UI to obtain rich prediction metadata."""
        img = self._read_image(input_) if isinstance(input_, (str, Path)) else np.asarray(input_, dtype=np.uint8)
        img_norm, mask = self.pre.process(img, float_mask=False)
        vec = self.feat.shape_vector(
            img_norm,
            mask,
            dim=self.cfg.dim,
            usar_gradiente_en_3D=self.cfg.usar_gradiente_en_3D,
        ).astype(np.float32, copy=False)
        cluster, d2 = self.model.predict_info(vec[np.newaxis, :])
        idx = int(cluster[0])
        label = self.cluster_to_label.get(idx, f"cluster_{idx}")
        color = self.class_colors.get(label, (128, 128, 128))
        payload: Dict[str, Any] = {
            "class_idx": idx,
            "class_name": label,
            "color": color,
            "d2_min": float(d2[0]),
            "vecF": vec,
            "meta": {
                "dim": self.cfg.dim,
                "usar_gradiente_en_3D": self.cfg.usar_gradiente_en_3D,
                "preproc": self.pre.cfg,
            },
        }
        if self.cfg.return_intermediate:
            payload["intermediate"] = {"img_norm": img_norm, "mask": mask}
        return payload

    def run_batch(self, inputs: Sequence[Union[ImagePath, ColorImageU8]]) -> list[Dict[str, Any]]:
        """Legacy batch interface that mirrors the old behaviour."""
        return [self.run(item) for item in inputs]

    def run_from_dir(
        self,
        root: ImagePath,
        recursive: bool = True,
        exts: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
    ) -> list[Dict[str, Any]]:
        """Legacy helper: scan a directory and run predictions for matching files."""
        base = Path(root)
        if not base.exists():
            raise FileNotFoundError(f"No existe el directorio: {base}")
        it: Iterator[Path] = base.rglob("*") if recursive else base.glob("*")
        valid = sorted(p for p in it if p.suffix.lower() in {e.lower() for e in exts})
        if not valid:
            raise RuntimeError(f"No encontré imágenes en: {base}")
        return self.run_batch(valid)

    # --- helpers -----------------------------------------------------------------
    def _read_image(
            self, 
            path: ImagePath
            ) -> ColorImageU8:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"No pude leer la imagen: {path}")
        return img

    def _extract_feature_from_array(
            self, 
            img: np.ndarray
            ) -> VecF:
        
        img_bgr = np.asarray(img, dtype=np.uint8)
        
        img_norm = self.pre.normalize(img_bgr)
        
        mask = self.pre.segment(img_norm)
        
        vec = self.feat.shape_vector(
            img_norm,
            mask,
            dim=self.cfg.dim,
            usar_gradiente_en_3D=self.cfg.usar_gradiente_en_3D,
        )
        
        return vec.astype(np.float32, copy=False)

    def _prepare_dataset(
            self, 
            paths: Sequence[ImagePath]
            ) -> Tuple[MatF, list[str]]:
        
        features: list[np.ndarray] = []
        labels: list[str] = []
        
        for path in paths:
            img = self._read_image(path)
            vec = self._extract_feature_from_array(img)
            features.append(vec.reshape(-1))
            label = Path(path).parent.name.strip().lower()
            labels.append(label if label else f"cluster_{len(labels)}")
        
        X = np.vstack(features).astype(np.float32, copy=False)
        
        return X, labels

    def _build_mapping(self, assignments: np.ndarray, labels: Sequence[str]) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        for cluster in range(self.model.k):
            assigned = [labels[i] for i, c in enumerate(assignments) if int(c) == cluster]
            if assigned:
                major = Counter(assigned).most_common(1)[0][0]
                mapping[cluster] = major
            else:
                mapping[cluster] = f"cluster_{cluster}"
        return mapping


# ----------------------------- module level convenience -----------------------------
_DEFAULT_ORCHESTRATOR = ImgOrchestrator()


def fit_from_paths(paths: Sequence[ImagePath], *, k: int = 4, seed: int = 42) -> None:
    """Fit the shared orchestrator from raw image paths."""
    _DEFAULT_ORCHESTRATOR.fit_from_paths(paths, k=k, seed=seed)


def load_centroids_from_file(path: ImagePath) -> None:
    """Load centroids into the shared orchestrator."""
    _DEFAULT_ORCHESTRATOR.load_centroids_from_file(path)


def identify(img: np.ndarray) -> str:
    """Identify the class name for an already-loaded image."""
    return _DEFAULT_ORCHESTRATOR.identify(img)


def identify_path(path: ImagePath) -> str:
    """Identify the class for an image stored on disk."""
    return _DEFAULT_ORCHESTRATOR.identify_path(path)


def identify_batch(paths: Sequence[ImagePath]) -> list[str]:
    """Identify a batch of image paths."""
    return _DEFAULT_ORCHESTRATOR.identify_batch(paths)


def fit_from_repo(repo: "Repo", split: str = "train") -> None:
    """Fit the shared orchestrator using the paths yielded by a repository split."""
    paths = list(repo.iter_split("vision", split))  # type: ignore[attr-defined]
    fit_from_paths(paths, k=_DEFAULT_ORCHESTRATOR.model.k, seed=_DEFAULT_ORCHESTRATOR.model.random_state or 42)


def identify_from_repo(repo: "Repo", split: str = "test", n: Optional[int] = None) -> list[str]:
    """Identify images coming from a repository split."""
    paths = list(repo.iter_split("vision", split))  # type: ignore[attr-defined]
    if n is not None:
        paths = paths[:n]
    return identify_batch(paths)


__all__ = [
    "ImgOrchestrator",
    "OrchestratorCfg",
    "fit_from_paths",
    "load_centroids_from_file",
    "identify",
    "identify_path",
    "identify_batch",
    "fit_from_repo",
    "identify_from_repo",
]
