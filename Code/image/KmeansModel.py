"""
Implementación ligera de K-Means + normalización estándar sin depender de scikit-learn.

- Permite registrar nombres de features y validar dimensiones.
- Expone utilitarios fit, predict, fit_predict y get_centers con escalado reversible.
- Mantiene la clase `KMeans` como compatibilidad retro, emitiendo advertencias.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np

from Code.types import MatF, VecF, VecI

# --------------------------------------------------------------------------- #
# Utilitarios internos (normalizador y K-Means básico con inicialización k++ )
# --------------------------------------------------------------------------- #
class _StandardScaler:
    """Normalizador simple que replica el comportamiento esencial de StandardScaler."""

    def __init__(self) -> None:
        self.mean_: Optional[MatF] = None
        self.scale_: Optional[MatF] = None

    def fit(self, data: MatF) -> "_StandardScaler":
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("Los datos deben tener shape (N, D).")
        mean = arr.mean(axis=0)
        scale = arr.std(axis=0)
        scale[scale < 1e-12] = 1.0  # evita divisiones por cero en features constantes
        self.mean_ = mean
        self.scale_ = scale
        return self

    def transform(self, data: MatF) -> MatF:
        self._assert_ready()
        arr = np.asarray(data, dtype=np.float64)
        return (arr - self.mean_) / self.scale_

    def fit_transform(self, data: MatF) -> MatF:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: MatF) -> MatF:
        self._assert_ready()
        arr = np.asarray(data, dtype=np.float64)
        return (arr * self.scale_) + self.mean_

    def _assert_ready(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler no entrenado. Llamá primero a fit().")


class _KMeans:
    """K-Means mínimo viable con inicialización k-means++ y múltiples reinicios."""

    def __init__(
        self,
        n_clusters: int,
        random_state: Optional[int],
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 4,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters debe ser > 0")
        if max_iter <= 0:
            raise ValueError("max_iter debe ser > 0")
        if n_init <= 0:
            raise ValueError("n_init debe ser > 0")
        self.n_clusters = int(n_clusters)
        self.random_state = random_state
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self.cluster_centers_: Optional[MatF] = None
        self.labels_: Optional[VecI] = None
        self.inertia_: Optional[float] = None

    # ------------------------------------------------------------------ #
    def fit(self, data: MatF) -> "_KMeans":
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("X debe tener shape (N, D)")
        n_samples = arr.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError("n_clusters no puede superar la cantidad de muestras.")

        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf
        best_centers = None
        best_labels = None

        for _ in range(self.n_init):
            centers = self._init_centroids(arr, rng)
            for _ in range(self.max_iter):
                distances = self._squared_distances(arr, centers)
                labels = np.argmin(distances, axis=1)
                new_centers = self._recompute_centers(arr, labels, rng)
                shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
                centers = new_centers
                if shift <= self.tol:
                    break

            inertia = float(np.sum(np.min(self._squared_distances(arr, centers), axis=1)))
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = np.argmin(self._squared_distances(arr, centers), axis=1)

        self.cluster_centers_ = np.asarray(best_centers, dtype=np.float64)
        self.labels_ = np.asarray(best_labels, dtype=np.int64)
        self.inertia_ = float(best_inertia)
        return self

    # ------------------------------------------------------------------ #
    def predict(self, data: MatF) -> VecI:
        self._assert_fitted()
        arr = np.asarray(data, dtype=np.float64)
        distances = self._squared_distances(arr, self.cluster_centers_)
        return np.argmin(distances, axis=1).astype(np.int64, copy=False)

    # ------------------------------------------------------------------ #
    def transform(self, data: MatF) -> MatF:
        """Devuelve distancias euclidianas a cada centro."""

        self._assert_fitted()
        arr = np.asarray(data, dtype=np.float64)
        distances = self._squared_distances(arr, self.cluster_centers_)
        return np.sqrt(distances, dtype=np.float64)

    # ------------------------------------------------------------------ #
    def _init_centroids(self, data: MatF, rng: np.random.Generator) -> MatF:
        """Inicialización k-means++."""

        n_samples = data.shape[0]
        indices = []
        first_idx = rng.integers(0, n_samples)
        indices.append(first_idx)

        while len(indices) < self.n_clusters:
            centers = data[indices]
            distances = np.min(self._squared_distances(data, centers), axis=1)
            total = distances.sum()
            if total <= 0.0 or not np.isfinite(total):
                # si todas las distancias son cero, elegir un punto aleatorio
                next_idx = rng.integers(0, n_samples)
            else:
                probs = distances / total
                next_idx = int(rng.choice(n_samples, p=probs))
            if next_idx not in indices:
                indices.append(next_idx)
            else:
                # fuerza diversidad cuando choice devuelve índice repetido
                indices.append(int(rng.integers(0, n_samples)))
        return data[indices]

    def _recompute_centers(
        self,
        data: MatF,
        labels: VecI,
        rng: np.random.Generator,
    ) -> MatF:
        centers = []
        for k in range(self.n_clusters):
            mask = labels == k
            if not np.any(mask):
                idx = rng.integers(0, data.shape[0])
                centers.append(data[idx])
            else:
                centers.append(data[mask].mean(axis=0))
        return np.vstack(centers)

    @staticmethod
    def _squared_distances(data: MatF, centers: MatF) -> MatF:
        diff = data[:, None, :] - centers[None, :, :]
        return np.sum(diff * diff, axis=2, dtype=np.float64)

    def _assert_fitted(self) -> None:
        if self.cluster_centers_ is None:
            raise RuntimeError("El modelo no está entrenado. Llamá primero a fit().")


# --------------------------------------------------------------------------- #
# API pública
# --------------------------------------------------------------------------- #
class KMeansModel:
    """Wrapper sobre `_StandardScaler` + `_KMeans` con trazabilidad de features."""

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: Optional[int] = 0,
        *,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 4,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters debe ser > 0")
        self.n_clusters = int(n_clusters)
        self.random_state = random_state
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self.scaler_: Optional[_StandardScaler] = None
        self.model_: Optional[_KMeans] = None
        self.feature_names_: Optional[List[str]] = None
        self.dim_: Optional[int] = None

    # ------------------------------------------------------------------ #
    def fit(self, X: MatF, feature_names: Optional[Sequence[str]] = None) -> "KMeansModel":
        """Ajusta scaler + K-Means y registra dimensión/feature names."""

        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("X debe tener shape (N, D)")
        n_samples, dim = data.shape
        if n_samples < self.n_clusters:
            raise ValueError("n_clusters no puede superar la cantidad de muestras.")

        self.dim_ = dim
        if feature_names is not None:
            feature_names = list(feature_names)
            if len(feature_names) != dim:
                raise ValueError(f"feature_names debe tener longitud {dim}, recibió {len(feature_names)}")
            self.feature_names_ = feature_names
        else:
            self.feature_names_ = [f"f{i}" for i in range(dim)]

        self.scaler_ = _StandardScaler()
        data_scaled = self.scaler_.fit_transform(data)

        self.model_ = _KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
        )
        self.model_.fit(data_scaled)
        return self

    # ------------------------------------------------------------------ #
    def predict(self, X: MatF) -> VecI:
        """Predice labels validando que la dimensión coincida con la usada en fit."""

        self._assert_fitted()
        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("X debe tener shape (N, D)")
        if data.shape[1] != self.dim_:
            raise ValueError(
                f"Se esperaban vectores con D={self.dim_} (features={self.feature_names_}), "
                f"pero se recibió shape {data.shape}"
            )
        data_scaled = self.scaler_.transform(data)
        labels = self.model_.predict(data_scaled)
        return labels.astype(np.int64, copy=False)

    # ------------------------------------------------------------------ #
    def predict_one(self, vec: VecF) -> int:
        """Atajo para predecir una sola muestra."""

        sample = np.asarray(vec, dtype=np.float64)
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        elif sample.ndim != 2 or sample.shape[0] != 1:
            raise ValueError("predict_one acepta un vector (D,) o (1, D)")
        return int(self.predict(sample)[0])

    # ------------------------------------------------------------------ #
    def fit_predict(
        self,
        X: MatF,
        feature_names: Optional[Sequence[str]] = None,
    ) -> VecI:
        """Ajusta el modelo y devuelve los labels en un único paso."""

        return self.fit(X, feature_names=feature_names).predict(X)

    # ------------------------------------------------------------------ #
    def get_centers(self) -> Tuple[MatF, Optional[List[dict]]]:
        """
        Retorna centroides desescalados y, si hay nombres, la versión mapeada.

        Returns
        -------
        centers : MatF
            Centroides en el espacio original (shape (k, D)).
        named_centers : list[dict] | None
            Lista de diccionarios {feature: valor} por centroide.
        """

        self._assert_fitted()
        centers_scaled = self.model_.cluster_centers_
        centers = self.scaler_.inverse_transform(centers_scaled)
        named = None
        if self.feature_names_:
            named = [
                {name: float(val) for name, val in zip(self.feature_names_, center)}
                for center in centers
            ]
        return centers.astype(np.float64, copy=False), named

    # ------------------------------------------------------------------ #
    def _assert_fitted(self) -> None:
        if self.model_ is None or self.scaler_ is None or self.dim_ is None:
            raise RuntimeError("El modelo no está entrenado. Llamá primero a fit().")


# ------------------------------------------------------------------------------ #
# Legacy wrapper for backwards compatibility
# ------------------------------------------------------------------------------ #
class KMeans(KMeansModel):
    """Compatibilidad con la API anterior basada en KMeans propio."""

    def __init__(
        self,
        k: int,
        tol: float = 1e-4,
        max_iter: int = 100,
        random_state: Optional[int] = None,
    ) -> None:
        warnings.warn(
            "KMeans está deprecado. Usa KMeansModel para obtener escalado y utilitarios.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            n_clusters=k,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
        )
        self._legacy_tol = tol
        self._legacy_max_iter = max_iter

    def fit(self, matParametros: MatF, seeds: Optional[MatF] = None) -> "KMeans":
        if seeds is not None:
            warnings.warn(
                "El parámetro seeds se ignora en la versión actual de KMeans.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().fit(matParametros)
        return self

    def predict(self, matParametros: MatF) -> VecI:
        return super().predict(matParametros)

    def predict_one(self, vec: VecF) -> int:
        return super().predict_one(vec)

    def fit_predict(self, matParametros: MatF) -> VecI:
        return super().fit_predict(matParametros)

    def predict_info(self, matParametros: MatF) -> Tuple[VecI, MatF]:
        warnings.warn(
            "predict_info está deprecado; usa KMeansModel.get_centers para analizar distancias.",
            DeprecationWarning,
            stacklevel=2,
        )
        labels = self.predict(matParametros)
        data = np.asarray(matParametros, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data_scaled = self.scaler_.transform(data)
        dist = self.model_.transform(data_scaled)
        min_dist = np.min(dist, axis=1)
        return labels, (min_dist ** 2).astype(np.float64, copy=False)
