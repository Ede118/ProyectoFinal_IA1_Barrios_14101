from dataclasses import dataclass, field
import numpy as np
from typing import Tuple, Optional
import scipy.signal as sps

from Code.types import VecF, MatF, F32

@dataclass(slots=True)
class Standardizer:
    """
    Z-score por dimensión: x' = (x - mu)/sigma.
    """
    mu: Optional[np.ndarray] = None     # (D,)
    sigma: Optional[np.ndarray] = None  # (D,)
    eps: float = 1e-8                   # evita divisiones por ~0

    # ---- entrenamiento (calcula estadísticas de TRAIN) ----
    def calculate_statistics(
        self, 
        X: np.ndarray
        ) -> "Standardizer":
        """
        X: (N, D) float-like
        Calcula mu y sigma por columna. Devuelve self.
        """
        X = np.asarray(X, dtype=F32)
        
        if X.ndim != 2:
            raise ValueError("X debe ser 2D (N, D)")
        
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        
        # columnas casi constantes → evita división por ~0
        sigma = np.where(sigma < self.eps, 1.0, sigma).astype(F32)
        
        self.mu = mu.astype(F32)
        self.sigma = sigma
        
        return self

    # ---- chequeo interno ----
    def _check(self) -> None:
        if self.mu is None or self.sigma is None:
            raise RuntimeError("Standardizer no está ajustado. Corre fit(X) primero.")

    # ---- transformación batch ----
    def transform(
        self, 
        X: MatF
        ) -> MatF:
        """
        X: (N, D) o (D,) → retorna mismas shapes estandarizadas.
        """
        self._check()
        
        X = np.asarray(X, dtype=F32)
        
        if X.ndim == 1:
            return ((X - self.mu) / self.sigma).astype(F32)
        if X.ndim == 2:
            return ((X - self.mu[None, :]) / self.sigma[None, :]).astype(F32)
        
        raise ValueError("X debe ser 1D o 2D")

    # ---- transformación para un solo vector ----
    def transform_one(
        self, 
        x: VecF
        ) -> VecF:
        """
        x: (D,) → (D,)
        """

        self._check()
        x = np.asarray(x, dtype=F32)
        if x.ndim != 1:
            raise ValueError("x debe ser 1D (D,)")
        
        return ((x - self.mu) / self.sigma).astype(F32)

    # ---- inversa (útil para debug) ----
    def inverse_transform(
        self, 
        X: MatF
        ) -> MatF:
        """
        Revierte el z-score: X = X'*sigma + mu
        """
        self._check()
        X = np.asarray(X, dtype=F32)
        if X.ndim == 1:
            return (X * self.sigma + self.mu).astype(F32)
        if X.ndim == 2:
            return (X * self.sigma[None, :] + self.mu[None, :]).astype(F32)
        raise ValueError("X debe ser 1D o 2D")

    # ---- persistencia ----
    def save(
        self, 
        path: str
        ) -> None:
        """
        Guarda mu y sigma en .npz comprimido.
        """
        self._check()
        np.savez_compressed(path, mu=self.mu, sigma=self.sigma)

    @classmethod
    def load(
        cls, 
        path: str
        ) -> "Standardizer":
        """
        Carga desde .npz y devuelve una instancia lista para usar.
        """
        data = np.load(path)
        mu = data["mu"].astype(F32)
        sigma = data["sigma"].astype(F32)
        return cls(mu=mu, sigma=sigma)
