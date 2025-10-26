from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2 as cv
import numpy as np

from Code.types import ColorImageU8, MaskU8


@dataclass(frozen=True, slots=True)
class ImgPreprocV2Cfg:
    """Parámetros configurables para el pipeline basado en substracción de fondo."""

    target_size: Tuple[int, int] = (256, 256)
    blur_kernel: int = 25
    contrast_weight: float = 1.5
    smooth_mode: str = "gaussian"
    detect_edge_mode: str = "percentile"
    morph_kernel_size: Tuple[int, int] = (3, 3)


class ImgPreprocV2:
    """Pipeline de preprocesamiento robusto a iluminación basado en substracción de fondo."""

    def __init__(self, cfg: ImgPreprocV2Cfg | None = None) -> None:
        self.cfg = cfg or ImgPreprocV2Cfg()
        self.kernel = np.ones(self.cfg.morph_kernel_size, dtype=np.uint8)

    def process(self, path: Path) -> Tuple[ColorImageU8, MaskU8, Dict[str, object]]:
        """Ejecuta el pipeline completo y retorna imagen recortada, máscara y metadatos."""
        path = Path(path)
        img = self._path2img(path)
        lab = self._BGR2Lab(img)
        L_channel = lab[:, :, 0]

        L_normalized = self._normalize_luminance(L_channel)
        L_smoothed = self._smooth(L_normalized)
        edges = self._edges_with_canny(L_smoothed)
        mask = self._build_mask(edges)
        cropped_img, cropped_mask, meta = self._crop_and_normalize(img, mask)
        return cropped_img, cropped_mask, meta

    # ------------------------------------------------------------------ #
    def _path2img(self, path: Path) -> ColorImageU8:
        img = cv.imread(str(path), cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
        return img

    def _BGR2Lab(self, img: np.ndarray) -> np.ndarray:
        return cv.cvtColor(img, cv.COLOR_BGR2Lab)

    def _normalize_luminance(self, L_channel: np.ndarray) -> np.ndarray:
        """Elimina fondo suave y reescala la luminancia."""
        blur_size = max(3, int(self.cfg.blur_kernel))
        if blur_size % 2 == 0:
            blur_size += 1  # kernel debe ser impar

        L_float = L_channel.astype(np.float32)
        background = cv.GaussianBlur(L_float, (blur_size, blur_size), 0)
        background_weight = -(self.cfg.contrast_weight - 1.0)
        rebalance = cv.addWeighted(
            L_float,
            self.cfg.contrast_weight,
            background,
            background_weight,
            0.0,
        )

        normalized = cv.normalize(rebalance, None, 0, 255, cv.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def _smooth(self, L_channel: np.ndarray) -> np.ndarray:
        mode = getattr(self.cfg, "smooth_mode", "gaussian").lower()

        if mode == "gaussian":
            return cv.GaussianBlur(L_channel, (5, 5), 0)

        if mode == "guided":
            if not hasattr(cv, "ximgproc"):
                raise RuntimeError("Guided filter no disponible (instala opencv-contrib-python)")
            guide = L_channel
            gf = cv.ximgproc.createGuidedFilter(guide, radius=8, eps=50)
            filtered = gf.filter(L_channel)
            return np.clip(filtered, 0, 255).astype(np.uint8)

        if mode == "bilateral":
            return cv.bilateralFilter(L_channel, d=9, sigmaColor=75, sigmaSpace=75)

        raise ValueError(f"Modo de suavizado no reconocido: {mode}")

    def _edges_with_canny(self, L_channel: np.ndarray) -> np.ndarray:
        mode = getattr(self.cfg, "detect_edge_mode", "percentile").lower()

        if mode == "percentile":
            low, high = np.percentile(L_channel, (5, 95))
            lower = int(max(0, low))
            upper = int(min(255, high))
        elif mode == "stddev":
            mean, std = cv.meanStdDev(L_channel)
            lower = int(max(0, float(mean - std)))
            upper = int(min(255, float(mean + std)))
        else:
            raise ValueError(f"Modo de detección de bordes no reconocido: {mode}")

        if lower == upper:
            upper = min(255, lower + 1)
        return cv.Canny(L_channel, lower, upper)

    def _build_mask(self, edges: np.ndarray) -> MaskU8:
        edges_closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, self.kernel, iterations=2)
        contours, _ = cv.findContours(edges_closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No se detectaron contornos.")
        contour = self._get_largest_contour(contours)
        mask = np.zeros_like(edges_closed, dtype=np.uint8)
        cv.drawContours(mask, [contour], -1, color=255, thickness=-1)
        return mask

    def _crop_and_normalize(
        self,
        img: ColorImageU8,
        mask: MaskU8,
    ) -> Tuple[ColorImageU8, MaskU8, Dict[str, object]]:
        coords = cv.findNonZero(mask)
        if coords is None:
            raise ValueError("La máscara está vacía.")
        x, y, w, h = cv.boundingRect(coords)
        cropped_img = img[y : y + h, x : x + w]
        cropped_mask = mask[y : y + h, x : x + w]

        img_norm = self._resize_with_padding(cropped_img, self.cfg.target_size)
        mask_norm = self._resize_with_padding(cropped_mask, self.cfg.target_size)
        mask_norm = np.where(mask_norm > 0, 255, 0).astype(np.uint8, copy=False)

        meta = {
            "bbox": (int(x), int(y), int(w), int(h)),
            "input_shape": tuple(int(v) for v in img.shape[:2]),
            "normalized_shape": self.cfg.target_size,
        }
        return img_norm, mask_norm, meta

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resize_with_padding(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        target_h, target_w = size
        if img.size == 0:
            if img.ndim == 3:
                return np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
            return np.zeros((target_h, target_w), dtype=img.dtype)

        h, w = img.shape[:2]
        scale = min(target_h / max(1, h), target_w / max(1, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        interpolation = cv.INTER_NEAREST if ImgPreprocV2._is_mask_like(img) else cv.INTER_AREA
        resized = cv.resize(img, (new_w, new_h), interpolation=interpolation)

        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left

        border_value = 0
        if img.ndim == 3:
            border_value = [0, 0, 0]

        return cv.copyMakeBorder(resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=border_value)

    @staticmethod
    def _get_largest_contour(contours) -> np.ndarray:
        return max(contours, key=cv.contourArea)

    @staticmethod
    def _is_mask_like(img: np.ndarray) -> bool:
        if img.ndim != 2:
            return False
        if img.dtype != np.uint8:
            return False
        unique_values = np.unique(img)
        return (
            np.array_equal(unique_values, [0])
            or np.array_equal(unique_values, [255])
            or np.array_equal(unique_values, [0, 255])
        )

