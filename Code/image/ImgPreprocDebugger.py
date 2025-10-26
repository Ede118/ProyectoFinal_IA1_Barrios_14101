from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Dict, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from Code.types import ColorImageU8, MaskU8
from .ImgPreproc import ImgPreproc, ImgPreprocCfg


class ImgPreprocDebugger(ImgPreproc):
    """Extiende ImgPreproc almacenando y visualizando cada etapa del pipeline."""

    def __init__(self, cfg: ImgPreprocCfg | None = None) -> None:
        super().__init__(cfg)
        self.stages: Dict[str, object] = {}

    def process(
        self,
        path: Path,
    ) -> Tuple[ColorImageU8 | None, MaskU8 | None, Dict[str, object], Dict[str, object]]:
        """Ejecuta el pipeline guardando las salidas intermedias para depuración."""

        self.stages = {}
        cropped_img: ColorImageU8 | None = None
        cropped_mask: MaskU8 | None = None
        meta: Dict[str, object] = {}

        stage = "input"
        try:
            path = Path(path)
            img = self._read_image(path)
            self.stages["input"] = img.copy()

            stage = "lab"
            lab = self._to_lab(img)
            self.stages["lab"] = lab.copy()

            stage = "L_eq"
            L_eq = self._equalize_luminance(lab)
            self.stages["L_eq"] = L_eq.copy()

            stage = "L_filtered"
            L_filtered = self._filter_adaptive(L_eq)
            self.stages["L_filtered"] = L_filtered.copy()

            stage = "edges"
            edges = self._detect_edges(L_filtered)
            self.stages["edges"] = edges.copy()

            stage = "mask"
            mask = self._build_mask(edges)
            self.stages["mask"] = mask.copy()

            stage = "crop"
            cropped_img, cropped_mask, meta = self._crop_and_normalize(img, mask)
            self.stages["cropped_img"] = cropped_img.copy()
            self.stages["cropped_mask"] = cropped_mask.copy()

        except Exception as exc:  # pylint: disable=broad-except
            self.stages["error"] = exc
            if not meta:
                meta = {"error": str(exc), "stage": stage}
            else:
                meta.setdefault("error", str(exc))
                meta.setdefault("stage", stage)

        return cropped_img, cropped_mask, meta, self.stages.copy()

    # ------------------------------------------------------------------ #
    def show(self, stage_name: str) -> None:
        """Visualiza una etapa específica del pipeline."""

        if stage_name not in self.stages:
            raise KeyError(f"Etapa desconocida: {stage_name}")

        value = self.stages[stage_name]
        if isinstance(value, Exception):
            raise RuntimeError(f"No se puede visualizar la etapa '{stage_name}': {value}") from value
        if not isinstance(value, np.ndarray):
            raise TypeError(f"La etapa '{stage_name}' no contiene una imagen visualizable.")

        plt.figure(figsize=(5, 5))
        self._imshow(value, title=stage_name)
        plt.tight_layout()
        plt.show()

    def show_all(self) -> None:
        """Visualiza todas las etapas disponibles en una cuadrícula 2×N."""

        entries = [(name, value) for name, value in self.stages.items() if isinstance(value, np.ndarray)]
        if not entries:
            raise RuntimeError("No hay etapas visualizables disponibles.")

        cols = ceil(len(entries) / 2)
        rows = 2 if len(entries) > 1 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        for ax, (name, value) in zip(axes, entries):
            self._imshow(value, title=name, ax=ax)

        for ax in axes[len(entries) :]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _imshow(img: np.ndarray, title: str, ax=None) -> None:
        """Renderiza una imagen respetando color/grises."""

        ax = ax or plt.gca()
        ax.set_title(title)
        ax.axis("off")

        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            ax.imshow(img if img.ndim == 2 else img[:, :, 0], cmap="gray")
            return

        if img.ndim == 3 and img.shape[2] == 3:
            if title.lower() == "lab":
                display = cv.cvtColor(img, cv.COLOR_Lab2RGB)
            else:
                display = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            ax.imshow(display)
            return

        raise TypeError("Formato de imagen no soportado para visualización.")
