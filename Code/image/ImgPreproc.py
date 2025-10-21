from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import cv2
import numpy as np

from Code.types import ColorImageU8, MaskU8


@dataclass(slots=True)
class SegMeta:
    """Metadata geométrica resultante de la segmentación y el recorte."""

    contour: np.ndarray
    rect: tuple
    centroid: tuple[float, float]
    inertia_ratio: float
    aspect_ratio: float
    circularity: float
    holes: int
    M_warp: np.ndarray


@dataclass(slots=True)
class PreprocOutput:
    """Salida canónica del preprocesamiento de imágenes."""

    img: np.ndarray
    mask: MaskU8
    meta: Optional[SegMeta]


@dataclass(slots=True)
class ImgPreprocCfg:
    """
    Configuración del pipeline de preprocesamiento.

    La intención es replicar la lógica de `tests/test_image_ImgPreproc.py`
    removiendo cualquier preocupación de visualización.
    """

    target_size: Tuple[int, int] = (512, 512)
    keep_aspect: bool = True
    illum_sigma: float = 35.0
    use_adaptive: bool = False
    open_ksize: int = 3
    close_ksize: int = 3
    min_area_ratio: float = 0.00085
    penalize_border: bool = True
    pad_ratio: float = 0.12
    rotate_crop_mode: Literal["auto", "square", "rect", "none"] = "auto"
    return_meta: bool = True
    crop_illum_ratio: float = 0.20
    use_seed_from_stage1: bool = True


@dataclass(slots=True)
class ImgPreproc:
    """
    Pipeline de preprocesamiento geométrico y fotométrico.

    - Normaliza iluminación.
    - Segmenta el objeto dominante.
    - Estima geometría para un recorte alineado.
    - Devuelve imagen y máscara ya redimensionadas a `target_size`.
    """

    cfg: ImgPreprocCfg = field(default_factory=ImgPreprocCfg)

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def process(self, img_bgr: ColorImageU8) -> PreprocOutput:
        # 1) Gris inicial
        if img_bgr.ndim == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.asarray(img_bgr, dtype=np.uint8)

        # 2) Segmentación gruesa con corrección de iluminación global
        gray_norm = self._normalize_illum(gray)
        mask0 = self._bin_mask(gray_norm, self.cfg.open_ksize, self.cfg.close_ksize)

        min_area = max(1, int(self.cfg.min_area_ratio * mask0.size))
        if cv2.countNonZero(mask0) < min_area and not self.cfg.use_adaptive:
            mask0 = self._bin_mask(
                gray_norm,
                self.cfg.open_ksize,
                self.cfg.close_ksize,
                force_adaptive=True
            )

        cnts, hier = self._contours_with_holes(mask0)
        best_score = -1.0
        best_feat: Optional[dict] = None
        best_contour: Optional[np.ndarray] = None

        for idx, contour in enumerate(cnts):
            hrow = hier[idx] if idx < len(hier) else None
            feat = self._features(mask0.shape, contour, hrow)
            if feat is None or feat["area"] < min_area:
                continue
            score = self._score(feat, mask0.shape)
            if score > best_score:
                best_score = score
                best_feat = feat
                best_contour = contour

        # Si no hay nada confiable, devolvemos algo bien formado
        if best_feat is None or best_contour is None:
            resized = self._resize_pad(gray_norm, self.cfg.target_size, self.cfg.keep_aspect)
            return PreprocOutput(
                img=self._normalize_unit(resized),
                mask=np.zeros(self.cfg.target_size, dtype=np.uint8),
                meta=None
            )

        # 3) Crop alineado (o axis-aligned si así se pidió), con un padding suave
        obj_mask = np.zeros_like(mask0, dtype=np.uint8)
        cv2.drawContours(obj_mask, [best_contour], -1, color=255, thickness=-1)

        rotate_mode = self.cfg.rotate_crop_mode
        square_crop = False
        use_rotation = True
        if rotate_mode == "auto":
            if best_feat["holes"] >= 1 and best_feat["circularity"] > 0.6:
                square_crop = True
            elif best_feat["aspect_ratio"] >= 2.0 and best_feat["inertia_ratio"] < 0.25:
                square_crop = False
            else:
                square_crop = True
        elif rotate_mode == "square":
            square_crop = True
        elif rotate_mode == "rect":
            square_crop = False
        elif rotate_mode == "none":
            square_crop = False
            use_rotation = False
        else:
            raise ValueError(f"rotate_crop_mode desconocido: {rotate_mode}")

        if use_rotation:
            crop_img, crop_mask, M = self._crop_aligned(img_bgr, obj_mask, best_feat["rect"], square_crop)
        else:
            crop_img, crop_mask, M = self._crop_axis_aligned(img_bgr, obj_mask, best_feat["rect"], square_crop)

        if crop_img.size == 0 or crop_mask.size == 0:
            resized = self._resize_pad(gray_norm, self.cfg.target_size, self.cfg.keep_aspect)
            return PreprocOutput(
                img=self._normalize_unit(resized),
                mask=np.zeros(self.cfg.target_size, dtype=np.uint8),
                meta=None
            )

        # 4) Trabajar SIEMPRE en gris en el crop
        crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if crop_img.ndim == 3 else crop_img

        # 5) Redimensionado canónico y semilla opcional de etapa 1
        img_resized = self._resize_pad(crop_gray, self.cfg.target_size, self.cfg.keep_aspect)
        seed_resized = self._resize_pad(crop_mask, self.cfg.target_size, self.cfg.keep_aspect, is_mask=True)

        # 6) Nivelar iluminación a escala del crop (separación de escalas) y binarizar
        H, W = self.cfg.target_size
        sigma_crop = max(H, W) * float(self.cfg.crop_illum_ratio)
        gray_eq = self._normalize_illum_sigma(img_resized, sigma_crop)

        mask_fina0 = self._bin_mask(
            gray_eq,
            open_ksize=self.cfg.open_ksize,
            close_ksize=self.cfg.close_ksize
        )

        # 7) Opcional: recortar por la semilla para ignorar fondo ruidoso
        if self.cfg.use_seed_from_stage1:
            mask_fina0 = cv2.bitwise_and(mask_fina0, seed_resized)

        # 8) Materializar la máscara “materia = exterior − agujeros” con jerarquía
        mask_materia = self._material_mask_from(mask_fina0, prefer_seed=seed_resized if self.cfg.use_seed_from_stage1 else None)

        # 9) Salidas normalizadas
        img_norm = self._normalize_unit(img_resized)
        mask_u8 = self._ensure_mask_uint8(mask_materia)

        meta = None
        if self.cfg.return_meta:
            meta = SegMeta(
                contour=best_contour.copy(),
                rect=best_feat["rect"],
                centroid=best_feat["centroid"],
                inertia_ratio=float(best_feat["inertia_ratio"]),
                aspect_ratio=float(best_feat["aspect_ratio"]),
                circularity=float(best_feat["circularity"]),
                holes=int(best_feat["holes"]),
                M_warp=M.astype(np.float32, copy=False),
            )
        return PreprocOutput(img=img_norm, mask=mask_u8, meta=meta)


    # ------------------------------------------------------------------ #
    # Helpers privados
    # ------------------------------------------------------------------ #
    def _normalize_illum(
            self, 
            gray: np.ndarray
            ) -> np.ndarray:
        """Filtra iluminación de baja frecuencia y reescala a [0,255]."""

        bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=self.cfg.illum_sigma, sigmaY=self.cfg.illum_sigma)
        norm = cv2.addWeighted(gray, 1.0, bg, -1.0, 128.0)
        return cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _bin_mask(
            self, 
            gray_norm: np.ndarray, 
            open_ksize: int = 3,
            close_ksize: int = 3,
            *, 
            force_adaptive: Optional[bool] = None
            ) -> MaskU8:
        """Genera una máscara binaria robusta a partir de la imagen normalizada."""

        use_adaptive = self.cfg.use_adaptive if force_adaptive is None else force_adaptive
        g = cv2.GaussianBlur(gray_norm, (5, 5), 0)
        if use_adaptive:
            mask = cv2.adaptiveThreshold(
                255 - g,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                2,
            )
        else:
            _, mask = cv2.threshold(255 - g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if open_ksize > 1:
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
        if close_ksize > 1:
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
        return mask.astype(np.uint8, copy=False)

    def _contours_with_holes(self, mask: MaskU8) -> tuple[list[np.ndarray], np.ndarray]:
        """Obtiene contornos junto con su jerarquía RETR_CCOMP."""

        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is not None and len(hier) > 0:
            hier = hier[0]
        else:
            hier = np.zeros((0, 4), dtype=np.int32)
        return cnts, hier

    def _features(
        self,
        shape: Tuple[int, int],
        contour: np.ndarray,
        hierarchy_row: Optional[np.ndarray],
    ) -> Optional[dict]:
        """Extrae descriptores geométricos simples para un contorno."""

        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            return None

        perim = float(cv2.arcLength(contour, True))
        circularity = float(4.0 * np.pi * area / (perim * perim + 1e-9))

        moments = cv2.moments(contour)
        cx = float(moments["m10"] / (moments["m00"] + 1e-9))
        cy = float(moments["m01"] / (moments["m00"] + 1e-9))

        cov_xx = moments["mu20"] / (moments["m00"] + 1e-9)
        cov_yy = moments["mu02"] / (moments["m00"] + 1e-9)
        cov_xy = moments["mu11"] / (moments["m00"] + 1e-9)
        cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
        eigvals, _ = np.linalg.eig(cov + 1e-9 * np.eye(2))
        lam_min, lam_max = np.sort(eigvals)
        inertia_ratio = float(lam_min / (lam_max + 1e-9))

        rect = cv2.minAreaRect(contour)
        (_, _), (w_rect, h_rect), _ = rect
        aspect_ratio = float(max(w_rect, h_rect) / max(1.0, min(w_rect, h_rect)))

        H, W = shape
        dist_center = float(np.hypot(cx - W / 2.0, cy - H / 2.0) / np.hypot(W / 2.0, H / 2.0))

        holes = 0
        if hierarchy_row is not None and len(hierarchy_row) == 4 and int(hierarchy_row[2]) != -1:
            holes = 1

        return dict(
            area=area,
            circularity=circularity,
            inertia_ratio=inertia_ratio,
            aspect_ratio=aspect_ratio,
            centroid=(cx, cy),
            rect=rect,
            dist_center=dist_center,
            holes=holes,
        )

    def _score(self, features: Optional[dict], shape: Tuple[int, int]) -> float:
        """Puntúa un contorno en función de su tamaño y centralidad."""

        if features is None:
            return -1.0

        box = cv2.boxPoints(features["rect"]).astype(int)
        x, y, w, h = cv2.boundingRect(box)
        H, W = shape
        touches = x <= 2 or y <= 2 or (x + w) >= W - 2 or (y + h) >= H - 2
        centrality = np.exp(-((features["dist_center"] ** 2) / 0.15))
        penalty = 0.5 if (self.cfg.penalize_border and touches) else 1.0
        return float(features["area"] * centrality * penalty)

    def _crop_aligned(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        rect,
        square: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recorta alineando la caja mínima rotada del contorno."""

        (cx, cy), (w, h), theta = rect
        if square:
            side = max(w, h)
            w = h = side

        w *= 1.0 + self.cfg.pad_ratio
        h *= 1.0 + self.cfg.pad_ratio

        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        warp_img = cv2.warpAffine(
            img,
            M,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warp_mask = cv2.warpAffine(
            mask,
            M,
            (mask.shape[1], mask.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        x0 = max(0, int(round(cx - w / 2.0)))
        y0 = max(0, int(round(cy - h / 2.0)))
        x1 = min(img.shape[1], int(round(cx + w / 2.0)))
        y1 = min(img.shape[0], int(round(cy + h / 2.0)))

        return (
            warp_img[y0:y1, x0:x1],
            warp_mask[y0:y1, x0:x1],
            M.astype(np.float32, copy=False),
        )

    def _crop_axis_aligned(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        rect,
        square: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recorte axis-aligned sin aplicar rotación explícita."""

        (cx, cy), (w, h), _ = rect
        if square:
            side = max(w, h)
            w = h = side

        w *= 1.0 + self.cfg.pad_ratio
        h *= 1.0 + self.cfg.pad_ratio

        x0 = max(0, int(round(cx - w / 2.0)))
        y0 = max(0, int(round(cy - h / 2.0)))
        x1 = min(img.shape[1], int(round(cx + w / 2.0)))
        y1 = min(img.shape[0], int(round(cy + h / 2.0)))

        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return (
            img[y0:y1, x0:x1],
            mask[y0:y1, x0:x1],
            M,
        )

    def _resize_pad(
        self,
        x: np.ndarray,
        target: Tuple[int, int],
        keep_aspect: bool,
        *,
        is_mask: bool = False,
    ) -> np.ndarray:
        """Redimensiona y aplica padding centrado si se solicita."""

        H, W = target
        if x.size == 0:
            return np.zeros((H, W), dtype=x.dtype)

        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        if keep_aspect:
            ih, iw = x.shape[:2]
            scale = min(H / ih, W / iw)
            nh = max(1, int(round(ih * scale)))
            nw = max(1, int(round(iw * scale)))
            resized = cv2.resize(x, (nw, nh), interpolation=interpolation)

            top = (H - nh) // 2
            bottom = H - nh - top
            left = (W - nw) // 2
            right = W - nw - left

            border_value = 0
            if not is_mask and x.ndim == 3:
                border_value = [0, 0, 0]

            resized = cv2.copyMakeBorder(
                resized,
                top,
                bottom,
                left,
                right,
                borderType=cv2.BORDER_CONSTANT,
                value=border_value,
            )
            return resized

        return cv2.resize(x, (W, H), interpolation=interpolation)

    def _ensure_mask_uint8(self, mask: np.ndarray) -> MaskU8:
        """Convierte cualquier máscara binaria al formato uint8 {0,255}."""

        mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8, copy=False)
        return mask_u8

    def _normalize_unit(self, x: np.ndarray) -> np.ndarray:
        """Escala un arreglo al rango [0,1] en float32."""

        x = x.astype(np.float32, copy=False)
        xmin = float(x.min(initial=0.0))
        xmax = float(x.max(initial=1.0))
        if xmax - xmin > 1e-6:
            x = (x - xmin) / (xmax - xmin)
        else:
            x.fill(0.0)
        return x


    def _normalize_illum_sigma(self, gray: np.ndarray, sigma: float) -> np.ndarray:
        """
        # Versión parametrizable de la nivelación de iluminación para trabajar a escala del crop.
        """
        bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        norm = cv2.addWeighted(gray, 1.0, bg, -1.0, 128.0)
        return cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    


    


    def _material_mask_from(self, mask_bin: np.ndarray, prefer_seed: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Construye máscara de 'materia' respetando jerarquía:
        toma el mejor exterior (por solape con seed o por área) y resta sus hijos (agujeros).
        """
        cnts, hier = cv2.findContours(mask_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None or len(cnts) == 0:
            return np.zeros_like(mask_bin, dtype=np.uint8)
        hier = hier[0]

        # elegir exterior: máximo solape con seed, si hay; si no, por área
        best_idx = -1
        best_score = -1.0
        for i, c in enumerate(cnts):
            if hier[i][3] != -1:  # parent != -1 => no es exterior
                continue
            area = cv2.contourArea(c)
            if area <= 0:
                continue
            score = area
            if prefer_seed is not None is not None:
                tmp = np.zeros_like(mask_bin, dtype=np.uint8)
                cv2.drawContours(tmp, [c], -1, 255, -1)
                inter = cv2.countNonZero(cv2.bitwise_and(tmp, prefer_seed))
                score = inter + 1e-3 * area  # prior: solape manda, el área desempata
            if score > best_score:
                best_score, best_idx = score, i

        if best_idx < 0:
            return np.zeros_like(mask_bin, dtype=np.uint8)

        # dibujar exterior
        out = np.zeros_like(mask_bin, dtype=np.uint8)
        cv2.drawContours(out, [cnts[best_idx]], -1, 255, -1)

        # restar agujeros (hijos del exterior elegido)
        child = hier[best_idx][2]
        while child != -1:
            cv2.drawContours(out, [cnts[child]], -1, 0, -1)
            child = hier[child][0]  # siguiente hermano

        return out
