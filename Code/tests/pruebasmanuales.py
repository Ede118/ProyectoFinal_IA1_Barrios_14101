import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pytest

from Code.types import MaskU8, VecF, GrayImageF32, F32
from Code.image import ImgOrchestrator, ImgFeat, ImgPreproc,fit_from_paths, identify_path
from Code.image.ImgPreproc import ImgPreprocCfg
from Code.image.KmeansModel import KMeans
from Code.adapters import Repo

from Code.tests.utilities import (
    normalize_to_uint8,
    show_img,
    imshow_grid,
    show_hist,
    plot_series,
    describe_array,
    cv2_backend,
    can_use_highgui,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_disc(radius: int = 30) -> np.ndarray:
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(img, (64, 64), radius, (255, 255, 255), -1)
    return img

# --------------------------------------------------------------------------- #
# Inicialización de los objetos 
# --------------------------------------------------------------------------- #


pre = ImgPreproc()
feat = ImgFeat()
model = KMeans(k=1, random_state=0)
orch = ImgOrchestrator(
    pre=pre,
    feat=feat,
    model=model,
    class_names=["Tornillo"],
    class_colors={"Tornillo": (0, 255, 0)},
)

"""
def rugosidad_estrella_iterativo():
    feat = ImgFeat()
    star_base = np.zeros((128, 128), np.uint8)
    pts = np.array(
        [[64, 20], [74, 50], [116, 50], [82, 72], [94, 110],
         [64, 90], [34, 110], [46, 72], [12, 50], [54, 50]],
        np.int32,
    ).reshape((-1, 1, 2))
    
    # for scale, blur in [(1.0, 0), (0.9, 0), (1.0, 3), (1.1, 5)]:
    for s in range(21):
        scale, blur = 0.1*s, 0
        scale = np.round(scale, 2)
        canvas = np.zeros_like(star_base)
        pts_scaled = (pts.astype(np.float32) * scale + 0.5).astype(np.int32)
        cv2.fillPoly(canvas, [pts_scaled], 255)
        if blur:
            canvas = cv2.GaussianBlur(canvas, (blur, blur), 0)
            canvas = (canvas > 127).astype(np.uint8) * 255
        r = feat.rugosidad(canvas) * 10.0
        print(f"\nscale={scale}, blur={blur} -> rugosidad={r:.3f}")
        assert r >= 0
        
    print("\n------------------------------------------------------------")
    
    for radius in range(10, 100, 3):
        disc = np.zeros((128, 128), np.uint8)
        cv2.circle(disc, (64, 64), radius, 255, -1)
        rugosidad = feat.rugosidad(disc) * 10.0
        print(f"\nradio = {radius} -> rugosidad = {rugosidad}")
        
    for radius in range(10, 100, 3):
        disc = np.zeros((256, 256), np.uint8)
        cv2.circle(disc, (64, 64), radius, 255, -1)
        rugosidad = feat.rugosidad(disc) * 10.0
        print(f"\nradio = {radius} -> rugosidad = {rugosidad}")
        
    for radius in range(10, 250, 3):
        disc = np.zeros((512, 512), np.uint8)
        cv2.circle(disc, (64, 64), radius, 255, -1)
        rugosidad = feat.rugosidad(disc) * 10.0
        print(f"\nradio = {radius} -> rugosidad = {rugosidad}")
    

def rugosidad_disc_vs_estrella():
    feat = ImgFeat()
    
    for radius in range(10, 65, 5):
        disc = np.zeros((128, 128), np.uint8)
        cv2.circle(disc, (64, 64), radius, 255, -1)
        print("\n", radius, feat.rugosidad(disc))


    # Máscara convexa (círculo): rugosidad ≈ 0
    disc = np.zeros((128, 128), np.uint8)
    cv2.circle(disc, (64, 64), 60, 255, -1)
    r_disc = feat.rugosidad(disc)
    print(f"Rugosidad disco: {r_disc:.4f}")  # usar pytest -s para verlo
    assert r_disc == pytest.approx(0.0, abs=1e-2)

    # Máscara dentada (estrella): rugosidad > 0
    star = np.zeros((128, 128), np.uint8)
    pts = np.array(
        [[64, 20], [74, 50], [116, 50], [82, 72], [94, 110],
         [64, 86], [34, 110], [46, 72], [12, 50], [54, 50]],
        np.int32,
    ).reshape((-1, 1, 2))
    cv2.fillPoly(star, [pts], 255)
    r_star = feat.rugosidad(star)
    print(f"Rugosidad estrella: {r_star:.4f}")
    assert r_star > 0.05  # ajustá el umbral si hace falta
"""
radii = range(10, 100, 3)
res128 = []
res256 = []
res512 = []

for radius in radii:
    disc128 = np.zeros((128, 128), np.uint8)
    cv2.circle(disc128, (64, 64), radius, 255, -1)
    res128.append(feat.rugosidad(disc128))

    disc256 = np.zeros((256, 256), np.uint8)
    cv2.circle(disc256, (128, 128), radius * 2, 255, -1)  # escala
    res256.append(feat.rugosidad(disc256))

    disc512 = np.zeros((512, 512), np.uint8)
    cv2.circle(disc512, (256, 256), radius * 4, 255, -1)
    res512.append(feat.rugosidad(disc512))

dif_256_128 = np.array(res256) - np.array(res128)
dif_512_256 = np.array(res512) - np.array(res256)
dif_512_128 = np.array(res512) - np.array(res128)

dif_256_128 = np.round(dif_256_128, 3)
dif_512_256 = np.round(dif_512_256, 3)
dif_512_128 = np.round(dif_512_128, 3)


print(dif_256_128, "\n\n")
print(dif_512_128, "\n\n")
print(dif_512_256, "\n\n")
