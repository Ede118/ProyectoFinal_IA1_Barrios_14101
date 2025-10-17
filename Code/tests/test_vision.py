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

"""
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

# --------------------------------------------------------------------------- #
# ImgPreproc Tests 
# --------------------------------------------------------------------------- #

def  test_normalize_preserves_aspect():
    pre = ImgPreproc()
    sample1 = np.full((100, 50), 1, dtype=F32)
    sample2 = np.full((200, 500), 1, dtype=F32)
    sample3 = np.full((150, 150), 1, dtype=F32)
    
    sample1 = pre.normalize(sample1)
    sample2 = pre.normalize(sample2)
    sample3 = pre.normalize(sample3)
    
    assert sample1.ndim == 2, "\nFallo dimensión 1"
    assert sample2.ndim == 2, "\nFallo dimensión 2"
    assert sample3.ndim == 2, "\nFallo dimensión 3"
    
    assert sample1.shape == (128, 128), f"\nFallo tamaño de matriz\nTiene tamaño {sample1.shape}"
    assert sample2.shape == (128, 128), f"\nFallo tamaño de matriz\nTiene tamaño {sample2.shape}"
    assert sample3.shape == (128, 128), f"\nFallo tamaño de matriz\nTiene tamaño {sample3.shape}"
    

def  test_segment_rejects_small_blob():
    cfg = ImgPreprocCfg(seg_min_area=0.2)   # exige que el objeto ocupe ≥20% del frame
    pre = ImgPreproc(cfg=cfg)

    img = np.zeros((128, 128, 3), np.uint8)
    cv2.circle(img, (64, 64), 5, (255, 255, 255), -1)  # blob muy chico

    _, mask = pre.process(img, float_mask=False)

    assert mask.shape == (128, 128)
    assert cv2.countNonZero(mask) == 0

def  test_normalize_zscore_range():
    cfg = ImgPreprocCfg(normalize_kind='zero_one')
    pre = ImgPreproc(cfg=cfg)
    repo = Repo(root=PROJECT_ROOT)
    
    paths = repo.list_images("1")
    
    for p, i in enumerate(paths, 1):
        img = cv2.imread(str(p))
        Gray = pre._normalize_values(img)
        assert Gray <= pytest.approx(1, rel=1e-3)
        


# --------------------------------------------------------------------------- #
# ImgFeat Tests 
# --------------------------------------------------------------------------- #

def test_identificar_agujeros():
    cfg = ImgPreprocCfg(normalize_kind='zero_one')
    pre = ImgPreproc(cfg=cfg)
    feat = ImgFeat()
    repo = Repo(root=PROJECT_ROOT)
    
    pathsA = repo.list_images("1", ["Arandela"])
    parameters = []
    
    for p in pathsA:
        img = cv2.imread(str(p))
        griz, mascara = pre.process(img)
        huecos = feat.contar_huecos(mask=mascara)
        parameters.append(huecos)
        assert huecos == pytest.approx(1, rel=1e-3)
        

def test_contar_vertices_convex_vs_nonconvex():
    
    pass

def test_rugosidad_estrella_iterativo():
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
    

def test_rugosidad_disc_vs_estrella():
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

def test_rugosidad():
    cfg = ImgPreprocCfg(normalize_kind='zero_one')
    pre = ImgPreproc(cfg=cfg)
    feat = ImgFeat()
    repo = Repo(root=PROJECT_ROOT)
    
    pathsT = repo.list_images("1", ["Tornillo"])
    pathsC = repo.list_images("1", ["Clavo"])
    
    rugosidadTornillos = []
    rugosidadClavos = []
    
    for p in pathsT:
        img = cv2.imread(str(p))
        griz, mascara = pre.process(img)
        rugosidad = feat.rugosidad(mask=mascara)
        rugosidadTornillos.append(rugosidad)
        
    for p in pathsC:
        img = cv2.imread(str(p))
        griz, mascara = pre.process(img)
        rugosidad = feat.rugosidad(mask=mascara)
        rugosidadTornillos.append(rugosidad)
        
    assert rugosidadClavos == pytest.approx(0, rel=0.1)
    assert rugosidadClavos >= pytest.approx(0.5, rel=0.1)

def test_gradiente_interno_sensitive():
    pass

# --------------------------------------------------------------------------- #
# KMeansModel Tests 
# --------------------------------------------------------------------------- #

def test_fit_reproducible_with_seed():
    pass

def test_predict_object():
    pass

def test_fit_handles_empty_cluster():
    pass

# --------------------------------------------------------------------------- #
# Orchestrator Tests 
# --------------------------------------------------------------------------- #


