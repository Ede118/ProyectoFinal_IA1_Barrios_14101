import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple, Union, List

import cv2
import numpy as np

from Code.types import MaskU8, VecF, GrayImageF32
from Code.vision import ImgOrchestrator, ImgFeat, ImgPreproc, KmeansModel,fit_from_paths, identify_path
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

#$end

repo: Repo = field(default_factory=Repo)
pre: ImgPreproc = field(default_factory=ImgPreproc)

paths: List[Path] = []

paths = repo.list_images("1", ["Arandela", "Tornillo", "Clavo", "Tuerca"])
Images = []
ImagesGray = []
ImagesMask = []

for p in paths:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    Images.append(img)
    ImgGray, ImgMask = pre.process(img)
    ImagesGray.append(ImgGray)
    ImagesMask.append(ImgMask)

show_img(Images[1])