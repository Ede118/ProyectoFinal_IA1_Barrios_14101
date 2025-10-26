"""Aliases tipográficos compartidos por los módulos de visión."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

ArrayLike: TypeAlias = npt.ArrayLike

VecF: TypeAlias = npt.NDArray[np.float64]
VecI: TypeAlias = npt.NDArray[np.int64]
MatF: TypeAlias = npt.NDArray[np.float64]

DTYPE: TypeAlias = np.dtype
F32 = np.float32

ColorImageU8: TypeAlias = npt.NDArray[np.uint8]
GrayImageF32: TypeAlias = npt.NDArray[np.float32]
MaskU8: TypeAlias = npt.NDArray[np.uint8]
