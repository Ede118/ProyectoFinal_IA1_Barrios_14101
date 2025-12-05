"""
Convenience exports so `import Code` brinda acceso rápido a tipos y orquestadores.
"""

# Tipos y constantes
from .AliasesUsed import (
    PROJECT_ROOT,
    DTYPE,
    ScalarF,
    VecF,
    MatF,
    ImgGray,
    ImgColor,
    Mask,
    ImgGrayF,
    ImgColorF,
    AudioSignal,
    Spectrogram,
    FeatVec,
    FeatMat,
    LabelArray,
    ProbVec,
    ProbMat,
    LogProbVec,
    LogProbMat,
    ClassIdx,
)

# Orquestadores y controlador
from .image import ImgOrchestrator
from .audio import AudioOrchestrator
from .Estadisticas import BayesAgent
from .app.AppController import AppController

# UI principal (Qt)
from .ui.main_window import MainWindow

__all__ = [
    # tipos
    "PROJECT_ROOT",
    "DTYPE",
    "ScalarF",
    "VecF",
    "MatF",
    "ImgGray",
    "ImgColor",
    "Mask",
    "ImgGrayF",
    "ImgColorF",
    "AudioSignal",
    "Spectrogram",
    "FeatVec",
    "FeatMat",
    "LabelArray",
    "ProbVec",
    "ProbMat",
    "LogProbVec",
    "LogProbMat",
    "ClassIdx",
    # lógica principal
    "ImgOrchestrator",
    "AudioOrchestrator",
    "BayesAgent",
    "AppController",
    # UI
    "MainWindow",
]
