from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Code.image import ImgOrchestrator
from Code.audio import AudioOrchestrator
from Code.Estadisticas import BayesAgent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Database" / "data"
MODELS_DIR = PROJECT_ROOT / "Database" / "models"


@dataclass(frozen=False)
class AppController:
    IOrch: ImgOrchestrator = field(default_factory=ImgOrchestrator)
    AOrch: AudioOrchestrator = field(default_factory=AudioOrchestrator)
    Bayes: BayesAgent = field(default_factory=BayesAgent)
    
    def predecir_img(self, ruta_imagen: Path) -> dict:
        """
        Analiza una imagen utilizando el orquestador de imágenes.
        Devuelve un diccionario con los resultados del análisis.
        """
        resultados = self.IOrch.predecir(ruta_imagen)
        return resultados

    def predecir_carpeta_img(self, ruta_carpeta: Path) -> pd.DataFrame:
        """
        Analiza todas las imágenes en una carpeta utilizando el orquestador de imágenes.
        Devuelve un DataFrame con los resultados del análisis.
        """
        resultados_df = self.IOrch.analizar_carpeta(ruta_carpeta)
        return resultados_df

    def comenzar_grabacion_audio(self, duracion_segundos: float, ruta_salida: Path) -> None:
        """
        Comienza una grabación de audio por la duración especificada y guarda el archivo en la ruta dada.
        """
        self.AOrch.grabar_audio(duracion_segundos, ruta_salida)
        return None
    
    def analizar_audio(self, ruta_audio: Path) -> dict:
        """
        Analiza un archivo de audio utilizando el orquestador de audio.
        Devuelve un diccionario con los resultados del análisis.
        """
        resultados = self.AOrch.analizar_audio(ruta_audio)
        return resultados

    
