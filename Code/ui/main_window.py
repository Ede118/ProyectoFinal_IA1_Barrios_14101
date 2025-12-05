import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "Database" / "input" / "image"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "Database" / "input" / "audio"

UI_DIR = Path(__file__).resolve().parent
UI_FILE = UI_DIR / "InterfazGrafica.ui"

from PyQt6 import QtWidgets, uic

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure

from Code.app.AppController import AppController

class MplCanvas(FigureCanvas):
	def __init__(self, parent=None):
		fig = Figure(constrained_layout=True)
		self.ax = fig.add_subplot(111)
		super().__init__(fig)
		self.setParent(parent)

class MainWindow(QtWidgets.QMainWindow):
	def __init__(self, controller: AppController | None = None) -> None:
		super().__init__()

		# Backend (ImgOrchestrator + AudioOrchestrator + BayesAgent)
		self._controller: AppController = controller or AppController()

		# Estado interno relacionado con paths
		self._current_image_dir: Path = DEFAULT_IMAGE_DIR
		self._current_folder_dir: Path = DEFAULT_IMAGE_DIR
		self._last_image_path: Path | None = None
		self._last_folder_path: Path | None = None
		self._last_image_result: dict[str, object] | None = None
		self._last_folder_result: object | None = None  # luego puede ser un DataFrame

		# --- Inicialización en etapas ---
		self._load_ui()
		self._setup_plots()      # se llenará en el Paso 5
		self._connect_signals()  # ahora sí conectamos imagen/carpeta
		self._init_state()

	# ---------------------------------------------------------
	# 1) Cargar UI
	# ---------------------------------------------------------
	def _load_ui(self) -> None:
		uic.loadUi(str(UI_FILE), self)

	# ---------------------------------------------------------
	# 2) Plots (stub, Paso 5)
	# ---------------------------------------------------------
	def _setup_plots(self) -> None:
		# Más adelante: crear canvases de matplotlib y meterlos en
		# KMeansGraph, KNNGraph y BayesGraph.
		pass

	# ---------------------------------------------------------
	# 3) Conectar señales (Paso 2: solo imagen/carpeta)
	# ---------------------------------------------------------
	def _connect_signals(self) -> None:
		"""
		Conecta widgets interactivos de la ventana principal.

		En este paso solo:
		- BotonImagen
		- BotonCarpeta
		(El botón de grabar queda para el Paso 3).
		"""
		if hasattr(self, "BotonImagen"):
			self.BotonImagen.clicked.connect(self._on_click_analizar_imagen)

		if hasattr(self, "BotonCarpeta"):
			self.BotonCarpeta.clicked.connect(self._on_click_analizar_carpeta)

		# BotonGrabar se conecta en el Paso 3
		# if hasattr(self, "BotonGrabar"):
		#     self.BotonGrabar.clicked.connect(self._on_click_grabar)

	# ---------------------------------------------------------
	# 4) Estado visual inicial
	# ---------------------------------------------------------
	def _init_state(self) -> None:
		if hasattr(self, "ImagenPath"):
			# Por defecto, ninguna imagen seleccionada
			self.ImagenPath.setText("Ninguna imagen seleccionada")

		if hasattr(self, "CarpetaPath"):
			# Carpeta por defecto: /Database/input/image
			self.CarpetaPath.setText(str(DEFAULT_IMAGE_DIR))

		if hasattr(self, "BarraCargando"):
			self.BarraCargando.setMinimum(0)
			self.BarraCargando.setMaximum(100)
			self.BarraCargando.setValue(0)

	# =========================================================
	# Handlers de botones (PASO 2)
	# =========================================================

	def _on_click_analizar_imagen(self) -> None:
		"""
		Handler del botón 'Analizar Imagen'.

		- Abre un diálogo de selección de archivo (usa el file dialog del sistema,
		  en KDE se integra con Dolphin).
		- Actualiza el label ImagenPath.
		- Llama al backend para predecir la imagen y guarda el resultado
		  (sin mostrarlo todavía; eso se usará al hacer los gráficos).
		"""
		start_dir = self._current_image_dir if self._current_image_dir.exists() else DEFAULT_IMAGE_DIR

		file_path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
			self,
			"Seleccionar imagen a analizar",
			str(start_dir),
			"Imágenes (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
		)

		if not file_path_str:
			return  # usuario canceló

		path = Path(file_path_str)
		self._current_image_dir = path.parent
		self._last_image_path = path

		if hasattr(self, "ImagenPath"):
			self.ImagenPath.setText(str(path))

		# Lógica de backend: predecir imagen
		try:
			self._last_image_result = self._controller.predecir_img(path)
			# Más adelante usaremos esto para actualizar los gráficos.
		except Exception as exc:
			QtWidgets.QMessageBox.critical(
				self,
				"Error al analizar imagen",
				f"No se pudo analizar la imagen:\n{exc}"
			)

	def _on_click_analizar_carpeta(self) -> None:
		"""
		Handler del botón 'Analizar Carpeta'.

		- Abre un diálogo de selección de carpeta.
		- Actualiza el label CarpetaPath.
		- Llama al backend para analizar la carpeta completa y guarda
		  el resultado (por ahora solo lo almacenamos).
		"""
		start_dir = self._current_folder_dir if self._current_folder_dir.exists() else DEFAULT_IMAGE_DIR

		dir_path_str = QtWidgets.QFileDialog.getExistingDirectory(
			self,
			"Seleccionar carpeta de imágenes",
			str(start_dir)
		)

		if not dir_path_str:
			return  # usuario canceló

		carpeta = Path(dir_path_str)
		self._current_folder_dir = carpeta
		self._last_folder_path = carpeta

		if hasattr(self, "CarpetaPath"):
			self.CarpetaPath.setText(str(carpeta))

		# Lógica de backend: analizar carpeta
		try:
			self._last_folder_result = self._controller.predecir_carpeta_img(carpeta)
			# Más adelante podremos usar este DataFrame para estadísticas / debug.
		except Exception as exc:
			QtWidgets.QMessageBox.critical(
				self,
				"Error al analizar carpeta",
				f"No se pudo analizar la carpeta:\n{exc}"
			)


__all__ = ["MainWindow"]