import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "Database" / "input" / "image"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "Database" / "input" / "audio"

DEFAULT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

UI_DIR = Path(__file__).resolve().parent
UI_FILE = UI_DIR / "InterfazGrafica.ui"

RECORD_SECONDS = 2.0

from PyQt6 import QtWidgets, QtCore, uic
from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

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

class RecordCommandTask(QtCore.QRunnable):
	"""
	Tarea que llama al backend para grabar un comando de voz.
	Se ejecuta en un thread del QThreadPool para no bloquear la UI.
	"""
	def __init__(self, controller: AppController, duration_sec: float, output_path: Path) -> None:
		super().__init__()
		self._controller = controller
		self._duration_sec = duration_sec
		self._output_path = output_path

	def run(self) -> None:
		# Esta función se ejecuta en background
		self._controller.grabar_audio(duracion_segundos=self._duration_sec, ruta_salida=self._output_path)


class MainWindow(QtWidgets.QMainWindow):
	def __init__(self, controller: AppController | None = None) -> None:
		super().__init__()

		# Backend (ImgOrchestrator + AudioOrchestrator + BayesAgent)
		self._controller: AppController = controller or AppController()

		# --- AUDIO PLAYER PARA REPRODUCIR EL COMANDO ---
		self._audio_output = QAudioOutput(self)
		self._player = QMediaPlayer(self)
		self._player.setAudioOutput(self._audio_output)

		# Estados iniciales por defecto
		self._current_image_dir: Path = DEFAULT_IMAGE_DIR
		self._current_folder_dir: Path = DEFAULT_IMAGE_DIR
		self._last_image_path: Path | None = None
		self._last_folder_path: Path | None = None
		self._last_image_result: dict[str, object] | None = None
		self._last_folder_result: object | None = None

		# --- Inicialización en etapas ---
		self._load_ui()
		self._setup_plots()      # se llenará en el Paso 5
		self._connect_signals()  # ahora sí conectamos imagen/carpeta
		self._init_state()

		# Estado para la grabación de audio
		self._is_recording: bool = False
		self._record_duration_ms: int = int(RECORD_SECONDS * 1000)
		self._record_elapsed_ms: int = 0
		self._current_record_path: Path | None = None
		self._record_timer = QtCore.QTimer(self)
		self._record_timer.setInterval(50)  # 20 FPS aprox.
		self._record_timer.timeout.connect(self._on_record_tick)
		self._last_command_label: str | None = None
		self._classify_retries: int = 0

	# =========================================================
	# 1) Cargar UI
	# =========================================================
	def _load_ui(self) -> None:
		uic.loadUi(str(UI_FILE), self)

	# =========================================================
	# 2) Cargar plots de matplotlib
	# =========================================================
	def _setup_plots(self) -> None:
		# Más adelante: crear canvases de matplotlib y meterlos en
		# KMeansGraph, KNNGraph y BayesGraph.
		pass

	# =========================================================
	# 3) Conectar señales a handlers
	# =========================================================
	def _connect_signals(self) -> None:
		"""
		Conecta widgets interactivos de la ventana principal.

		Botones a los que se conecta:
		- BotonImagen -> _on_click_analizar_imagen
		- BotonCarpeta -> _on_click_analizar_carpeta
		- BotonGrabar -> _on_click_grabar
		- BotonConfirmarCmd -> _on_confirmar_comando
		- BotonReproducirCmd -> _on_reproducir_comando

		"""

		if hasattr(self, "BotonImagen"):
			self.BotonImagen.clicked.connect(self._on_click_analizar_imagen)
		if hasattr(self, "BotonCarpeta"):
			self.BotonCarpeta.clicked.connect(self._on_click_analizar_carpeta)
		if hasattr(self, "BotonGrabar"):
			self.BotonGrabar.clicked.connect(self._on_click_grabar)
		if hasattr(self, "BotonConfirmarCmd"):
			self.BotonConfirmarCmd.clicked.connect(self._on_confirmar_comando)
		if hasattr(self, "BotonReproducirCmd"):
			self.BotonReproducirCmd.clicked.connect(self._on_reproducir_comando)


	# =========================================================
	# 4) Estado inicial de la UI
	# =========================================================
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

		self._hide_command_controls()

	# =========================================================
	# 5) Handlers de botones
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

	def _on_click_grabar(self) -> None:
		"""
		Handler del botón 'Comenzar Grabación'.

		- Si ya se está grabando, ignora el click.
		- Si no, arranca una nueva grabación de RECORD_SECONDS segundos.
		"""
		if self._is_recording:
			return

		self._start_recording()
	
	def _on_confirmar_comando(self) -> None:
		"""El usuario acepta el comando reconocido."""
		if self._last_command_label is not None:
			# Más adelante acá disparás la acción real según el comando
			QtWidgets.QMessageBox.information(
				self,
				"Comando confirmado",
				f"Se confirmó el comando: {self._last_command_label}"
			)

		# Después de confirmar, ocultamos los controles
		self._hide_command_controls()

	def _on_reproducir_comando(self) -> None:
		"""
		Reproduce el último archivo de audio grabado (self._current_record_path).
		"""
		if not self._current_record_path:
			QtWidgets.QMessageBox.warning(
				self,
				"Audio no disponible",
				"No hay un comando grabado para reproducir."
			)
			return

		if not self._current_record_path.is_file():
			QtWidgets.QMessageBox.warning(
				self,
				"Archivo no encontrado",
				f"No se encontró el archivo de audio:\n{self._current_record_path}"
			)
			return

		# Reiniciar el player con este archivo
		self._player.stop()
		self._player.setSource(QUrl.fromLocalFile(str(self._current_record_path)))
		self._player.play()

	# =========================================================
	# 6) Lógica de grabación de audio
	# =========================================================

	def _next_command_path(self) -> Path:
		"""
		Devuelve la ruta para el próximo archivo Comando##.wav
		en DEFAULT_AUDIO_DIR.
		"""

		max_idx = 0
		for p in DEFAULT_AUDIO_DIR.glob("Comando*.wav"):
			stem = p.stem  # 'Comando03'
			if not stem.startswith("Comando"):
				continue
			num_str = stem.replace("Comando", "")
			try:
				num = int(num_str)
			except ValueError:
				continue
			if num > max_idx:
				max_idx = num

		next_idx = max_idx + 1
		return DEFAULT_AUDIO_DIR / f"Comando{next_idx:02d}.wav"

	def _start_recording(self) -> None:
		"""Configura estado y lanza la grabación en background."""
		output_path = self._next_command_path()
		self._current_record_path = output_path
		self._classify_retries = 0

		self._is_recording = True
		self._record_elapsed_ms = 0

		# UI: desactivar botón de grabar mientras tanto
		if hasattr(self, "BotonGrabar"):
			self.BotonGrabar.setEnabled(False)
			self.BotonGrabar.setText("Grabando...")

		# Barra al 0%
		if hasattr(self, "BarraCargando"):
			self.BarraCargando.setValue(0)

		# Arrancar timer que actualiza la barra
		self._record_timer.start()

		# Lanzar tarea de grabación en otro hilo
		task = RecordCommandTask(self._controller, RECORD_SECONDS, output_path)
		QtCore.QThreadPool.globalInstance().start(task)

	def _on_record_tick(self) -> None:
		"""Actualiza la barra de progreso durante la grabación."""
		self._record_elapsed_ms += self._record_timer.interval()
		frac = min(1.0, self._record_elapsed_ms / self._record_duration_ms)

		if hasattr(self, "BarraCargando"):
			self.BarraCargando.setValue(int(frac * 100))

		if self._record_elapsed_ms >= self._record_duration_ms:
			self._record_timer.stop()
			self._finish_recording()

	def _finish_recording(self) -> None:
		"""Se llama cuando termina el tiempo de grabación."""
		self._is_recording = False

		if hasattr(self, "BotonGrabar"):
			self.BotonGrabar.setEnabled(True)
			self.BotonGrabar.setText("Comenzar Grabación")

		if hasattr(self, "BarraCargando"):
			self.BarraCargando.setValue(100)

		if self._current_record_path is not None:
			self._clasificar_ultimo_comando()

	def _clasificar_ultimo_comando(self) -> None:
		"""Usa el backend para reconocer el comando y muestra la UI de confirmación."""
		if not self._current_record_path:
			return

		try:
			result = self._controller.predecir_audio(self._current_record_path)

			label = None
			for key in ("label", "comando", "command", "pred"):
				if isinstance(result, dict) and key in result:
					label = str(result[key])
					break
			if label is None:
				label = str(result)

			self._last_command_label = label

			if hasattr(self, "ComandoLabel"):
				self.ComandoLabel.setText(f"Comando identificado: {label}. ¿Confirmar?")

			# Ahora sí mostramos todo el bloque de comando
			self._show_command_controls()

		except Exception as exc:
			QtWidgets.QMessageBox.critical(
				self,
				"Error en reconocimiento de voz",
				f"No se pudo reconocer el comando:\n{exc}"
			)

	def _hide_command_controls(self) -> None:
		"""Oculta el label y los botones del comando de voz."""
		if hasattr(self, "ComandoLabel"):
			self.ComandoLabel.setVisible(False)
		if hasattr(self, "BotonConfirmarCmd"):
			self.BotonConfirmarCmd.setVisible(False)
		if hasattr(self, "BotonReproducirCmd"):
			self.BotonReproducirCmd.setVisible(False)

	def _show_command_controls(self) -> None:
		"""Muestra el label y los botones del comando de voz."""
		if hasattr(self, "ComandoLabel"):
			self.ComandoLabel.setVisible(True)
		if hasattr(self, "BotonConfirmarCmd"):
			self.BotonConfirmarCmd.setVisible(True)
		if hasattr(self, "BotonReproducirCmd"):
			self.BotonReproducirCmd.setVisible(True)


__all__ = ["MainWindow"]