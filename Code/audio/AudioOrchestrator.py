from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa

from Code.audio.AudioFeat import AudioFeat, AudioFeatConfig
from Code.audio.AudioPreproc import AudioPreproc, AudioPreprocConfig
from Code.audio.Standardizer import Standardizer
from Code.audio.KnnModel import KnnModel, KnnConfig

AudioPath = str | Path

from Code.AliasesUsed import PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "Database" / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "knn.npz"

@dataclass(slots=True)
class AudioOrchestrator:
	"""
	Coordina preprocesamiento, extracción de features, PCA opcional y k-NN.
	- `entrenar` arma la base y deja el KNN listo (y puede guardar el modelo).
	- `cargar_modelo` restaura lo guardado.
	- `predecir_comando` aplica el pipeline de inferencia.
	"""

	AProc: AudioPreproc = field(
		default_factory=lambda: AudioPreproc(
			config=AudioPreprocConfig(
				target_sr=16000,
				T_sec=1.2,
				frame_ms=25.0,
				hop_ms=10.0,
				top_dB=35.0,
				corte_pasaalto=80.0,
				orden_pasaalto=4,
				coeficiente_pre_enfasis=0.97,
				norm_mode="RMS",
				rms_target_dbfs=-20.0,
				peak_ref=0.98,
				max_gain_db=18.0,
				gate_dbfs=-60.0,
				pad_mode="edge",
			)
		)
	)

	feat: AudioFeat = field(
		default_factory=lambda: AudioFeat(
			config=AudioFeatConfig(
				sr_target=16_000,
				win_ms=25.0,
				hop_ms=10.0,
				N_MFCC=20,
				delta_order=1,
				RMS=True,
				ZCR=True,
				stats=("mean", "std", "p10", "p90"),
			)
		)
	)

	stats: Standardizer = field(default_factory=Standardizer)

	knn: KnnModel = field(
		default_factory=lambda: KnnModel(
			config=KnnConfig(
				k_vecinos=5,
				tipo_distancia="cosine",
				weighted=True,
				reject_max_dist=None,
				ratio_max=None,
			)
		)
	)
	_X_store: np.ndarray | None = field(default=None, init=False, repr=False)       # (N, D) std
	_y_store: np.ndarray | None = field(default=None, init=False, repr=False)       # (N,)
	_X_store_raw: np.ndarray | None = field(default=None, init=False, repr=False)   # (N, D_raw)
	_X_store_proj: np.ndarray | None = field(default=None, init=False, repr=False)  # (N, k)
	_feature_names: list[str] | None = field(default=None, init=False, repr=False)
	_eigvecs: np.ndarray | None = field(default=None, init=False, repr=False)       # (D, D)
	_eigvals: np.ndarray | None = field(default=None, init=False, repr=False)       # (D,)
	_k_used: int | None = field(default=None, init=False, repr=False)

	# -------------------------------------------------------------------------------------------------  #
	#                                      --------- Entrenamiento ---------                             #
	# -------------------------------------------------------------------------------------------------  #

	def entrenar(
		self,
		paths: Sequence[AudioPath],
		labels: Sequence[str],
		*,
		var_objetivo: float = 0.95,
		k_componentes: int | None = None,
		guardar_en: Path | str | None = None,
	) -> dict[str, int | float]:
		"""
		### Entrenar orquestador
		Preprocesa audios, extrae features, ajusta Standardizer, aplica PCA y carga KNN.
		- `var_objetivo`: fracción de varianza a retener si no fijas `k_componentes`
		- `k_componentes`: fija explícitamente cuántas PCs usar (ignora `var_objetivo`)
		- `guardar_en`: ruta opcional para persistir el modelo al final
		"""
		path_list = list(paths)
		label_list = [str(l) for l in labels]
		if not path_list:
			raise ValueError("paths no puede estar vacío.")
		if len(path_list) != len(label_list):
			raise ValueError("paths y labels deben tener la misma longitud.")

		# 1) Preproceso + features
		vectores_raw: list[np.ndarray] = []
		for p in path_list:
			y_proc, sr = self.AProc.procesar(p)
			vec = self.feat.extraer_caracteristicas(y_proc, sr)
			vectores_raw.append(vec.astype(np.float64, copy=False))

		X_raw = np.stack(vectores_raw, axis=0).astype(np.float64, copy=False)  # (N, D_raw)
		self._feature_names = self.feat.nombres_features()

		# 2) Standardizer (Z-score)
		self.stats.calculate_statistics(X_raw)
		X_std = self.stats.transform(X_raw).astype(np.float32, copy=False)

		# 3) PCA sobre X_std
		eigvals, eigvecs = self._pca(X_std)
		if k_componentes is None:
			k = int(np.searchsorted(np.cumsum(eigvals) / eigvals.sum(), float(var_objetivo)) + 1)
		else:
			k = max(1, min(int(k_componentes), eigvecs.shape[1]))
		
		X_proj = X_std @ eigvecs[:, :k]

		# 4) KNN con proyecciones
		self.knn.cargar_lote(X_proj.astype(np.float32, copy=False), label_list)

		# 5) Persistir estado en memoria
		self._X_store_raw = X_raw
		self._X_store = X_std
		self._X_store_proj = X_proj
		self._y_store = np.asarray(label_list, dtype=np.str_)
		self._eigvecs = eigvecs
		self._eigvals = eigvals
		self._k_used = k

		if guardar_en is not None:
			self.guardar_modelo(guardar_en)

		return {
			"N": X_raw.shape[0],
			"D_raw": X_raw.shape[1],
			"D_proj": k,
			"var_retenida": float(eigvals[:k].sum() / eigvals.sum()),
		}

	# -------------------------------------------------------------------------------------------------  #
	#                                         --------- Modelo ---------                                 #
	# -------------------------------------------------------------------------------------------------  #

	def guardar_modelo(self, path: Path | str | None = None) -> None:
		"""Guardar modelo entrenado en npz (stats, PCA, base proyectada, config KNN)."""
		self._ensure_listo()
		npz_path = Path(path) if path is not None else DEFAULT_MODEL_PATH
		if not npz_path.is_absolute():
			npz_path = DEFAULT_MODEL_PATH.parent / npz_path
		npz_path.parent.mkdir(parents=True, exist_ok=True)
		np.savez_compressed(
			npz_path,
			mu=self.stats.mu,
			sigma=self.stats.sigma,
			eigvals=self._eigvals,
			eigvecs=self._eigvecs,
			k_used=self._k_used,
			X_proj=self._X_store_proj,
			labels=np.array(self._y_store, dtype=np.str_),
			feature_names=np.array(self._feature_names, dtype=object) if self._feature_names else None,
			knn_k=self.knn.config.k_vecinos,
			knn_metric=str(self.knn.config.tipo_distancia),
			knn_weighted=self.knn.config.weighted,
		)

	def cargar_modelo(self, path: Path | str | None = None) -> None:
		"""Cargar modelo desde npz y reconstruir KNN."""
		npz_path = Path(path) if path is not None else DEFAULT_MODEL_PATH
		if not npz_path.is_absolute():
			npz_path = DEFAULT_MODEL_PATH.parent / npz_path
		data = np.load(npz_path, allow_pickle=True)

		mu = data["mu"].astype(np.float32, copy=False)
		sigma = data["sigma"].astype(np.float32, copy=False)
		eigvals = data["eigvals"].astype(np.float64, copy=False)
		eigvecs = data["eigvecs"].astype(np.float64, copy=False)
		k_used = int(data["k_used"])
		X_proj = data["X_proj"].astype(np.float32, copy=False)
		labels = data["labels"].astype(str)
		feature_names = data["feature_names"]
		knn_k = int(data["knn_k"])
		knn_metric_raw = data["knn_metric"]
		knn_metric = str(knn_metric_raw.item() if hasattr(knn_metric_raw, "item") else knn_metric_raw)
		if knn_metric not in ("cosine", "euclidean"):
			raise ValueError(f"Métrica KNN inválida cargada: {knn_metric}")
		knn_weighted = bool(data["knn_weighted"])

		self.stats.mu = mu
		self.stats.sigma = sigma
		self._eigvals = eigvals
		self._eigvecs = eigvecs
		self._k_used = k_used
		self._feature_names = list(feature_names.tolist())
		self._X_store_proj = X_proj
		self._X_store = None
		self._X_store_raw = None
		self._y_store = labels

		self.knn = KnnModel(config=KnnConfig(k_vecinos=knn_k, tipo_distancia=knn_metric, weighted=knn_weighted))
		self.knn.cargar_lote(X_proj, labels.tolist())


	# -------------------------------------------------------------------------------------------------  #
	#                                        --------- Predicción ---------                              #
	# -------------------------------------------------------------------------------------------------  #

	def predecir_comando(
		self,
		entrada: AudioPath,
		*,
		devolver_distancia: bool = True,
	) -> dict[str, str | float]:
		"""
		### Inferencia
		Preprocesa la entrada, extrae features, z-score, proyecta y predice con KNN.
		"""
		self._ensure_listo()
		y_proc, sr = self._preprocesar_audio(entrada)
		vec_raw = self.feat.extraer_caracteristicas(y_proc, sr)
		vec_std = self.stats.transform_one(vec_raw)
		k = int(self._k_used or vec_std.shape[0])
		vec_proj = vec_std @ self._eigvecs[:, :k]

		label = self.knn.predecir(vec_proj.astype(np.float32, copy=False))
		salida: dict[str, str | float] = {"label": label}
		
		if devolver_distancia:
			distancias = self.knn.distancias(vec_proj.astype(np.float32, copy=False))
			salida["distancia_min"] = float(np.min(distancias))

		return salida
	
	def grabar_audio(
		self,
		*,
		dur_sec: float = 2.0,
		salida: str | Path | None = None,
	) -> Path:
		"""Graba ~dur_sec segundos (mono), resamplea al target y guarda en WAV."""
		info = sd.query_devices(kind="input")
		sr_hw = int(info.get("default_samplerate", 0) or 0)
		if sr_hw <= 0:
			raise RuntimeError("No se pudo obtener sample rate del dispositivo de entrada.")
		n_frames = max(1, int(sr_hw * float(dur_sec)))

		# 2) Grabar
		audio = sd.rec(frames=n_frames, samplerate=sr_hw, channels=1, dtype="float32")
		sd.wait()
		audio = np.asarray(audio, dtype=np.float32).squeeze()

		# 3) Resamplear a target si lo querés uniforme para el pipeline
		sr_target = int(self.AProc.config.target_sr)
		if sr_hw != sr_target:
			audio = librosa.resample(audio, orig_sr=sr_hw, target_sr=sr_target)
			sr_save = sr_target
		else:
			sr_save = sr_hw

		# 4) Guardar en Database/input/audio (o en la ruta solicitada)
		if salida is None:
			out_dir = Path(__file__).resolve().parents[2] / "Database" / "input" / "audio"
			out_dir.mkdir(parents=True, exist_ok=True)
			out_path = out_dir / "grabacion.wav"
		else:
			out_path = Path(salida)
			if out_path.suffix == "":
				out_path = out_path / "grabacion.wav"
			out_path.parent.mkdir(parents=True, exist_ok=True)
		audio = audio.astype(np.float32, copy=False)
		sf.write(out_path, audio, sr_save, subtype="PCM_16")
		return out_path

	# -------------------------------------------------------------------------------------------------  #
	#                                    --------- Helpers privados ---------                            #
	# -------------------------------------------------------------------------------------------------  #

	def _pca(self, X_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""
		PCA clásico vía autovalores de la matriz de covarianza (X_std ya centrado/escala).
		Devuelve (eigvals_desc, eigvecs_ordenados).
		"""
		cov = np.cov(X_std, rowvar=False)
		eigvals, eigvecs = np.linalg.eigh(cov)
		idx = np.argsort(eigvals)[::-1]
		return eigvals[idx], eigvecs[:, idx]

	def _preprocesar_audio(
		self,
		entrada: AudioPath,
	) -> tuple[np.ndarray, int]:
		"""Carga o procesa un path de audio y devuelve señal mono normalizada y sample rate."""
		if isinstance(entrada, (str, Path)):
			return self.AProc.procesar(entrada)
		raise TypeError("Entrada inválida. Usa un path a archivo de audio.")

	def _ensure_listo(self) -> None:
		"""Confirma que stats, PCA y KNN estén disponibles antes de inferir o guardar."""
		if (
			self.stats.mu is None
			or self.stats.sigma is None
			or self._eigvecs is None
			or self._k_used is None
			or self.knn is None
		):
			raise RuntimeError("Modelo no entrenado/cargado. Llamá a entrenar() o cargar_modelo().")


__all__ = ["AudioOrchestrator", "AudioPath"]
