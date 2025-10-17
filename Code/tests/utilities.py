""" 
viz_utils.py

Utilidades de visualización para imágenes y datos numéricos.

"""

from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Mapping
import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]

    # -------------------------------------------------------------------------------------------------  #
    #                           ---------- Modificar Img ----------                                      #
    # -------------------------------------------------------------------------------------------------  #

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Normaliza cualquier imagen/array numérico al rango [0, 255] en uint8.
    Útil para visualizar resultados intermedios (filtros, mapas de calor, etc.).
    """
    a = np.asarray(img)
    if a.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    a = a.astype(np.float32)
    mn, mx = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.zeros(a.shape[:2], dtype=np.uint8) if a.ndim >= 2 else np.zeros_like(a, dtype=np.uint8)
    s = (a - mn) / (mx - mn)
    s = (s * 255.0).clip(0, 255)
    if a.ndim == 3 and a.shape[2] in (3, 4):
        return s.astype(np.uint8)
    return s.astype(np.uint8)

def _to_rgb_for_mpl(img: np.ndarray) -> np.ndarray:
    """
    Convierte BGR->RGB para matplotlib y asegura uint8 si tiene rango extraño.
    No modifica imágenes ya RGB/GRAY razonables.
    """
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] == 3:
        # Heurística: si parece BGR (OpenCV) lo invertimos a RGB.
        # No podemos detectarlo 100% sin contexto, pero normalmente llega en BGR desde cv2.
        # Para no arruinar datos, solo cambiamos el orden de canales si el dtype es uint8.
        if a.dtype == np.uint8:
            return a[:, :, ::-1]
        return normalize_to_uint8(a)[:, :, ::-1]
    if a.ndim == 2:
        return normalize_to_uint8(a)
    if a.ndim == 3 and a.shape[2] == 4:  # BGRA -> RGBA
        b = normalize_to_uint8(a)
        return np.dstack([b[:, :, 2], b[:, :, 1], b[:, :, 0], b[:, :, 3]])
    return normalize_to_uint8(a)


    # -------------------------------------------------------------------------------------------------  #
    #                               ---------- CV2 ----------                                            #
    # -------------------------------------------------------------------------------------------------  #
    

def cv2_backend() -> str:
    """Devuelve el UI backend de OpenCV (p.ej. 'QT', 'GTK3') o cadena vacía si no hay."""
    try:
        import cv2  # type: ignore
        b = cv2.currentUIFramework()
        return b or ""
    except Exception:
        return ""

def can_use_highgui() -> bool:
    """
    Test suave de si podemos abrir ventanas de cv2.
    Crea y destruye una ventana dummy. Si falla, no hay backend usable.
    """
    try:
        import cv2  # type: ignore
        name = "__viz_tmp__"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(name)
        return True
    except Exception:
        return False

    # -------------------------------------------------------------------------------------------------  #
    #                           ---------- Visualización ----------                                      #
    # -------------------------------------------------------------------------------------------------  #
    
def show_img(
    img: np.ndarray,
    title: str = "AREPL",
    backend: str = "auto",
    block: bool = True,
    size: Optional[Tuple[int, int]] = None,   # (w, h) para ventana cv2
    move: Optional[Tuple[int, int]] = None,   # (x, y) para ventana cv2
) -> None:
    """
    Muestra una imagen.
    - backend='auto' intenta cv2 y cae en matplotlib si no hay GUI.
    - En AREPL: poné las llamadas a show_img DEBAJO de `#$end` para evitar
      que se reabra la ventana en cada tecla.
    """
    if backend not in ("auto", "cv2", "mpl"):
        raise ValueError("backend debe ser 'auto', 'cv2' o 'mpl'")

    use_cv2 = (backend in ("auto", "cv2")) and can_use_highgui()
    if use_cv2:
        import cv2  # type: ignore
        img_show = normalize_to_uint8(img)
        if img_show.ndim == 3 and img_show.shape[2] == 3:
            # Matplotlib quiere RGB, OpenCV muestra BGR. Para cv2 no tocamos.
            pass
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        if size:
            cv2.resizeWindow(title, *size)
        if move:
            cv2.moveWindow(title, *move)
        cv2.imshow(title, img_show)
        cv2.waitKey(0 if block else 1)
        if block:
            cv2.destroyWindow(title)
        return

    # Fallback a Matplotlib
    import matplotlib.pyplot as plt  # type: ignore
    arr = _to_rgb_for_mpl(img)
    plt.figure(title)
    if arr.ndim == 2:
        plt.imshow(arr, cmap="gray")
    else:
        plt.imshow(arr)
    plt.axis("off")
    plt.tight_layout()
    plt.show(block=block)

def imshow_grid(
    images: Sequence[np.ndarray],
    titles: Optional[Sequence[str]] = None,
    cols: int = 3,
    figsize: Tuple[int, int] = (10, 6),
    block: bool = True,
) -> None:
    """
    Muestra una grilla de imágenes con Matplotlib. Ideal para comparar salidas en tests.
    """
    import matplotlib.pyplot as plt  # type: ignore

    if len(images) == 0:
        return
    n = len(images)
    rows = int(np.ceil(n / cols))
    plt.figure("grid", figsize=figsize)
    for i, img in enumerate(images, 1):
        plt.subplot(rows, cols, i)
        arr = _to_rgb_for_mpl(img)
        if arr.ndim == 2:
            plt.imshow(arr, cmap="gray")
        else:
            plt.imshow(arr)
        if titles and i - 1 < len(titles):
            plt.title(titles[i - 1])
        plt.axis("off")
    plt.tight_layout()
    plt.show(block=block)

def show_hist(img: np.ndarray, bins: int = 256, title: str = "histograma", block: bool = True) -> None:
    """
    Muestra el histograma de una imagen. Si es color, dibuja por canal.
    """
    import matplotlib.pyplot as plt  # type: ignore

    a = np.asarray(img)
    plt.figure(title)
    if a.ndim == 2:
        plt.hist(a.ravel(), bins=bins)
        plt.title(f"{title} (gris)")
    elif a.ndim == 3 and a.shape[2] == 3:
        # asumimos BGR típico de cv2; para leer con matplotlib convertimos visualmente, pero el hist es por canal bruto
        colors = ["b", "g", "r"]
        for i, c in enumerate(colors):
            plt.hist(a[:, :, i].ravel(), bins=bins, alpha=0.5, label=c)
        plt.legend()
        plt.title(f"{title} (B,G,R)")
    else:
        plt.hist(normalize_to_uint8(a).ravel(), bins=bins)
        plt.title(title)
    plt.tight_layout()
    plt.show(block=block)

def plot_series(
    y: ArrayLike,
    x: Optional[ArrayLike] = None,
    title: str = "serie",
    labels: Optional[Sequence[str]] = None,
    block: bool = True,
) -> None:
    """
    Dibuja una o varias series 1D. Si y es lista de arrays, grafica múltiples curvas.
    """
    import matplotlib.pyplot as plt  # type: ignore

    def _is_multi(obj) -> bool:
        return isinstance(obj, (list, tuple)) and len(obj) > 0 and hasattr(obj[0], "__len__")

    plt.figure(title)
    if _is_multi(y):
        ys: List[np.ndarray] = [np.asarray(v) for v in y]  # type: ignore
        for i, yy in enumerate(ys):
            xx = np.arange(len(yy)) if x is None else np.asarray(x)
            lab = labels[i] if labels and i < len(labels) else f"serie {i}"
            plt.plot(xx, yy, label=lab)
        plt.legend()
    else:
        yy = np.asarray(y)
        xx = np.arange(len(yy)) if x is None else np.asarray(x)
        plt.plot(xx, yy)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show(block=block)

def describe_array(a: ArrayLike) -> Mapping[str, Union[str, Tuple[int, ...], float, int]]:
    """
    Devuelve un resumen para logs/aserciones: shape, dtype, min, max, mean, std, nans.
    Útil para prints cuando un test falla y querés contexto.
    """
    arr = np.asarray(a)
    info = {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "min": float(np.nanmin(arr)) if arr.size else float("nan"),
        "max": float(np.nanmax(arr)) if arr.size else float("nan"),
        "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
        "std": float(np.nanstd(arr)) if arr.size else float("nan"),
        "n_nans": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
    }
    return info

__all__ = [
    "show_img",
    "imshow_grid",
    "show_hist",
    "plot_series",
    "describe_array",
    "normalize_to_uint8",
    "cv2_backend",
    "can_use_highgui",
]
