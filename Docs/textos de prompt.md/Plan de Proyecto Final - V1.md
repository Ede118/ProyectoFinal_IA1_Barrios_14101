# Proyecto Final IAI 2025
## Clasificador de piezas por visión artificial, controlado por voz, con estimación bayesiana

> Carrera: Ing. Mecatrónica · Asignatura: Inteligencia Artificial I · Año: 2025

---

## 0) Resumen ejecutivo
Desarrollar un agente que integra: (i) reconocimiento de voz (KNN implementado por el estudiante) para comandos “Proporción”, “Contar”, “Salir”; (ii) clasificación de imágenes de piezas (K-Means) para identificar tornillos, tuercas, arandelas y clavos; (iii) estimación bayesiana de la proporción de piezas en la caja a partir de una muestra de 10 objetos. Se entrega código + informe PDF + demostración.

---

## 1) Requisitos oficiales (compendiados)
- Trabajo **individual**. Informe **PDF** + defensa oral.
- Documento con: Título, Resumen (≤200 palabras), Introducción (visión, voz, problema), **Especificación del agente** (tipo, **tabla REAS**, propiedades del entorno), **Diseño** (algoritmos por etapa), **Código** (adjunto), **Ejemplos** con resultados, **Resultados** (datos, pruebas, estadísticas), **Conclusiones**, **Bibliografía** y uso de **IA generativa** con prompts y referencias.
- **Escenario**: cajas con 1000 piezas de 4 tipos. La caja real pertenece a uno de 4 perfiles de proporciones (H1…H4). El sistema extrae **10 piezas** al azar, las **clasifica** visualmente y estima la proporción de la caja completa. Los comandos de voz disparan acciones.
- **Bases de datos** a construir:
  - **Imágenes**: ≥6 imágenes por objeto y por clase, en distintas poses/condiciones.
  - **Voz**: varias muestras por palabra, al menos **5 personas** distintas.
- **Algoritmos**:
  1) Voz: **KNN** implementado por el estudiante.
  2) Imágenes: **K-Means**.
  3) Estimación: **clasificador bayesiano**.

---

## 2) Arquitectura del agente
```mermaid
flowchart LR
A[Inicio] --> B[Selección aleatoria de caja (simulada)]
B --> C[Extracción de 10 piezas]
C --> D[Captura/lectura de imágenes]
D --> E[Extracción de características de imagen]
E --> F[K-Means: asignación a 4 clusters]
F --> G[Conteo por clase en la muestra]
G --> H[Estimación bayesiana de proporciones en la caja]
H --> I[Entrada de voz]
I --> J[KNN (voz): reconocer comando]
J -->|"Proporción"| K[Mostrar estimación de proporciones]
J -->|"Contar"| L[Mostrar conteo de la muestra]
J -->|"Salir"| M[Finalizar]
```

**Tipo de agente**: reactivo-basado en objetivos (respuesta a comandos) con componentes de aprendizaje.

**Entorno**: parcialmente observable, estocástico (muestreo aleatorio), secuencial corta duración, estático durante cada episodio, discreto en acciones y percepciones (tras cuantización de features).

**REAS (Resumen)**:
- **R**: precisión de clasificación, tiempo de respuesta, facilidad de uso.
- **E**: iluminación variable, fondo neutro recomendado, micrófonos variados.
- **A**: clasificar, estimar, responder a comandos.
- **S**: cámara, micrófono, dataset local.

---

## 3) Modelos y algoritmos
### 3.1 Reconocimiento de voz (KNN propio)
- **Entrada**: audio mono (8–16 kHz), ventanas 25 ms, hop 10 ms.
- **Features** (mínimas viables): **MFCC** (12–20 coef.), energía media, tasa de cruces por cero. Alternativa simple si se evita MFCC: espectro por FFT con bancos de filtros mel y log-energías.
- **KNN**: distancia Euclídea o coseno, **K=3–7**. Normalizar features por z-score.
- **Entrenamiento**: almacenar vectores de referencia por clase {“proporción”, “contar”, “salir”}. Balancear por hablante.
- **Validación**: leave-one-speaker-out y matriz de confusión.

### 3.2 Visión: K-Means sobre features de imagen
- **Preprocesado**: recorte/segmentación simple (fondo liso), escalado, normalización.
- **Features**: 2–3 familias para robustez:
  - **Color**: histogramas HSV (H y S, 16–32 bins).
  - **Forma**: Hu moments o compactness/aspect ratio/solidez.
  - **Textura**: LBP o energía de filtros Gabor.
- **K-Means**: K=4, init k-means++ con centroides preinicializados desde ejemplos etiquetados.
- **Asignación**: cada pieza → label ∈ {tornillo, clavo, arandela, tuerca}. Resolver el **mapping cluster→clase** con un conjunto pequeño etiquetado (argmax por mayoría o matching húngaro sobre matriz de confusión).

### 3.3 Estimación bayesiana de la caja
- **Hipótesis** fijas H1..H4 (perfiles de proporciones p_k = (p1, p2, p3, p4)).
- **Observación**: conteos en la muestra, \(\mathbf{n}=(n_1,\dots,n_4), \sum n_i=10\).
- **Verosimilitud** multinomial: \(\mathcal{L}(\mathbf{n}\mid H_k) \propto \prod_i p_{k,i}^{\,n_i}\).
- **Prior**: uniforme sobre {H1..H4} o informada si la cátedra lo pide.
- **Posterior**: \(\Pr(H_k\mid \mathbf{n}) \propto \mathcal{L}(\mathbf{n}\mid H_k)\,\Pr(H_k)\). Reportar la **más probable** y el vector de probabilidades.
- **Alternativa**: modelo Dirichlet-Multinomial si se estima una proporción continua en lugar de elegir entre H1..H4.

---

## 4) Diseño de datos y adquisición
### 4.1 Imágenes
- 4 clases × ≥6 imágenes/objeto × ≥1 objeto por clase. Mejor: ≥20 por clase.
- Variar pose/rotación, leve variación de luz, fondo **uniforme**.
- Formato recomendado: 640×480 JPG/PNG.

### 4.2 Voz
- Palabras: “proporción”, “contar”, “salir”.
- ≥5 hablantes × ≥10 repeticiones por palabra. Grabar en lugares distintos.
- 16 kHz, 16-bit PCM, WAV mono. Recorte automático por VAD simple.

Estructura de carpetas de datos:
```
./data/
  audio/
    proporcion/ speaker_01_001.wav ...
    contar/     speaker_01_001.wav ...
    salir/      speaker_01_001.wav ...
  images/
    tornillo/   img_001.jpg ...
    clavo/      img_001.jpg ...
    arandela/   img_001.jpg ...
    tuerca/     img_001.jpg ...
```

---

## 5) Esqueleto de código (Python)
```
project/
  main.py
  config.py
  audio/
    features.py      # MFCC / ZCR / energía
    knn.py           # KNN propio
    vad.py           # detector de voz simple
  vision/
    features.py      # color/forma/textura
    kmeans.py        # wrapper + mapping cluster→clase
  bayes/
    estimator.py     # posterior sobre {H1..H4}
  utils/
    io.py, viz.py, metrics.py, seed.py
  data/ ...
  docs/ informe.tex|md
```

Pseudocódigo de lazo principal:
```python
# main.py
imgs = capturar_muestra(10)
X_img = [vision.features.extract(img) for img in imgs]
labels = vision.kmeans.predict(X_img)  # {tornillo,clavo,arandela,tuerca}
conteo = contar(labels)
post = bayes.estimator.posterior(conteo)
cmd = audio.knn.recognize(stream())    # "proporción" | "contar" | "salir"
if cmd == "proporción": mostrar(post)
elif cmd == "contar":  mostrar(conteo)
else: salir()
```

---

## 6) Métricas y validación
- **Voz**: accuracy macro, matriz de confusión por hablante. Medir WER si segmentas frases.
- **Visión**: accuracy por clase, macro-F1. Control de iluminación y fondo.
- **Bayes**: tasa de acierto de hipótesis verdadera; curva de confianza vs tamaño de muestra.
- **Tiempo**: latencia total por ciclo < 1 s preferible.

---

## 7) Plan de trabajo (10 días efectivos)
**Día 1–2**: Setup repo, arbol de carpetas, script de captura, VAD, MFCC básico.

**Día 3–4**: KNN voz con validación LOSO, baseline >90% en 3 clases.

**Día 5**: Features de imagen (color+forma). Dataset mínimo por clase.

**Día 6**: K-Means + mapping cluster→clase. Validación en hold-out.

**Día 7**: Módulo Bayes + pruebas con conteos sintéticos.

**Día 8**: Integración loop principal + CLI/simple GUI.

**Día 9**: Métricas, gráficos y tablas. Capturas de pantalla.

**Día 10**: Informe PDF completo, ensayo de demo.

---

## 8) Informe final — plantilla mínima
1. **Título**.
2. **Resumen**.
3. **Introducción**: visión, voz, problema.
4. **Agente y entorno**: tipos, REAS, propiedades.
5. **Diseño**: voz (features, K, distancia), visión (features, K-means, mapping), Bayes.
6. **Datos**: cómo se capturó todo, ejemplos.
7. **Resultados**: métricas, figuras, discusión.
8. **Conclusiones y futuros**.
9. **Código**: referencia a repo/zip.
10. **Bibliografía** y **uso de IA generativa** con prompts.

---

## 9) Riesgos y mitigaciones
- **Audio ruidoso**: VAD + normalización + trimming. Pedir repetición si SNR < umbral.
- **Iluminación**: fondo uniforme, luz difusa; normalizar brillo/contraste.
- **K-Means inestable**: múltiples inits; fijar semilla; k-means++.
- **Desbalance**: muestrear balanceado por clase/hablante; ponderación en distancia.

---

## 10) Preguntas de chequeo (para demostrar dominio)
1) ¿Por qué KNN en voz puede funcionar mejor que un softmax pequeño con tan pocos datos?
2) ¿Cuándo K-Means falla separando por color, y qué feature alternativo elegirías?
3) ¿Qué prior usarías si el proceso real privilegia tornillos y clavos? Justificá.
4) ¿Cuál es la complejidad temporal por ciclo del agente y cómo la reducirías?
5) ¿Cómo demostrarías robustez a cambios de hablante o de iluminación en la defensa?

---

## 11) Anexos breves
- **Fórmula de la posterior multinomial**: \(\Pr(H_k\mid \mathbf{n}) = \dfrac{\Pr(H_k)\,\prod_i p_{k,i}^{n_i}}{\sum_j \Pr(H_j)\,\prod_i p_{j,i}^{n_i}}\).
- **Hu moments**: invariantes a traslación, escala y rotación; útiles para arandelas vs tuercas.
- **ZCR**: sensible a consonantes fricativas; combinar con energía para segmentación.

