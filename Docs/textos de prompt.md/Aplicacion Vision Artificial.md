## Visión (imágenes)

> Objetivo

Asignar a cada una de las 10 imágenes a una categoría o "Non Identifiable Class" $\{\text{tornillo, tuerca, arandela, clavon, NIC}\}$ y devolver el conteo por clase. El objetivo principal es conseguir un macro-accuracy  $\geq 85 \%$  en validación interna, contando con una latencia $<$ 200 mili segundos por imagen. 

> Preprocesado.

- **Redimensionado**
	La imagen tomada se redimensiona a una resolución de `[256×256] pixeles` \[explicar por qué se hace\].
- **Normalización** 
	Se normaliza la iluminación (p. ej., CLAHE opcional).
- **Segmentación** 
	fondo/objeto en **HSV** con umbral por rango y tolerancia `[τ]`; limpieza con morfología. \[¿por qué se hace esto? ¿es necesario? Redactar de nuevo.\]

\[¿hace falta este preprocesado? ¿por qué?\]

**Características (vector concatenado).**

\[ Me parece que solo voy a necesitar un filtro de contorno tipo `robert`, `log`, `sobel`. Lo que me interesa es sacar el contorno, no considero que el color sea necesario.\]



1. **Color**: histogramas **HSV** en H y S con `[H=24, S=16]` bins, normalizados (L1). \[¿Realmente necesario?\].
2. **Forma**: **Momentos de Hu (7)** con `log |Hu|` para estabilidad.
3. **Textura**: **LBP** uniforme con radio `[R=2]`, puntos `[P=16]`, histograma normalizado.



**Algoritmo de decisión.**

\[ Otra cosa que planteo: encontrar momentos variables con la rotación: se posiciona el objeto en el centro (traslación o recorte) y luego se analiza el momento no invariable: si al rotar la imagen el momento se modifica -> clavo o tornillo, si al rotar el momento se mantiene casi constante -> arandel o tuerca. Ese me gustaría que fuera el primer filtro.

El segundo filtro esta a discución para diferenciar tornillo-clavo y otro para diferencial arandel-tuerca.\]

- **K-Means** con **K=4**, inicialización **k-means++**, `n_init=[10]`, `max_iter=[300]`.
    
- **Mapeo cluster→clase**: se obtiene con un set chico etiquetado y se asigna por **mayoría** o por **asignación húngara** sobre la matriz de confusión.
    
- **Salida**: etiquetas y `conteo` por clase.
    

**Complejidad.** Para N imágenes y d features, K-Means ≈ `O(N·d·K·I)` por corrida (I iteraciones); aquí N=10 y K=4, así que es instantáneo incluso en CPU mediocre.

**Criterios de parada y robustez.**

- Parar cuando el cambio de centroides < `[ε]` o `iter=300`.
    
- Si **tuerca vs arandela** se confunden, añadir **solidez**, **relación de aspecto** y “hueco interior” por contornos.
    
- Fijar `random_state` y aumentar `n_init` si hay inestabilidad.
    

**Pseudocódigo.**

```python
X = [feat(img) for img in imgs]  # HSV+Hu+LBP
labels = kmeans_fit_predict(X, K=4, n_init=10, seed=SEED)
conteo = count(labels)  # dict clase -> cantidad
```

