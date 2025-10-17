## Entradas de Imágenes
Tipo de Agente

Tabla Reas

| Entorno de Trabajo | Entrada de Imágenes                                                                                                                      |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Descripción        | Se obtiene una serie de 10 imágenes de objetos pertenecientes a una de cuatro categorías $\{\text{tornillo, tuerca, arandela, clavo}\}$. |
| Rendimiento        | Se clasifican correctamente los 10 objetos en su clase correspondiente.                                                                  |
| Entorno            | Ambiente con mismo nivel de iluminación para sacar fotos en las mismas condiciones.                                                      |
| Actuadores         | ?                                                                                                                                        |
| Sensores           | Cámara fotográfica o sensor de visión artificial.                                                                                        |


Propiedades del entorno

| Propiedades              |                                                                                                                                                                      |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Observabilidad           | Parcial. Tengo una imagen bidimensional de mi objeto tridimensional. Solo puedo identificar las características del objeto observado a partir de una imagen (plano). |
| Determinista/Estocástico | Estocástico. Dependiendo de los valores de ciertos parámetros de clasificación, categorizo el objeto en base a la categoria a la que es más probable pertenecer.     |
| Episódico/Secuencial     | Episódico. La imagen de un objeto y su análisis es totalmente independiente de la imagen y análisis de otro objeto diferente.                                        |
| Estático/Dinámico        | Estático. No hay cambio en el entorno durante el proceso de clasificación.                                                                                           |
| Discreto/Contínuo        | Discreto. Las categorias posibles son discretas: $\{\text{tornillo, tuerca, arandela, clavo}\}$                                                                      |
| Agente                   | Agente único.                                                                                                                                                        |

## Entradas de Audio
Tipo de Agente

Tabla Reas

| Entorno de Trabajo | Entrada de Audio                                                                                                                                                      |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Descripción        | Se obtiene un audio con uno de tres comandos de voz: “proporción”, “contar”, “salir”. A partir de la identificación del comando, se realiza la acción correspondiente |
| Rendimiento        | Identificar el comando de voz y realizar la acción acorde.                                                                                                            |
| Entorno            | Ambiente con mínimo o nulo ruido de ambiente, para poder identificar la voz del comando.                                                                              |
| Actuadores         | ?                                                                                                                                                                     |
| Sensores           | Micrófono de sonido o sensor de audio artificial.                                                                                                                     |

Propiedades del entorno

| Propiedades              |                                                                                                                                                                                                                                                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Observabilidad           | Total.  El comando por audio es lo único que se debe analizar, y es el único input que se tiene.                                                                                                                                                                                                                    |
| Determinista/Estocástico | Estocástico. Las máquinas no interpretan "consignas de voz" per se, sino que dependiendo de las diferencias de presiones atmosféricas (y las salidas de variaciones de voltaje) identifican "patrones de voz" asociados a diferencias de potencial. No hay 2 voces de personas iguales, sino que hay similaritudes. |
| Episódico/Secuencial     | Episódico. Cada comando de voz es independiente de los demás.                                                                                                                                                                                                                                                       |
| Estático/Dinámico        | Estático. El entorno no varía mientras el agente identifica el comando. Ello implica que no deberá darse un comando hasta que se identifique y se responda al comando de voz anterior.                                                                                                                              |
| Discreto/Contínuo        | Discreto. Existe una cantidad acotada de comandos de voz aceptables.                                                                                                                                                                                                                                                |
| Agente                   | ?                                                                                                                                                                                                                                                                                                                   |

## Hipótesis y bayes
Tipo de Agente

Tabla Reas

| Entorno de Trabajo | Hipótesis y Procesado bayesiano                                                                                                                                                                                                                                   |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Descripción        | A partir de las $N=10$ piezas sacadas, considerando que se identificaron correctamente sus categorías en el "procesamiento de imágenes", se realiza el aprendizaje bayesiano para calcular las probabilidades de «de qué caja salieron las piezas identificadas». |
| Rendimiento        | Calcular correctamente las probabilidades de que las $N$ piezas sacadas (y considerando que fueron correctamente identificadas y clasificadas) pertenezca a una determinada caja.                                                                                 |
| Entorno            | ?                                                                                                                                                                                                                                                                 |
| Actuadores         | ?                                                                                                                                                                                                                                                                 |
| Sensores           | ?                                                                                                                                                                                                                                                                 |

Propiedades del entorno

| Propiedades              |                                                                                                                                                                                                                |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Observabilidad           | Total.                                                                                                                                                                                                         |
| Determinista/Estocástico | Estocástico. Estamos hablando de probabilidades: a priori, hipótesis y a posteriori.                                                                                                                           |
| Episódico/Secuencial     | Secuencial. Dependiendo de la categoría de la pieza anterior, las probabilidades "a qué caja pertenece" se modifica para las piezas posteriores.                                                               |
| Estático/Dinámico        | Estático. El entorno varia de "pieza a pieza". Mientras no se saque una nueva pieza, el entorno no varía. En ese sentido, lo consideramos "estático" mientras no se tome una acción (sacarse una nueva pieza). |
| Discreto/Contínuo        | ?                                                                                                                                                                                                              |
| Agente                   | ?                                                                                                                                                                                                              |

