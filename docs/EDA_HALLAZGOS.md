# Hallazgos del EDA de SpaceLAB

Ejecución local del 7 de noviembre de 2026, con semilla 42. Resultados calculados por `notebooks/01_EDA_SpaceLAB.ipynb`; no incluyen entrenamiento.

## Integridad

| Partición | Imágenes | Etiquetas | Objetos anotados válidos |
| --- | ---: | ---: | ---: |
| train | 5.926 | 5.926 | 22.155 |
| test | 801 | 801 | 3.964 |
| oculto | 1.743 | 0 | Desconocidos |

Las cifras coinciden con la entrega oficial. No se detectaron imágenes ilegibles, etiquetas ausentes, vacías o huérfanas, líneas YOLO inválidas ni anotaciones idénticas duplicadas bajo una tolerancia normalizada de 1e-6. Todas las imágenes son JPEG RGB de tres canales. Estas comprobaciones estructurales no certifican la exactitud semántica de cada anotación: la inspección visual sigue siendo necesaria.

## Clases en train

| Clase | Objetos | Participación | Imágenes con la clase |
| --- | ---: | ---: | ---: |
| small_boat | 5,600 | 25.28% | 1,843 |
| cargo_ship | 5,600 | 25.28% | 2,642 |
| fishing_vessel | 4,988 | 22.51% | 714 |
| warship | 4,255 | 19.21% | 1,463 |
| work_vessel | 1,712 | 7.73% | 936 |

La razón entre la clase más frecuente y la menos frecuente es 3.27. `work_vessel` es la minoritaria. Esto motiva evaluar su cobertura y rendimiento por separado en una etapa posterior; las frecuencias por sí solas no demuestran sesgo ni justifican un método de balanceo concreto.

## Geometría, escala y densidad en train

- Ancho: 481–1500 px; alto: 411–1500 px. Hay dimensiones menores de 600 px, aunque el documento presenta 600–1.500 px como rango general.
- Geometría: 2.982 imágenes cuadradas, 2.617 horizontales y 327 verticales.
- Mediana del área relativa de las cajas: 0.2636%. El 21.83% ocupa menos del 0,05% de la imagen.
- Densidad máxima: 316 objetos en una imagen. Hay 1 imagen(es) con más de 300 objetos, el máximo oficial de predicciones permitidas por imagen. Conviene tener presente este límite al interpretar el recall.

| Lado largo hipotético | Mediana del lado corto de caja | Cajas con lado corto <8 px |
| ---: | ---: | ---: |
| 640 px | 21.33 px | 13.24% |
| 1024 px | 34.13 px | 3.09% |

Las proyecciones son cálculos geométricos sin redimensionar imágenes. No equivalen a una evaluación de detección ni demuestran que una resolución o estrategia de recortes sea superior. Justifican formular comparaciones posteriores.

## Fuga de información y validación

No se encontraron imágenes idénticas mediante SHA-256 de archivos o de píxeles nativos, ni candidatos de escena compartidos entre train/test al retirar el sufijo `__tX_Y`. Estas reglas no descartan imágenes similares, capturas relacionadas o recortes solapados que tengan nombres distintos. La fuente se infiere del prefijo del nombre y no sustituye metadatos de adquisición. El origen de las imágenes anonimizadas del oculto sigue desconocido.

La auditoría estructural leyó las etiquetas de test; los análisis exploratorios se centraron en train. Los baselines oficiales utilizan test para validación y ajuste de umbral. Para conservar una comprobación final independiente, el siguiente paso será acordar una validación dentro de train y revisar los posibles grupos. Aquí no se crea ninguna partición ni se ajusta ningún modelo.

## Reproducción y límites

Las 22 celdas se ejecutaron sin errores con CPU. Se probaron además etiquetas inválidas, no finitas, duplicadas, ausentes, vacías y huérfanas; imágenes corruptas; IDs ambiguos; y la búsqueda de dataset con rutas anidadas o ausentes. Se comprobó que tamaño y fecha de modificación de todos los archivos originales permanecieran iguales tras la ejecución.

Los CSV completos, seis figuras y el notebook ejecutado están en `reports/eda/`, excluido de Git. El notebook fuente no contiene salidas ni imágenes incrustadas. La compatibilidad de Colab se preparó mediante dependencias y rutas portables; la ejecución verificada se realizó localmente en Windows con Python 3.13, no en una sesión remota de Colab.

Huellas de esta entrega, reproducibles a partir del inventario ordenado:

- Imágenes: `fb97916f9e0c7225e5a89d5d428942fddab7b5c4f63a81f2c02273ff4b32f5db`.
- Etiquetas: `efaaa8d0ca89670dd439864fc4ea035d18ff24eea39cba8dc1bf02fe993a9e0f`.

Las referencias y la distinción entre principios bibliográficos y adaptaciones específicas de SpaceLAB están en el notebook. Ningún resultado permite aún seleccionar arquitectura, optimizador, augmentation o configuración de entrenamiento.
