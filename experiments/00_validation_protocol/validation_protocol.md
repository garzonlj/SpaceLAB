# Protocolo común de validación

No se ha creado ninguna partición nueva durante la implementación. El notebook
`00_create_splits.ipynb` la creará cuando el usuario active `RUN_SPLITS`.

## Origen y preparación

Dataset compartido: [carpeta de Google Drive](https://drive.google.com/drive/folders/1vgxmmpi5zclSXRo_y2GwNIkx35QFqr5t).
Añadir un acceso directo a Mi unidad y ajustar `SOURCE` en `00_prepare_data.ipynb`.
El nombre `spacelab_dataset` del acceso directo es una convención configurable,
no un nombre de carpeta verificado en Drive. Montar Drive autenticándose desde Colab.
Si el origen contiene un ZIP, seleccionar su ruta exacta en `SOURCE_ZIP`.
La copia opcional a `/content` evita entrenar leyendo imágenes desde Drive.
No se infiere la ruta montada a partir del ID del enlace ni se usa gdown sin autenticación.

## Partición persistente

- Universo: exclusivamente `images/train` y `labels/train` originales.
- Holdout objetivo: aproximadamente 80/20, semilla 42.
- Heurística repetida de asignación de unidades, equilibrando número de imágenes,
  presencia multilabel, objetos por clase, tres rangos de área nativa y fuentes por nombre.
- Screening: aproximadamente 25% del entrenamiento interno, seleccionado una vez con
  la misma heurística; toda configuración de screening reutiliza esos mismos IDs.
- Las cinco clases deben estar presentes en ambos lados. Si la heurística no lo
  consigue, se detiene y se pide revisar grupos/fracción, sin esconder clases ausentes.
- Persistencia: `STORAGE/protocol.json`. No se regenera al abrir otros notebooks.
  Contiene IDs, conteos, tamaños, fuente inferida, grupo, dimensiones y SHA-256 de
  cada imagen/etiqueta del train original. Las rutas son relativas al dataset.
- La identidad del protocolo incluye su contenido completo. Los checkpoints registran
  esa identidad y rechazan una reanudación con configuración o protocolo distintos.

## Grupos y fuga

El EDA solo sugiere candidatos al retirar un sufijo `__tX_Y`; no confirma escenas.
No se emplea esa inferencia como si fuera verdad. Se persiste el solapamiento de
candidatos entre el entrenamiento y validación internos para revisión posterior.

Si se dispone de grupos fiables, aportar un CSV `id,group,verified` con cobertura
exacta de train, grupo no vacío y `verified=true` en todas las filas. La partición,
screening y los dos folds opcionales respetan esas unidades completas. No se
fusionan escenas utilizando etiquetas de test o del oculto. La detección de un
duplicado exacto durante la creación detiene la división para revisar agrupación.

Sin grupos verificados, la unidad es la imagen. El balance multilabel es aproximado,
no garantiza independencia de escenas ni preservación exacta de todas las marginales.
No se aplica StratifiedKFold de clase única. La validación cruzada usa solo el
entrenamiento interno, tiene dos folds y permanece desactivada.

## Test, selección y final

Los 801 ejemplos oficiales de test no participan en selección de arquitectura,
resolución, augmentation, umbral, checkpoint, early stopping o hiperparámetros.
La auditoría estructural previa del EDA se reconoce; no se utiliza para ajustar
modelos. El oculto nunca se utiliza para métricas supervisadas.

Cada época completa se evalúa sobre validación interna con `evaluar.py`; el
checkpoint best maximiza el puntaje oficial, calibrando el umbral solo allí.
Se guardan predicciones a confianza mínima 0.001 y hasta 300 detecciones por imagen.
El evaluador lee los TXT oficiales ya redondeados, iguales a los artefactos de envío.

La consulta final requiere `frozen_selection.json`, creado antes de abrir test:
SHA del checkpoint, configuración, protocolo, umbral y receta de inferencia.
El umbral permanece fijo al evaluar test y predecir oculto. Los marcadores de inicio
impiden repetir silenciosamente una consulta final. Si se interrumpe, revisar logs
y justificar cualquier recuperación; no borrar marcadores para hacer tuning.

## Comparabilidad y limitaciones

Una cohorte comparte protocolo, fase, IDs de screening, épocas objetivo, batch,
receta y semilla; solo se excluye de su firma la variable declarada de la ablación.
Entrenamientos interrumpidos o con OOM que cambia el batch no se promueven.
Early stopping implica distinta exposición y queda fuera del ranking estricto.
Screening y finalista no se ordenan como si hubieran entrenado igual.

El sobremuestreo cambia deliberadamente la exposición y el número de updates;
el informe muestra su efecto sobre todas las clases, no solo work_vessel.
Comparar sus resultados como intervención de muestreo, sin afirmar igualdad de cómputo.
La diferencia pequeña entre dos resultados de una semilla no prueba significancia.

Fundamento general: Elgendy, caps. 4 y 6; Lakshmanan, Görner y Gillard, caps. 5 y 8.
La heurística específica, hashes, tamaño del screening y receta son decisiones del
proyecto pendientes de comprobación, no algoritmos atribuidos a los libros.
