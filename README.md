# SpaceLAB

## Un solo notebook para Colab web

Archivo: [`SpaceLAB_Colab.ipynb`](SpaceLAB_Colab.ipynb).

1. En https://colab.research.google.com/, elegir Archivo → Subir notebook y seleccionar ese archivo local.
2. Elegir Entorno de ejecución → Cambiar tipo → GPU T4.
3. Ejecutar las celdas en orden y autorizar Google Drive. El código está incrustado;
   no hace falta clonar GitHub, subir carpetas o utilizar la extensión de VS Code.
4. La configuración inicial prepara el entorno y el dataset. Para iniciar los
   experimentos, poner `RUN_PIPELINE=True` y ejecutar la sección 6.

La celda 3 de contenido muestra el nombre de la GPU realmente asignada. El dataset
se obtiene de la carpeta compartida configurada, con acceso montado o API autorizada.
Resultados y checkpoints se guardan en el STORAGE de Drive, con presupuesto persistente.
El acceso de la cuenta y la disponibilidad de T4 dependen de Google; el notebook
no se ha ejecutado aquí ni se han medido entrenamientos con él.

La copia desde Drive y la extracción de ZIP muestran progreso de archivos y bytes.
Primero se cuenta el contenido para obtener un total real. También se guarda
`/content/spacelab_dataset.progress.json` con fase, porcentaje y tiempo transcurrido.
Estas barras aparecen al ejecutar la versión actualizada; una copia que ya está
corriendo con una versión anterior no adquiere progreso retroactivamente.

Para reconstruir el archivo después de cambiar módulos o configuraciones:
`scripts/create_single_colab.ps1` y después
`scripts/package_colab_notebooks.ps1 -NotebookPaths SpaceLAB_Colab.ipynb`.

## Pipeline experimental T4 — preparado, no ejecutado

Diseño, matriz completa, presupuesto y verificaciones pendientes:
[`EXPERIMENTAL_DESIGN.md`](EXPERIMENTAL_DESIGN.md).
Índice de notebooks: [`experiments/README.md`](experiments/README.md).
Dependencias futuras: [`requirements.txt`](requirements.txt).

El [dataset compartido está en Google Drive](https://drive.google.com/drive/folders/1vgxmmpi5zclSXRo_y2GwNIkx35QFqr5t).
En Colab, añadir un acceso directo en Mi unidad y preparar los datos con
`experiments/00_validation_protocol/00_prepare_data.ipynb` antes de iniciar el presupuesto.

Todos los entrenamientos, evaluaciones y benchmarks nuevos empiezan desactivados.
Usar el mismo `STORAGE` persistente en todos los notebooks para compartir protocolo,
calibración y reloj de 90 minutos. El test solo se abre después de congelar la selección.
Este código nuevo no se ha ejecutado ni verificado funcionalmente.

Detección de cinco clases de embarcaciones en imágenes satelitales ópticas RGB.
El documento oficial y los baselines están en [`kit_participantes/`](kit_participantes/).

## Etapa 1: análisis exploratorio

Notebook: [`notebooks/01_EDA_SpaceLAB.ipynb`](notebooks/01_EDA_SpaceLAB.ipynb).
Hallazgos de la ejecución local: [`docs/EDA_HALLAZGOS.md`](docs/EDA_HALLAZGOS.md).

[Abrir en Google Colab](https://colab.research.google.com/github/garzonlj/SpaceLAB/blob/main/notebooks/01_EDA_SpaceLAB.ipynb)
(cuando el notebook esté publicado en la rama `main`). Seleccionar CPU; no necesita GPU.
Cada compañero debe disponer de su copia local del dataset: las imágenes no se versionan.

1. Descomprimir los datos en el entorno local, `/content` o Google Drive.
2. Si se usa Drive, ejecutar la celda opcional de montaje.
3. Ajustar `DATASET_ROOT` si la búsqueda automática no encuentra una única raíz.
   Acepta tanto `spacelab_dataset/` como `spacelab_dataset/spacelab_dataset/`.
   También puede definirse la variable de entorno `SPACELAB_DATASET`.
4. Ejecutar las celdas en orden. La primera instala únicamente las dependencias ausentes.

Para usar Jupyter o la extensión Jupyter de VS Code en Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-eda.txt
```

Seleccionar `.venv` como kernel del notebook. En macOS/Linux, usar `python3 -m venv .venv`
y `.venv/bin/python -m pip install -r requirements-eda.txt`.

El notebook audita imágenes y etiquetas, comprueba clases y cajas YOLO, y genera
distribuciones de clases, coocurrencia, resoluciones, escala de objetos y densidad.
También registra duplicados exactos y candidatos de escena compartida por nombre.
La auditoría estructural incluye `test`; las gráficas exploratorias usan `train` por defecto.
No calcula rendimiento de modelos ni implementa entrenamiento, particiones o balanceo.

Los CSV completos, figuras, hallazgos y `manifest.json` se escriben en `reports/eda/`,
fuera de los datos originales y excluidos de Git. El manifiesto registra versiones,
configuración y huellas de contenido; los hashes requieren leer todas las imágenes.
`HASH_IMAGES=False` permite omitir ese coste, pero deja incompleta la detección de duplicados.

Los archivos vacíos se marcan como candidatos a negativos: una etiqueta ausente no
se interpreta como una imagen sin barcos. Los grupos por nombre son hipótesis y los
hashes exactos no descartan escenas similares o recortes solapados.

Antes de subir un notebook a GitHub, limpiar sus salidas: pueden contener imágenes
incrustadas aunque los archivos JPG estén ignorados. El notebook fuente se entrega sin salidas.
