# Índice de experimentos — preparados, sin ejecutar

## Arranque desde VS Code conectado a Colab

La primera celda de cada notebook experimental contiene una copia comprimida del
código y las configuraciones. Al ejecutarla en Colab se prepara automáticamente
en `/content/SpaceLAB_runtime/<versión>/`, sin subir carpetas ni clonar GitHub.
El paquete contiene únicamente `.py`, `.yaml` y requirements.txt: no incluye
imágenes, pesos, resultados ni los notebooks del kit. Los datos siguen viniendo de Drive.

Después de actualizar el notebook local, recargarlo y ejecutar su primera celda
con el kernel de Colab seleccionado. Si el kernel ya importó otra versión de
SpaceLAB, reiniciarlo antes. El empaquetado no instala dependencias de entrenamiento
ni inicia experimentos; RUN_SETUP y las acciones costosas siguen desactivados.

Si se modifica src/, configs/ o un YAML experimental, volver a empaquetar mediante
`scripts/package_colab_notebooks.ps1` antes de ejecutar en remoto. El paquete se
transporta dentro del notebook; no sincroniza cambios posteriores en tiempo real.
Un REPO_ROOT explícito permite trabajar con un checkout remoto gestionado manualmente.

Cada notebook contiene objetivo, metodología, configuración editable, rutas,
guardas de ejecución, acceso al protocolo compartido y conclusiones condicionadas
a resultados reales. Abrirlos desde un checkout del repositorio en Colab.
Los entrenamientos, evaluaciones y benchmarks comienzan desactivados. Por solicitud
posterior del equipo, `00_create_splits.ipynb` tiene RUN_SPLITS=True y prepara los
datos antes de iniciar el presupuesto. No se instala nada al importar el paquete.

Al ejecutar la primera celda, `INSTALL_MISSING_NOTEBOOK_DEPS=True` instala únicamente
dependencias básicas ausentes (`filelock`, NumPy, Pandas, Pillow y PyYAML) usando
el Python del kernel activo. No instala Torch/CUDA ni ejecuta experimentos.
Puede desactivarse en un entorno offline ya preparado. La celda muestra la ruta
de Python y avisa si el kernel es Windows local; instalar dependencias no conecta
ese kernel a la T4. Seleccionar el servidor Colab desde el selector de kernel.

La [matriz completa y decisiones metodológicas](../EXPERIMENTAL_DESIGN.md)
es la referencia para orden, presupuestos, variables y limitaciones.

## Árbol del sistema

```text
configs/
  global.yaml                 # Receta compartida y presupuesto
  t4.yaml                     # Política y batches conservadores
  experiments.yaml            # Catálogo de candidatos, no cola automática
src/spacelab/
  __init__.py
  io.py                       # Metadatos atómicos, rutas, hashes
  preparation.py              # Copia futura explícita de Drive o ZIP
  dataset.py                  # Lectura por imagen, letterbox y augmentations
  splits.py                   # Holdout multilabel y dos folds opcionales
  models.py                   # Adaptadores YOLO / Faster R-CNN
  training.py                 # Bucle, AMP, benchmark, OOM y estimaciones
  evaluation.py               # Evaluador oficial y diagnósticos
  experiment_registry.py      # Configuraciones, cohortes, selección congelada
  time_budget.py              # Reloj persistente y watchdog secuencial
  checkpointing.py            # Modelo, optimizador, AMP, RNG y protocolo
  inference.py                # Coordenadas nativas y fusión de tiles
  visualizations.py           # Figuras de datos reales guardados
  worker.py                   # Entrada del subproceso presupuestado
experiments/
  00_validation_protocol/
    00_prepare_data.ipynb
    00_create_splits.ipynb
    01_t4_benchmark.ipynb
    validation_protocol.md
  01_model_comparison/{yolo26n,yolo11n}/
  02_resolution/{resolution_640,resolution_896,resolution_1024}/
  03_augmentation/{baseline,geometric,photometric}/
  04_fine_tuning/{full_fine_tuning,partial_freezing,progressive_unfreezing}/
  05_hyperparameters/{learning_rate,optimizer,regularization}/
  06_class_imbalance/{original_distribution,controlled_oversampling}/
  07_optional_models/{yolo26s,faster_rcnn_mobilenet}/
  08_cross_validation/grouped_two_fold/
  09_inference/{confidence_threshold,nms,tiled_inference}/
  09_inference/final_evaluation.ipynb
  10_final_selection/finalist/
  10_final_selection/compare_all.ipynb
  10_final_selection/generate_submission.ipynb
reports/experiments/README.md
requirements.txt
EXPERIMENTAL_DESIGN.md
```

Cada subcarpeta de entrenamiento del árbol incluye `config.yaml`, `train.ipynb`
y `validate.ipynb`. Cada etapa 01–08 tiene `comparison.ipynb`. Las carpetas de
inferencia contienen `config.yaml`, `experiment.ipynb` y `validate.ipynb`.
El EDA existente en `notebooks/01_EDA_SpaceLAB.ipynb` se conserva.

## Qué hace cada notebook

| Notebook o familia | Propósito y parámetros principales |
|---|---|
| 00_prepare_data | Montar la carpeta Drive del usuario mediante acceso directo; copiar a /content o extraer ZIP explícito antes de la sesión |
| 00_create_splits | Crear una sola partición compartida; GROUPS_CSV opcional con grupos verificados |
| 01_t4_benchmark | Medir batches, lectura, inferencia muestral y VRAM; CONFIG_FILES selecciona configuraciones |
| */train | Entrenar la configuración de su carpeta; CONFIG_FILE, SELECTED_RUN, RESUME_CHECKPOINT; RUN_TRAINING=False |
| */validate | Cargar RUN_PATH/checkpoints/best.pt y evaluar val interna de forma independiente; RUN_EVALUATION=False |
| 08/grouped_two_fold/train | Dos entrenamientos secuenciales agrupados, solo con grupos verificados y ENABLE_OPTIONAL=True |
| 08/grouped_two_fold/validate | Evaluar cada fold especificando RUN_PATH al subdirectorio fold_0 o fold_1 |
| */comparison | Leer manifiestos reales, mostrar cohortes y ordenar solo la elegida en COHORT |
| confidence_threshold/{experiment,validate} | Reutilizar VALIDATION_RUN/predictions para barrido oficial de umbral sin GPU adicional |
| nms/{experiment,validate} | Comparar IoU solo si la salida del detector aplica NMS; rechazar cambio inoperante end-to-end |
| tiled_inference/{experiment,validate} | Inferencia por tiles, coordenadas a imagen original y fusión NMS por clase |
| 09/final_evaluation | Congelación previa y única evaluación de test, sin ajustar su umbral |
| 10/finalist/{train,validate} | Fase opcional con train interno completo, presupuesto reasignado y comparación separada de screening |
| 10/compare_all | Reunir resultados, señalar incompletos y mantener rankings separados por cohorte |
| 10/generate_submission | Congelar configuración/umbral; test opcional único y ZIP del oculto verificado por el kit |

Las configuraciones alternativas H02/H04/H06–H09 se eligen cambiando CONFIG_FILE
en los notebooks de su familia. No generan un producto cartesiano. Las variantes
de NMS o tiles se eligen explícitamente en INFERENCE_OVERRIDES.

## Resultados de una ejecución futura

```text
STORAGE/
  budget.json
  budget.lock
  gpu.lock
  protocol.json
  calibration/<modelo-resolución-batch-amp>.json
  jobs/<acción-uuid>/
    request.json
    result.json
    worker.log
    experiment_manifest.json          # en entrenamientos
    checkpoints/{best,last}.pt
    metrics/
      training_history.csv
      validation_metrics.json
      metrics_by_class.csv
      official_score.json
      runtime.json
      training_runtime.json
      slices.csv
      errors.csv
    figures/
      learning_curves.png
      confusion_matrix.png
      precision_recall.png
      predictions_examples.png
    predictions/
    validation_provenance.json        # en validaciones/inferencia
    submission.zip                    # solo generación final
  frozen_selection.json
  official_test_started.json
  submission_started.json
```

El esquema es lógico por configuración, pero cada acción conserva su directorio:
train guarda pesos e historial; validate guarda sus métricas/predicciones/figuras
y referencia al train. La primera validación deja su score asociado al train para
el registro. Una validación posterior no sobrescribe esa asociación.

Para reanudar, indicar el last.pt de un entrenamiento previo con exactamente la
misma configuración. El nuevo job conserva la historia y copia best.pt previo;
el entrenamiento original no se modifica. Reanudar no reinicia budget.json.
Los pesos usan un checkpoint propio de state_dict; cargarlos con los adaptadores
de este proyecto, no directamente con YOLO('best.pt') del trainer nativo.

Guardar STORAGE en Drive para persistencia, pero leer las imágenes de /content.
Conservar siempre el mismo STORAGE durante los 90 minutos, incluyendo entre
notebooks. Los resultados grandes, pesos y notebooks con salidas no se versionan.
