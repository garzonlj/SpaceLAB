# Diseño experimental SpaceLAB para una NVIDIA T4

**Estado: código preparado, sin ejecutar.** No se instalaron dependencias, descargaron
pesos, crearon splits, ejecutaron benchmarks, pruebas, entrenamientos o evaluaciones
durante esta etapa. La revisión fue de lectura e inspección estática. El EDA documentado
pertenece a la etapa anterior; no son resultados nuevos del pipeline experimental.

## Evidencia y decisiones

El [EDA](docs/EDA_HALLAZGOS.md) encontró 5.926 imágenes y 22.155 objetos de train.
`work_vessel` tiene 1.712 objetos frente a 5.600 en las clases más frecuentes.
El 21,83% de las cajas ocupa menos de 0,05% del área. La proyección a 640 estima
13,24% de lados cortos inferiores a 8 px, frente a 3,09% con 1024. Son cálculos
geométricos, no resultados de detección. Una escena contiene 316 objetos y el
formato oficial admite como máximo 300 predicciones: ese límite afecta el recall.

Estas observaciones justifican **probar** resolución, cobertura por clase y
transformaciones coherentes con vistas cenitales. No demuestran qué configuración
ganará. La ausencia de duplicados exactos no prueba independencia de escenas.

Los libros fundamentan separar preparación/selección/evaluación, usar baselines,
controlar variables y diagnosticar sobreajuste. La fracción de screening, épocas,
LR, resolución y política de AMP son hipótesis de ingeniería para medir en T4.

## Protocolo

Implementación: [`splits.py`](src/spacelab/splits.py). Detalle:
[`validation_protocol.md`](experiments/00_validation_protocol/validation_protocol.md).

Holdout 80/20 aproximado dentro de train, semilla 42. Estratificación heurística
multilabel de presencia y conteos, añadiendo tamaños y fuente inferida. Con grupos
verificados se asignan unidades completas; sin ellos se documenta la limitación
y los candidatos de escenas solapadas. Screening fijo del 25% del train interno.
Los grupos verificados permiten dos folds opcionales sobre el train interno;
no se ejecutan automáticamente ni se usa CV de cinco particiones.

El protocolo se crea una vez. Incluye IDs, conteos, dimensiones, fuente, grupos,
distribuciones, hashes de imágenes/etiquetas y un identificador de contenido.
Los originales no se alteran. El test oficial se mantiene reservado para después
de congelar configuración, checkpoint, receta de inferencia y umbral.

## Receta común y control de variables

- Modelos principales: YOLO26n y YOLO11n, pesos públicos, cinco clases, 640 px,
  batch 4, screening común, dos épocas objetivo, seed 42, AdamW, LR 0.001,
  betas (0.9,0.999), weight decay 0.0005, AMP FP16 y workers 2.
- Son candidatos sin rendimiento conocido. Modelos opcionales: YOLO26s (batch 2)
  y Faster R-CNN MobileNetV3-FPN (batch 1). No se incluyen m/l/x o ResNet50.
- El bucle PyTorch es compartido y llama al modelo/pérdida de Ultralytics;
  **no es una reproducción del trainer nativo** ni de su receta de optimización.
  Se conserva LR constante, sin optimizer=auto, scheduler oculto, EMA o warm-up
  implícito. Sus adaptadores internos necesitan verificación de compatibilidad.
- El trainer nativo de Ultralytics puede activar mosaic, HSV, traslación, escala
  y flips según versión/defaults. Aquí no se llama a ese trainer: **ninguna de
  esas augmentations se aplica implícitamente**. La baseline realiza letterbox
  bilineal con padding 114, RGB→tensor y división por 255. Las tres recetas son
  disjuntas: baseline; geometría ortogonal+flips; fotometría moderada sin geometría.
- Las rotaciones usan 0/90/180/270°, actualizando centros y dimensiones de cajas.
  No se confunden con rotaciones arbitrarias e interpolaciones de nuevos bordes.
  Las transformaciones solo se aplican a entrenamiento, nunca a val/test.
- El backbone YOLO se obtiene de la sección backbone de su YAML real, con registro
  de nombres de parámetros; Faster R-CNN usa backbone.body. DFL fijo queda excluido
  de optimización. No se congela el mismo índice arbitrario entre arquitecturas.
- En progressive se activa el backbone desde época 2 con requires_grad. Los
  grupos backbone/neck-head existen desde el inicio; los parámetros recién
  activos obtienen estado del optimizador al recibir gradiente. Se preserva el
  momentum de los módulos ya activos, y se guarda el registro de cada etapa.
- No se utiliza dropout como hiperparámetro de detección YOLO ni pérdidas de
  clase ponderadas que no estén verificadas. El sobremuestreo duplica solo
  imágenes de entrenamiento con work_vessel y registra todas las clases afectadas.

## Matriz completa de hipótesis

Todas las filas empiezan **pendientes, sin ejecutar**. Las prioridades indican orden,
no promesa de completar la matriz en 90 minutos. Cada comparación usa validación
interna; las métricas son mAP50, mAP50:95, P/R, F1 macro, score y AP/F1 por clase.
El presupuesto indicado es techo por configuración, no una medición. Los máximos
de una familia no son presupuestos adicionales al límite global.

| ID | Modelo | Hipótesis | Variable | Controladas | Validación / métricas | Prioridad | Techo | Estado |
|---|---|---|---|---|---|---|---|---|
| M01 | YOLO26n | Referencia compacta | Arquitectura | Receta, 640, screening, 2 épocas, batch 4 | Holdout / oficiales+clase | 1 | 15 min | Pendiente |
| M02 | YOLO11n | Alternativa compacta | Arquitectura | Igual que M01 | Holdout / oficiales+clase | 1 | 15 min | Pendiente |
| R01 | Seleccionado M | Referencia espacial | 640 px | Modelo, receta, split, épocas | Holdout / oficiales+tamaño | 2 | 10 min | Pendiente |
| R02 | Seleccionado M | Más detalle útil | 896 px | Igual que R01 | Holdout / oficiales+tamaño | 2 | 10 min | Pendiente |
| R03 | Seleccionado M | Objetos pequeños | 1024 px | Igual que R01 | Holdout / oficiales+tamaño | Opcional | 10 min | Desactivado |
| A01 | Seleccionado M | Referencia mínima | Baseline | Modelo, 640, receta, split | Holdout / oficiales+clase | 3 | 4 min | Pendiente |
| A02 | Seleccionado M | Invariancia cenital | Geometría | Igual que A01 | Holdout / oficiales+clase | 3 | 4 min | Pendiente |
| A03 | Seleccionado M | Robustez óptica | Fotometría | Igual que A01 | Holdout / oficiales+clase | 3 | 4 min | Pendiente |
| F01 | Seleccionado M | Adaptación completa | Full | Modelo, 640, receta, split | Holdout / oficiales+curvas | Opcional | 5 min | Desactivado |
| F02 | Seleccionado M | Regularización | Backbone congelado | Igual que F01 | Holdout / oficiales+curvas | Opcional | 5 min | Desactivado |
| F03 | Seleccionado M | Transferencia gradual | Unfreeze desde época 2 | Igual que F01 | Holdout / oficiales+curvas | Opcional | 5 min | Desactivado |
| H01 | Seleccionado M | LR candidato | 0.001 | AdamW, receta, 640, split | Holdout / oficiales+curvas | 4 | 5 min | Pendiente |
| H02 | Seleccionado M | Paso menor | 0.0001 | Igual que H01 | Holdout / oficiales+curvas | 4 | 5 min | Pendiente |
| H03 | Seleccionado M | Referencia optimización | AdamW | LR 0.001, receta, split | Holdout / oficiales+curvas | Opcional | 5 min | Desactivado |
| H04 | Seleccionado M | Dinámica alternativa | SGD momentum 0.9 | Igual que H03 | Holdout / oficiales+curvas | Opcional | 5 min | Desactivado |
| H05 | Seleccionado M | Regularización referencia | WD 0.0005 | Receta baseline, split | Holdout / oficiales+curvas | Opcional | 5 min | Desactivado |
| H06 | Seleccionado M | Mayor decaimiento | WD 0.001 | Igual que H05 | Holdout / oficiales+curvas | Opcional | 5 min | Desactivado |
| H07 | Seleccionado M | Regularización por datos | Geometría | Igual que H05 excepto augmentation | Holdout / oficiales+curvas | Opcional | 5 min | Desactivado |
| H08 | Seleccionado M | Parada anticipada | Patience 1 | Igual que H05 | Holdout / curvas, fuera de ranking estricto si para antes | Opcional | 5 min | Desactivado |
| H09 | Seleccionado M | Capacidad efectiva | Freeze parcial | Igual que H05 excepto freeze | Holdout / oficiales+curvas | Opcional | 5 min | Desactivado |
| B01 | Seleccionado M | Exposición original | Factor 1 | Modelo, receta, IDs | Holdout / oficiales+work_vessel | Opcional | 5 min | Desactivado |
| B02 | Seleccionado M | Cobertura minoritaria | Factor 2 | Igual que B01; updates cambian por diseño | Holdout / oficiales+work_vessel | Opcional | 5 min | Desactivado |
| O01 | YOLO26s | Capacidad extra | Modelo, batch 2 | Split, métricas, presupuesto | Holdout / cohorte separada | Opcional | 10 min | Desactivado |
| O02 | FRCNN MobileNet | Alternativa dos etapas | Modelo, batch 1 | Split, métricas, presupuesto | Holdout / cohorte separada | Opcional | 10 min | Desactivado |
| C01 | Seleccionado M | Estabilidad entre grupos | Dos folds | Grupos verificados, receta | CV agrupada / oficiales por fold | Opcional | 10 min total | Desactivado |
| I01 | Checkpoint fijo | Punto de operación | Umbral | Predicciones internas cacheadas | Holdout / F1 y score oficial | 5 | 5 min | Desactivado |
| I02 | Checkpoint fijo | Suprimir redundancia | NMS IoU 0.45 vs 0.6 | Pesos y datos internos | Holdout / oficiales+FP/FN | Opcional | 5 min | Desactivado |
| I03 | Checkpoint fijo | Preservar objetos pequeños | Tiles 640, overlap .25 | Pesos y datos internos | Holdout / oficiales+tamaño | Opcional | 5 min | Desactivado |
| FIN | Seleccionado M | Finalista, si cabe | Train interno completo, 6 épocas | Val interna y protocolo | Holdout / fase separada | Opcional | 10 min | Desactivado |

Los archivos `config_low.yaml`, `config_sgd.yaml` y `config_*.yaml` de regularización
son alternativas seleccionables desde CONFIG_FILE, no un grid cartesiano. Las pruebas
de resolución y augmentation parten por defecto del ganador **de arquitectura** a 640;
no heredan silenciosamente otros cambios. Registrar cualquier receta combinada como
nueva configuración, no atribuir sus efectos a una sola variable.

## Medición, evaluación y diagnóstico

La calibración real mide warm-up, batches de entrenamiento, lectura, pico de VRAM
y una muestra de inferencia, además de un batch de estrés con las imágenes más
densas del train interno. Lectura de imagen, verificación de hashes y preparación
se cronometran por separado; los tiempos de workers se agregan y no equivalen al
tiempo de pared de I/O con prefetch. Proyecta épocas a partir de esos batches, identificando
claramente el carácter estimado. La admisión de un entrenamiento exige calibración
para modelo/resolución/batch/AMP, augmentation, freeze y optimizador, y la misma GPU.
Un cambio de LR o weight decay mantiene la medición de su misma receta operativa;
la proyección no se presenta como un nuevo benchmark medido. Después de cada época se contrasta
el coste real de entrenamiento+val con el tiempo restante.

Cada época completa calcula pérdidas de train y val, métricas oficiales y clases.
El mejor checkpoint maximiza `0.7*mAP50:95 + 0.3*F1macro@0.5` sobre val interna.
El evaluador y el formateador del kit no se modifican. Se guardan y releen TXT
redondeados, confianza mínima 0.001, máximo 300 cajas. La recuperación OOM revierte
época/optimizador/RNG y reduce batch; se registra y se excluye del ranking estricto.
OOM en batch 1 interrumpe; no cambia resolución ni descarta imágenes densas.

`validate.ipynb` carga best.pt y el protocolo sin requerir que train.ipynb siga abierto.
Genera métricas, tiempos y VRAM; PR por clase; confusión diagnóstica; ejemplos; CSV
de FP/FN/errores de clase; curvas de pérdida/métricas; y slices por densidad y área.
La confusión usa matching geométrico sin clase y se etiqueta como diagnóstico.
Los slices de tamaño son adaptaciones descriptivas por área nativa, con ignorados
fuera de rango, **no COCOeval oficial**. El score de competencia siempre es el del kit.
Algunas pérdidas de validación en Torchvision requieren modo train con BN/dropout
en eval; el muestreo de propuestas sigue siendo estocástico. No comparar sin más
su magnitud con YOLO ni interpretar el gap train/val como sobreajuste universal.

La comparación final muestra resultados incompletos y separa cohortes por identidad
del protocolo, receta, fase y batch. Solo una cohorte compatible se ordena por score,
con mAP50 para desempatar. No selecciona por test. Con una semilla no se puede
afirmar significancia: confirmar mejoras pequeñas requerirá futuras repeticiones.

## Presupuesto persistente de 90 minutos

Preparar entorno y datos **antes** de iniciar la sesión. La primera acción presupuestada
crea budget.json con reloj absoluto de 5.400 s. Ese inicio no se reinicia entre notebooks:
**el tiempo ocioso entre ellos también cuenta**. Mantener el mismo STORAGE de Drive.
Cambiar de carpeta constituye otra sesión y no debe usarse para eludir el límite del reto.

| Actividad | Techo preliminar de planificación |
|---|---:|
| Calibración acumulada | 5 min |
| YOLO26n | 18 min (config común más conservadora: 15 min) |
| YOLO11n | 15 min |
| Resolución | 20 min |
| Augmentation | 12 min |
| LR | 10 min |
| Final + seguridad | 10 min (8 + 2) |

La tabla suma 90 min, pero no garantiza terminar todas las actividades. La creación
de splits, las validaciones independientes y cualquier otro trabajo dentro de la
sesión también consumen el reloj y reducen lo disponible para esa tabla. La calibración
acumulada está limitada a 300 s. Quedan reservados 480 s para test+ZIP y 120 s de margen.
Si se elige FIN, reducir explícitamente al menos 10 min de resolución/augmentation/LR;
no agregarlo encima de la planificación. La prioridad es completar M01 útil antes
de abrir variantes. Se puede reutilizar un resultado de referencia idéntico, evitando
reentrenarlo solo por estar en otra carpeta.

`time_budget.py` supervisa workers secuenciales con timeout de proceso, deadline
cooperativo entre batches/imágenes y guardado atómico. Un worker no cooperativo
se termina antes del límite asignado. Hay un lock de presupuesto y otro de GPU;
un worker huérfano impide iniciar otro sobre la misma T4. La terminación del proceso
y operaciones de CUDA/Drive no ofrecen latencia nula: el margen de 120 s absorbe
cleanup normal, pero una caída del host o bloqueo de almacenamiento no es controlable
desde Python. Revisar estas garantías en Colab; no se afirma verificación práctica.

## Orden de ejecución futuro

1. `00_prepare_data.ipynb`: montar Drive y seleccionar acceso directo/ZIP; copiar a
   `/content` si se desea. Fuente del usuario:
   [Drive](https://drive.google.com/drive/folders/1vgxmmpi5zclSXRo_y2GwNIkx35QFqr5t).
2. Preparar dependencias CUDA compatibles con requirements.txt; todos los notebooks
   incluyen RUN_SETUP=False. Evitar sustituir el par torch/torchvision de Colab sin motivo.
3. `00_create_splits.ipynb`: activar una vez RUN_SPLITS. Empieza el presupuesto común.
4. `01_t4_benchmark.ipynb`: activar RUN_BENCHMARK para las arquitecturas iniciales.
5. M01 train y validate. Revisar estado y presupuesto. M02 si su medición cabe.
6. Comparación de arquitectura; fijar SELECTED_RUN. Calibrar nueva resolución si cabe.
7. R/A/H según margen real; no ejecutar la matriz completa automáticamente.
8. Opcionales solo con ENABLE_OPTIONAL=True, calibración disponible y tiempo suficiente.
9. compare_all: elegir una cohorte; inspeccionar métricas por clase y curvas.
10. Congelar selección y receta con RUN_FREEZE, luego RUN_EVALUATION para test una vez
    y RUN_SUBMISSION para oculto/ZIP; ambos consumen reserva final. No ajustar tras test.

## Archivos, notebooks y persistencia

Cada carpeta de entrenamiento tiene config.yaml, train.ipynb y validate.ipynb.
Los notebooks son clientes del código compartido; no contienen implementaciones
copiadas de los entrenadores. Todas las acciones costosas empiezan desactivadas.
Las carpetas de inferencia tienen experiment.ipynb y validate.ipynb; no entrenan.
Los comparison.ipynb solo leen resultados reales y no inventan tablas ni gráficas
si no existen. La lista navegable completa está en [experiments/README.md](experiments/README.md).

Cada job recibe un UUID y un directorio propio bajo STORAGE/jobs. Train guarda
checkpoints best/last, historial, política de freeze, sampling, runtime y manifiesto.
Validate guarda por separado metrics, figures, predictions y su procedencia, y
asocia al train su primera validación para el registro. Un checkpoint interrumpido
solo representa la última época completa. No se reanudan parcialmente los batches
de una época, ni se sobrescriben ejecuciones anteriores al reanudar en otro job.

## Verificaciones pendientes

- Importación, sintaxis ejecutable e integración real de todos los notebooks/módulos.
- API interna de DetectionModel/loss/NMS en la versión instalada de Ultralytics,
  especialmente salida end-to-end y pérdidas YOLO26. No se asume que NMS IoU opere
  sobre salida end-to-end; el adaptador rechaza esa ablación. No se usa augment=True.
- AMP FP16, gradientes, carga de pesos, estado del optimizador y congelamiento real.
- Equivalencia de rotaciones/cajas, coordenadas de tiles y manejo de solapamientos.
- Recuperación OOM, interrupción, reanudación, locks, persistencia y watchdog en Colab.
- Fiabilidad de grupos, balance logrado, nombres/ruta del acceso directo de Drive
  y tiempos reales de I/O. No se accedió a la carpeta compartida durante esta etapa.
- Par compatible torch/torchvision, dependencias y disponibilidad de pesos públicos.
- Admisión de experimentos, gastos de validación completa/diagnósticos/guardado y
  margen para producir test y ZIP en el tiempo restante. Si no cabe, se omite/interrumpe.

No hay métricas, tiempos o consumos experimentales nuevos que permitan declarar
funcionalmente verificado este sistema. La primera ejecución debe ser calibración
conservadora y revisión de logs antes de emprender variantes.

Si el batch de estrés falla por OOM durante calibración, no hay una medición válida
para esa configuración: reducir explícitamente el batch en su configuración y
calibrar de nuevo si queda margen de los 300 s. No usar una proyección fallida ni
suponer que la misma configuración se entrenará sin errores. Los coeficientes de
pérdida YOLO (`loss_box`, `loss_cls`, `loss_dfl`) también están declarados en global.yaml;
no son recomendaciones de los libros ni valores óptimos medidos en SpaceLAB.

## Referencias

- Mohamed Elgendy, *Deep Learning for Vision Systems*, Manning. El pedido menciona
  2019; la ficha editorial de la edición publicada indica 2020. Caps. 4, 6 y 7:
  proyectos, métricas, preparación, optimización, transferencia y detección.
  [Editorial](https://www.manning.com/books/deep-learning-for-vision-systems).
- Valliappa Lakshmanan, Martin Görner y Ryan Gillard (2021), *Practical Machine
  Learning for Computer Vision*, O'Reilly. Caps. 3–8: modelos, detección, datasets,
  preprocesamiento, training pipeline y evaluación.
  [Índice editorial](https://www.oreilly.com/library/view/practical-machine-learning/9781098102357/).
- [Reto oficial](kit_participantes/reto_spacelab.pdf), evaluador y formateador del kit.
- Referencias técnicas primarias consultadas por lectura:
  [DetectionModel](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py),
  [pérdidas](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py),
  [NMS](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/nms.py),
  [Faster R-CNN MobileNet](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html).
  Consultar documentación no equivale a verificar la ejecución. Las referencias
  públicas de los libros permiten verificar su estructura; no se afirma haber
  revisado íntegramente las ediciones de pago ni se atribuyen recetas YOLO a los autores.
