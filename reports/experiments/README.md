No hay resultados experimentales nuevos en esta entrega.

Los notebooks escribirán resultados reales en STORAGE/jobs/<acción-uuid>/.
Para persistencia entre sesiones, configurar STORAGE en Google Drive. Mantener
el mismo STORAGE en todos los notebooks del presupuesto de 90 minutos.

El directorio de cada job contiene worker.log, request.json y result.json.
Cada entrenamiento añade checkpoints/, metrics/, figures/, predictions/ y
experiment_manifest.json. Las validaciones conservan sus propios artefactos y
apuntan al entrenamiento de origen. Los artefactos de ejecución están ignorados.
