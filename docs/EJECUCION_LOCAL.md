# Ejecución en la GPU local

La ejecución local usa las imágenes existentes en `spacelab_dataset`, sin copiar
desde Drive. `scripts/run_local_pipeline.py` busca la raíz anidada del dataset.
Se requiere PyTorch con CUDA; se comprueba la GPU antes de lanzar trabajos.

```powershell
.\.venv\Scripts\python.exe -u scripts/run_local_pipeline.py --storage reports/experiments/local_gpu
```

Por solicitud del equipo, esta modalidad no aplica el límite de 90 minutos.
Los notebooks de Colab conservan el presupuesto original. No mezclar carpetas
de resultados de estas dos modalidades.

La matriz se ejecuta secuencialmente con batch 1, workers 0 y AMP desactivado
(FP32) para la GTX 1650. La primera calibración de YOLO11n con precisión mixta
produjo pérdidas no finitas; FP32 se aplica a todos los candidatos para mantener
la misma receta de precisión en las comparaciones. Los intentos anteriores y
sus checkpoints se conservan en los directorios originales.

La GPU dispone de 4 GB. Se mantienen las resoluciones y semillas declaradas. Estas
mediciones pertenecen a la GPU local y no representan rendimiento en T4.
Cada configuración se calibra antes de entrenar; una configuración que falla
queda registrada y no se declara completada. La validación cruzada requiere
grupos verificados; no se inventan grupos si no existen.

Se incluyen las variantes opcionales del catálogo. La selección usa únicamente
validación interna; después se congela el modelo y la receta de inferencia para
la evaluación única de test y la generación del ZIP del oculto.

Dentro de la carpeta de sesión, `pipeline_state.json` registra los trabajos,
`current_action.json` indica la última acción iniciada y `jobs/*/worker.log`
muestra los batches, épocas e inferencias. `comparison.csv` y
`pipeline_completed.json` se escriben al finalizar la matriz.

Para continuar una sesión, usar el mismo `--storage`: no se repiten acciones
completadas. Revisar las acciones finales fallidas antes de reintentarlas; sus
marcadores protegen contra el uso repetido del test para seleccionar modelos.
