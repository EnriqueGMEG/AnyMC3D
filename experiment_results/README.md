# Snapshot de resultados experimentales

Snapshot creado el 4 de agosto de 2026 a partir de `outputs/`, que está ignorado por Git. La estructura bajo este directorio reproduce las rutas relativas originales sin copiar los artefactos pesados o identificables.

## Contenido

- 38 ficheros de resultados, aproximadamente 90 KB de contenido.
- `metrics.json` de cada fold disponible, incluidos experimentos parciales y smoke tests.
- `cv_summary.json` y `fold_metrics.csv` de las dos validaciones cruzadas completas.
- Curvas de entrenamiento limpias y sus resúmenes CSV.
- Métricas agregadas de la evaluación externa DPCG.

Los resultados principales interpretados están resumidos en `../MIGRATION_CONTEXT.md`. La variante `pmpd_v2_vitb_regularized_margin12_full_cv` solo tiene cuatro folds completos en este snapshot y no debe compararse como un CV de cinco folds terminado.

## Exclusiones deliberadas

No se copiaron:

- checkpoints ni pesos;
- predicciones OOF o externas a nivel paciente;
- atención, geometría o metadatos a nivel paciente/slice;
- imágenes, figuras o datos clínicos.

Estas exclusiones mantienen el snapshot pequeño y evitan versionar identificadores. Si se necesitan análisis de errores por paciente o reanudar entrenamiento, hay que transferir por separado `outputs/` y/o `checkpoints/` desde la workstation original.

## Uso tras la migración

Conservar este árbol sin editar como referencia histórica. Para comparar un experimento nuevo, generar sus métricas en un directorio nuevo y contrastar AUROC, PR-AUC, Brier y variabilidad por fold con los JSON/CSV correspondientes. No mezclar resultados parciales o smoke con las dos corridas completas.
