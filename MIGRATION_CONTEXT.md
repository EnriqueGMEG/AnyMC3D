# Contexto de migración — AnyMC3D

Última actualización: 2026-08-04 (Europe/Madrid).

## Snapshot Git

- Repositorio: https://github.com/EnriqueGMEG/AnyMC3D.git.
- Rama: main.
- Commit base: f72972e (origin/main).
- El commit de migración añade la adaptación PMPD/DINOv3, pruebas, configuraciones y documentación.
- No se ha hecho push. Checkpoints, datos preprocesados, manifiestos clínicos, outputs y entornos Python permanecen fuera de Git.

## Objetivo

Adaptar AnyMC3D a clasificación binaria de metástasis por paciente en CT abdominal portal-venoso. La máscara pancreática solo define un bounding box rectangular: no se multiplica por el CT, no se añade como canal y no llega al modelo.

~~~text
CT + máscara pancreática
  -> NIfTI canónico RAS y validación estricta
  -> ROI pancreática o tumoral + margen físico
  -> resampling a spacing común
  -> ventana HU [-150, 250] y escalado [0, 1]
  -> padding HxW sin resize; número de slices variable
  -> DINOv3 ViT-B/16 o ViT-L/16 congelado + LoRA
  -> CLS por slice real
  -> query pooling enmascarado
  -> un logit binario por paciente
~~~

## Implementación añadida

- Preprocesamiento reproducible en preprocessing/: orientación, alineamiento CT/máscara, resampling físico, ROI, padding, auditoría y contrato de inferencia.
- Construcción estricta de manifiestos PMPD_v2 y DPCG externo.
- Dataset y collate para volúmenes con número variable de slices.
- Backbones DINOv3 de Hugging Face con LoRA explícita sobre proyección de patches y atención; el código falla si falta un módulo objetivo.
- LightningModule binario, métricas clínicas, class weighting, early stopping, checkpoints por PR-AUC/AUROC y exportación fold/OOF.
- Entrenamiento CV en dos GPU, inferencia, ensemble y validación externa sin reajustar umbral sobre test.
- Pruebas unitarias y sintéticas de geometría, manifiestos, configuración, entrenamiento y validación externa.

IMPLEMENTATION_PLAN.md, README_PANCREAS_METASTASIS.md y PMPD_V2_TRAINING.md contienen las decisiones metodológicas y comandos completos.

## Dataset activo

Raíz usada: /local/radiomics/PMPD_v2_data.

El entrenamiento usa estrictamente los 479 pacientes de 5fold_cv_stratified_by_center_location_metastasis.csv:

| Cohorte portal-venosa | N |
|---|---:|
| Pangen_OG (PANGENE_OG en el CSV) | 111 |
| ZZU | 224 |
| RUM | 144 |

La variable objetivo es metastasis3 (No=0, Yes=1). Se conservaron los folds suministrados: 95/96/96/96/96 pacientes; 291 negativos y 188 positivos. No se usaron directorios arteriales. DPCG se preparó como cohorte externa sin inventar folds. Navarra y Hospital Universitario Miguel Servet quedaron reservados para validación externa futura.

Los manifiestos contienen IDs pseudonimizados, labels y rutas locales y están ignorados por Git. En la nueva máquina deben transferirse por un canal seguro o reconstruirse:

~~~bash
python prepare_pmpd_v2_manifest.py \
  --data-root /RUTA_NUEVA/PMPD_v2_data \
  --output data/pmpd_v2_manifest.csv
~~~

## Contratos geométricos

Todos usan spacing objetivo (x,y,z)=(0.782,0.782,2.0) mm, canvas múltiplo de 16, overflow_policy=error y 479 casos.

| Variante | ROI/margen | Canvas HxW | Slices min/mediana/p95/max |
|---|---|---:|---:|
| pmpd_v2 | todo mask>0, 4 mm | 160x240 | 21/42/53/65 |
| pmpd_v2_margin12 | todo mask>0, 12 mm | 192x256 | 21/44/57/73 |
| pmpd_v2_tumor_margin6 | label 2, componente mayor, fallback a páncreas, 6 mm | 144x224 | 9/20/37/62 |
| pmpd_v2_full_volume | CT completo, sin máscara, 0 mm | 640x640 | 21/45/62/77 |
| pmpd_v2_body | bbox del cuerpo, sin máscara, 0 mm | 544x576 | 21/45/62/77 |

Las dos últimas no usan segmentación (`mask_required: false`) y leen el CT ya
ventaneado por la fuente en lugar de reaplicar una ventana HU.

`pmpd_v2_body` recorta el aire alrededor del paciente con `roi.mode: body`. El
fondo está exactamente a 0 en estos volúmenes preventaneados, así que el umbral
`>0` da un bbox exacto y no hace falta margen. Cuesta 1224 patches por slice
frente a los 1600 del full-volume, un 1.31x menos de cómputo, y ningún paciente
queda recortado. Bajar más el canvas ya exige cortar anatomía: 496x576 da 1.43x
a costa de 5 pacientes (peor caso 16 mm por lado) y 400x512 da 2.00x recortando
al 59% de la cohorte.

Los contratos JSON reproducibles están en artifacts/. Los CSV por paciente se excluyen porque contienen IDs/rutas y se pueden regenerar.

## Calidad de datos observada en PMPD_v2

Dos hallazgos al auditar la cohorte completa para las variantes sin máscara.
Ambos vienen del origen, no del preprocesado.

**Slice constante final.** 10 casos de 479 (2.1%) terminan en un slice de valor
uniforme (0 o 50) sin anatomía: zzu_004, zzu_010, zzu_034, zzu_038, zzu_064,
zzu_114, zzu_180, RUM_044, PANCREAS_512 y PANCREAS_571. Siempre exactamente uno
y siempre el último del stack.

Importa porque ese slice produce su propio embedding CLS y entra en el query
pooling del paciente. En `pmpd_v2_full_volume` entran los 10 por construcción;
en la variante de crop pancreático ya se colaron 4, porque el bbox z de la
máscara llegaba al final del stack. `roi.mode: body` los recorta y falla en voz
alta si alguno apareciera en el interior del stack en vez de en los extremos.

Hay que detectarlos en el dominio origen, antes del resampling: el resampling
in-plane interpola el slice contra el valor de relleno y deja de ser exactamente
constante, lo que lo esconde de cualquier comprobación posterior.

**Un caso sin clampar.** PANGENE_OG:PANCREAS_516 lleva la misma función de
transferencia que el resto pero sin el clip final, con rango [-51, 306] sobre el
3.0% de sus vóxeles; los otros 478 están exactamente en [0, 255]. Su p01/p99
(-3/250) coincide con la cohorte, así que se satura de vuelta a [0, 255] con
`out_of_range_policy: clip`. Los presupuestos `max_out_of_range_fraction` y
`max_out_of_range_magnitude` siguen rechazando un volumen mal escalado de
verdad, por ejemplo HU crudo. Queda registrado por caso y en
`preprocessing_manifest.json`.

## Coste de entrenamiento medido

Medido en una RTX 5000 Ada (32 GB) sobre el canvas 640x640 del full-volume, con
DINOv3 ViT-B/16 y LoRA.

| slice_chunk_size | grad. checkpointing | Pico GPU | s/paciente |
|---:|:--:|---:|---:|
| 1 | sí | 4.97 GiB | 2.19 |
| 8 | sí | 4.66 GiB | 1.36 |
| 8 | no | 28.63 GiB | 0.98 |

`slice_chunk_size: 8` da 1.6x sin coste de memoria y es lo que usan los configs
de ambas variantes sin máscara. El checkpointing se queda activado: sin él las
activaciones de todos los slices se retienen para el pooling, y el peor caso de
77 slices se queda sin memoria (verificado, no estimado).

La GPU 1 aloja un servicio ajeno de larga duración, así que los scripts
`train_pmpd_v2_*_one_gpu.sh` recorren los cinco folds secuencialmente en la GPU
0. Con early stopping parando entre las épocas 21 y 25, como en las corridas
anteriores, un CV completo ronda las 22 h en full-volume y unas 17 h en body.

## Experimentos

### Baseline PMPD_v2, cinco folds completos

Salida: outputs/pmpd_v2_cv/.

- Media por fold: AUROC 0.6043 ± 0.0282; PR-AUC 0.5430 ± 0.0260.
- OOF global: AUROC 0.5814; PR-AUC 0.4882; Brier 0.2527.
- El umbral 0.5 fue inestable y predijo solo negativos en folds 2 y 5. Deben priorizarse AUROC, PR-AUC y Brier.

### ROI tumoral con fallback y margen 6 mm, cinco folds completos

Salida: outputs/pmpd_v2_vitb_tumor_margin6_cv/.

- Media por fold: AUROC 0.6756 ± 0.0644; PR-AUC 0.6255 ± 0.0746.
- OOF global: AUROC 0.6510; PR-AUC 0.5562; Brier 0.2519.
- Es el mejor agregado completo disponible, pero usa ROI tumoral cuando existe y representa un escenario distinto al bbox pancreático puro.

### Experimentos parciales

- pmpd_v2_vitb_regularized_margin12_full_cv: folds 1–4 terminados; fold 5 pendiente.
- Variantes BCE balanceada/PR-AUC y margin12 inicial: dos folds en varias corridas de desarrollo.
- ViT-L y smoke tests: parciales; no son resultados finales.
- Los directorios exactos están en outputs/ y los pesos en checkpoints/.

## Snapshot de métricas incluido en Git

`experiment_results/` conserva una copia ligera de las métricas agregadas y por fold, resúmenes CV, curvas de entrenamiento limpias y resultados externos disponibles el 4 de agosto de 2026. Incluye tanto experimentos completos como parciales para poder comparar futuras corridas después de la migración.

No contiene checkpoints, pesos, predicciones por paciente, atención por slice ni datos clínicos. Su inventario y límites están descritos en `experiment_results/README.md`.

Por tanto, un clone recupera las métricas necesarias para comparar modelos. Solo hace falta transferir `outputs/` aparte si se necesitan predicciones a nivel paciente, figuras u otros artefactos no agregados.

## Archivos fuera de Git

| Ruta | Tamaño aprox. | Acción |
|---|---:|---|
| checkpoints/ | 17 GB | Transferir aparte para reanudar/evaluar |
| data/ | 1.8 GB | Transferir preprocesados o regenerarlos |
| outputs/ | 11 MB | Opcional; transferir para predicciones/OOF a nivel paciente y figuras |
| artifacts/ | 1.2 MB | JSON versionados; CSV clínicos aparte |
| training_logs/ | 544 KB | Opcional, ignorado por Git |
| .venv/ | 437 MB | No transferir; recrear |

Un git clone recupera código y métricas seleccionadas, pero no checkpoints, datos ni resultados a nivel paciente. Antes de retirar la máquina antigua, copiar las rutas necesarias y verificar tamaños/checksums.

## Entorno registrado

- Python 3.10.19
- PyTorch 2.5.1
- Lightning 2.6.5
- Transformers 5.5.4
- PEFT 0.19.1
- Hydra Core 1.3.2
- nibabel 5.4.2
- NumPy 2.2.6

El proyecto usa uv.lock:

~~~bash
cd /RUTA_NUEVA/AnyMC3D
uv sync --extra dev
source .venv/bin/activate
pytest -q
~~~

DINOv3 es gated en Hugging Face. Hay que aceptar la licencia y ejecutar hf auth login en la nueva máquina; no copiar tokens dentro del repositorio.

## Reanudación

1. Restaurar el repo y comprobar el commit de migración con git log -1.
2. Recrear el entorno y ejecutar pytest -q.
3. Montar/copiar PMPD_v2_data y regenerar manifiestos con rutas nuevas.
4. Transferir o regenerar los NPZ del contrato geométrico elegido.
5. Usar experiment_results/ para comparaciones; transferir checkpoints/ y, si hace falta análisis por paciente, outputs/.
6. Completar fold 5 de pmpd_v2_vitb_regularized_margin12_full_cv antes de agregar resultados.
7. Validar un smoke test en una GPU y luego lanzar CV con los scripts de dos GPU.

## Comandos de referencia

~~~bash
python analyze_dataset_geometry.py \
  --manifest data/pmpd_v2_manifest.csv \
  --config configs/preprocessing/preserve_physical_size.yaml \
  --output-json artifacts/pmpd_v2_geometry_config.json \
  --cases-csv artifacts/pmpd_v2_geometry_audit_cases.csv \
  --overwrite

python preprocess_dataset.py \
  --manifest data/pmpd_v2_manifest.csv \
  --geometry-config artifacts/pmpd_v2_geometry_config.json \
  --config configs/preprocessing/preserve_physical_size.yaml \
  --output-dir data/pmpd_v2_preprocessed \
  --overwrite

./train_pmpd_v2_two_gpus.sh
~~~

Variante body, que no necesita máscara y recorta el fondo:

~~~bash
python analyze_dataset_geometry.py \
  --manifest data/pmpd_v2_manifest.csv \
  --config configs/preprocessing/body_prewindowed_0_255.yaml \
  --output-json artifacts/pmpd_v2_body_geometry_config.json \
  --cases-csv artifacts/pmpd_v2_body_geometry_audit_cases.csv \
  --overwrite

python preprocess_dataset.py \
  --manifest data/pmpd_v2_manifest.csv \
  --geometry-config artifacts/pmpd_v2_body_geometry_config.json \
  --config configs/preprocessing/body_prewindowed_0_255.yaml \
  --output-dir data/pmpd_v2_body_preprocessed

./train_pmpd_v2_body_one_gpu.sh
~~~

Antes de reanudar una variante, revisar PMPD_V2_TRAINING.md y los scripts train_pmpd_v2*_two_gpus.sh para confirmar nombres de salida y configuración.
