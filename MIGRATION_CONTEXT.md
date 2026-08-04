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

Los contratos JSON reproducibles están en artifacts/. Los CSV por paciente se excluyen porque contienen IDs/rutas y se pueden regenerar.

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

## Archivos fuera de Git

| Ruta | Tamaño aprox. | Acción |
|---|---:|---|
| checkpoints/ | 17 GB | Transferir aparte para reanudar/evaluar |
| data/ | 1.8 GB | Transferir preprocesados o regenerarlos |
| outputs/ | 11 MB | Transferir aparte; contiene predicciones y OOF |
| artifacts/ | 1.2 MB | JSON versionados; CSV clínicos aparte |
| training_logs/ | 544 KB | Opcional, ignorado por Git |
| .venv/ | 437 MB | No transferir; recrear |

Un git clone no recupera checkpoints ni resultados. Antes de retirar la máquina antigua, copiar las rutas necesarias y verificar tamaños/checksums.

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
5. Transferir checkpoints/ y outputs/ si se quiere conservar el estado experimental.
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

Antes de reanudar una variante, revisar PMPD_V2_TRAINING.md y los scripts train_pmpd_v2*_two_gpus.sh para confirmar nombres de salida y configuración.
