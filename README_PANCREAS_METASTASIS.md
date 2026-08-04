# AnyMC3D-DINOv3 para metástasis pancreática

> **Dataset activo:** el entrenamiento actual usa PMPD_v2 (479 pacientes y folds 1-5). Consulta [PMPD_V2_TRAINING.md](PMPD_V2_TRAINING.md). La sección histórica de 335 pacientes que aparece más abajo se conserva únicamente como trazabilidad.

Esta extensión entrena un clasificador binario a nivel de paciente a partir
de CT portal-venoso y una máscara fina de páncreas. La máscara solo localiza
un bounding box rectangular; nunca se multiplica por el CT, se concatena como
canal ni llega al modelo.

El flujo implementado es:

```text
CT + máscara pancreática
  -> NIfTI canónico RAS y validación estricta
  -> bbox pancreático + margen físico configurable
  -> resampling a spacing común
  -> ventana HU [-150, 250] y rango [0, 1]
  -> padding HxW sin resize, S variable
  -> DINOv3 ViT + LoRA
  -> CLS por slice real
  -> query pooling enmascarado
  -> un logit por paciente
```

## Estado de los datos suministrados

La auditoría se ejecutó sobre 335 pares con correspondencia exacta entre:

```text
/local/radiomics/PMPD_fix/PMPD_pancreas_train/data_zzu_cnio/01_rawdata/venous/
/local/radiomics/PMPD_fix/PMPD_pancreas_train/data_zzu_cnio/01_rawdata/labels/
```

Todos los pares pasaron shape, orientación, spacing, affine y máscara no
vacía, sin remuestrear máscaras. El contrato persistido es
`artifacts/geometry_config.json`:

| Resultado | Valor |
|---|---:|
| Casos | 335 |
| Spacing objetivo X/Y/Z (mm) | 0.832031 / 0.832031 / 2.0 |
| Canvas `H x W` | 160 x 224 |
| Slices: mínimo / mediana / p95 / máximo | 19 / 42 / 53.3 / 66 |
| Política | `dataset_max`, `overflow_policy=error` |

Los metadatos auxiliares permitieron construir `data/pancreas_metastasis_manifest.csv` para los 335 pacientes. La etiqueta binaria procede de `Meta.csv:metastasis3` y coincide exactamente con `Label` del experimento histórico. El fold procede de las filas `Center=Validation` y `Level=Patient Level` de `yuce_a4.csv`: 67 pacientes en cada uno de los cinco folds. El manifiesto contiene 208 negativos y 127 positivos y está excluido de Git por contener etiquetas clínicas pseudonimizadas.

## Instalación y acceso a DINOv3

```bash
cd /local/elopezl/AnyMC3D
uv sync --extra dev
source .venv/bin/activate
```

Los repositorios oficiales de Meta en Hugging Face son gated. Antes de
entrenar:

1. Aceptar la licencia en la página del checkpoint elegido.
2. Ejecutar `hf auth login`.
3. Comprobar el acceso con `hf auth whoami`.

El código exige Transformers `>=4.56.0` y nunca sustituye silenciosamente un
checkpoint inaccesible por otro modelo.

## Manifiesto

CSV obligatorio, una fila por paciente:

```csv
patient_id,ct_path,pancreas_mask_path,label,fold
PANCREAS_001,/ruta/venous/PANCREAS_001.nii.gz,/ruta/labels/PANCREAS_001.nii.gz,0,0
PANCREAS_002,/ruta/venous/PANCREAS_002.nii.gz,/ruta/labels/PANCREAS_002.nii.gz,1,1
```

Las rutas pueden ser absolutas o relativas al CSV. No se generan folds
nuevos. Se rechazan identificadores duplicados, labels no binarias, folds
vacíos/fraccionarios, ficheros ausentes, máscaras vacías y pares NIfTI
desalineados. La opción conservadora es
`alignment.resample_mask_to_ct: false`; si se activa, la máscara se lleva a la
malla del CT únicamente con nearest-neighbor.

## 1. Auditoría geométrica

Comando exacto ejecutado para los datos suministrados:

```bash
python analyze_dataset_geometry.py \
  --ct-dir /local/radiomics/PMPD_fix/PMPD_pancreas_train/data_zzu_cnio/01_rawdata/venous \
  --mask-dir /local/radiomics/PMPD_fix/PMPD_pancreas_train/data_zzu_cnio/01_rawdata/labels \
  --config configs/preprocessing/preserve_physical_size.yaml \
  --output-json artifacts/geometry_config.json \
  --cases-csv artifacts/geometry_audit_cases.csv \
  --overwrite
```

Para el dataset definitivo se puede usar `--manifest` en vez de las dos
carpetas. La herramienta no lee `label` ni usa el fold para elegir spacing o
canvas. `auto` se resuelve una vez a la mediana X/Y y queda guardado en JSON.

## 2. Preprocesado

```bash
python preprocess_dataset.py \
  --manifest data/pancreas_metastasis_manifest.csv \
  --geometry-config artifacts/geometry_config.json \
  --config configs/preprocessing/preserve_physical_size.yaml \
  --output-dir data/pancreas_metastasis_preprocessed
```

Cada `{patient_id}.npz` contiene exclusivamente:

- `volume`: `S x 1 x H x W`, float32 en `[0, 1]`;
- `slice_positions_mm`: una coordenada física por slice;
- `original_slice_indices`: índices sobre el NIfTI previo a canonicalizar;
- `geometry_json`: trazabilidad geométrica.

No se guarda la máscara en el artefacto. También se generan
`preprocessing_log.jsonl`, `patient_geometry.csv` y
`preprocessing_manifest.json`. Un output existente produce error salvo que
se solicite `--overwrite`.

## 3. Entrenamiento

ViT-B/16, fold 0:

```bash
python train.py \
  data=pancreas_metastasis \
  model=anymc3d_dinov3 \
  data.fold=0 \
  data.module.manifest_path=data/pancreas_metastasis_manifest.csv \
  data.module.preprocessed_root=data/pancreas_metastasis_preprocessed
```

ViT-L/16:

```bash
python train.py \
  data=pancreas_metastasis \
  model=anymc3d_dinov3_vitl \
  data.fold=0
```

Los valores por defecto son LoRA `r=8`, `alpha=16`, focal binaria
`gamma=2`, `alpha=0.25`, batch 2, chunks de 8 slices y checkpoint por
`val_pr_auc`. Para usar BCE:

```bash
python train.py ... model.loss=bce model.pos_weight=2.0
```

El backbone base queda congelado. Transformers expone:

- patch: `embeddings.patch_embeddings` (`Conv2d`);
- atención por bloque:
  `model.layer.N.attention.{q_proj,k_proj,v_proj,o_proj}` (`Linear`).

Q/K/V están separados en Transformers. Los tres adaptadores de un bloque
comparten su matriz LoRA A y tienen matrices B independientes, equivalente a
una actualización LoRA sobre una proyección QKV fusionada. La proyección de
salida y el patch embedding tienen adaptadores propios. B se inicializa a
cero, por lo que el modelo adaptado comienza con el comportamiento del
backbone.

Conteos verificados con la arquitectura oficial:

| Backbone | Total | Entrenables | Porcentaje | Módulos LoRA |
|---|---:|---:|---:|---:|
| DINOv3 ViT-B/16 | 86,116,609 | 456,193 | 0.530% | 49 |
| DINOv3 ViT-L/16 | 304,325,633 | 1,196,033 | 0.393% | 97 |

Los conteos arquitectónicos están en `artifacts/model_validation.json`. La validación con los pesos gated pretrained oficiales está en `artifacts/pretrained_forward_validation.json`. El forward completo sobre `PANCREAS_501`, con `[B,S,C,H,W]=[1,47,1,160,224]`, produjo logits `[1,1]`, atención `[1,47]` con suma 1 y `last_hidden_state=[1,145,768]` por slice: 1 CLS + 4 registers + 140 patches. El CLS es `last_hidden_state[:, 0, :]`.

## 4. Cross-validation

```bash
python train_cv.py --folds 0 1 2 3 4
```

Genera por fold y de forma combinada:

- `predictions.csv` y `oof_predictions.csv`;
- `slice_attention.csv` y `oof_slice_attention.csv`;
- `patient_geometry.csv` y `oof_patient_geometry.csv`;
- `metrics.json`, `fold_metrics.csv` y `cv_summary.json`.

El resumen incluye métricas por fold, media/desviación, métricas OOF
globales, clases, distribución de slices y distribución de atención.

Ejemplo real del smoke test:

```csv
patient_id,fold,label,logit,probability,prediction
patient_0,0,0,0.1233460754,0.5307974815,1
patient_1,0,1,0.1226680428,0.5306286216,1
```

```csv
patient_id,fold,slice_index,original_slice_index,z_position_mm,attention_weight,is_valid_slice
patient_0,0,0,0,0.0,0.125,True
patient_0,0,1,1,2.0,0.125,True
```

```csv
patient_id,physical_crop_dimensions_mm_xyz,resampled_shape_xyz,S,H,W,pancreas_bbox_volume_mm3,real_data_fraction_in_plane,padding_fraction_in_plane,num_slices,z_extent_mm
patient_0,"[14.0, 15.0, 16.0]","[14, 15, 8]",8,32,32,336.0,0.205078125,0.794921875,8,14.0
```

## 5. Inferencia

```bash
python inference.py \
  --ct /ruta/paciente.nii.gz \
  --pancreas-mask /ruta/paciente_mask.nii.gz \
  --checkpoint checkpoints/anymc3d-dinov3-vitb-pancreas-metastasis/fold_0/epoch=000.ckpt \
  --geometry-config artifacts/geometry_config.json \
  --preprocessing-config configs/preprocessing/preserve_physical_size.yaml \
  --patient-id PACIENTE_001 \
  --output-dir outputs/inference/PACIENTE_001 \
  --device cuda
```

Inferencia no recalcula spacing ni canvas: exige el JSON persistido y utiliza
el mismo `preprocess_case` que preparación de datos. Devuelve
`prediction.json`, `slice_attention.csv` y `geometry.json`. Por defecto usa el
threshold guardado en el checkpoint; `--threshold` permite fijarlo de forma
explícita, pero nunca se optimiza con el caso inferido.

## Tests y smoke tests

Suite completa:

```bash
pytest -q
```

Por bloque:

```bash
pytest -q tests/test_preprocessing.py
pytest -q tests/test_model_dinov3.py
pytest -q tests/test_data_training_integration.py
pytest -q tests/test_cross_validation.py
pytest -q tests/test_inference_contract.py
```

`test_synthetic_end_to_end_training_validation_and_exports` construye cuatro
pares NIfTI de tamaños distintos, los preprocesa, hace collate con S variable,
ejecuta una época Lightning y valida los tres CSV. Los tests cubren los 32
criterios de aceptación de forma agrupada.

## Diferencias intencionadas respecto al paper

El paper usa una entrada PDAC fija de `432 x 240 x 70`. Esta implementación:

- no redimensiona cada crop para llenar una shape fija;
- preserva el tamaño anatómico mediante spacing físico común;
- conserva todas las slices tras el crop;
- usa padding constante rectangular en plano y `slice_mask` en batch;
- no añade posición aprendida, LSTM, GRU ni Transformer entre slices;
- no implementa multivista, decoder 3D, segmentación auxiliar, supervisión
  pixel-level, heatmaps ni ensembling.

La auditoría del repositorio original y la separación entre decisiones del
paper y de la reproducción están documentadas en `IMPLEMENTATION_PLAN.md`.
