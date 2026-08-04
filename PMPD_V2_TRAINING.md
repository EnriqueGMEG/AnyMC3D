# Entrenamiento con PMPD_v2

## Cohorte de entrenamiento

La fuente es `/local/radiomics/PMPD_v2_data`. El entrenamiento se limita
estrictamente a los 479 pacientes presentes en
`5fold_cv_stratified_by_center_location_metastasis.csv`:

| Cohorte | Pacientes |
|---|---:|
| Pangen_OG (`PANGENE_OG` en el CSV) | 111 |
| ZZU | 224 |
| RUM | 144 |

DPCG dispone de imágenes y máscaras, pero no aparece en el CSV de splits y
por eso no se incorpora ni se le asigna un fold nuevo. Navarra y Hospital
Universitario Miguel Servet quedan fuera del entrenamiento y reservados para
la futura validación externa.

Los folds se conservan tal como aparecen en el fichero fuente: `1, 2, 3, 4,
5`. La etiqueta objetivo se lee de `metastasis3`, con `No -> 0` y `Yes -> 1`.

| Fold | Total | Negativos | Positivos |
|---:|---:|---:|---:|
| 1 | 95 | 58 | 37 |
| 2 | 96 | 59 | 37 |
| 3 | 96 | 58 | 38 |
| 4 | 96 | 58 | 38 |
| 5 | 96 | 58 | 38 |

## Bounding box

No se selecciona una etiqueta entera específica. El bounding box se calcula
sobre todos los voxels que cumplen `mask > 0`, y se añade un margen físico de
4 mm en cada eje. La máscara solo se usa para localizar ese rectángulo: no se
multiplica por el CT, no se añade como canal y no se guarda en los artefactos
de entrada al modelo.

## Preparación reproducible

El manifiesto local ya generado es `data/pmpd_v2_manifest.csv` y está
excluido de Git por contener etiquetas clínicas pseudonimizadas. Para
reconstruirlo:

```bash
python prepare_pmpd_v2_manifest.py \
  --data-root /local/radiomics/PMPD_v2_data \
  --output data/pmpd_v2_manifest.csv
```

El script exige una correspondencia uno-a-uno entre cada fila del split, su
CT y su máscara, conserva los folds 1–5 y valida los 479 pares NIfTI.

El contrato geométrico específico ya calculado está en
`artifacts/pmpd_v2_geometry_config.json`:

| Resultado | Valor |
|---|---:|
| Casos | 479 |
| Spacing X/Y/Z | 0.782 / 0.782 / 2.0 mm |
| Canvas H×W | 160×240 |
| Slices mínimo / mediana / p95 / máximo | 21 / 42 / 53 / 65 |
| Política | `dataset_max`, sin resize |

Para recalcularlo de forma explícita:

```bash
python analyze_dataset_geometry.py \
  --manifest data/pmpd_v2_manifest.csv \
  --config configs/preprocessing/preserve_physical_size.yaml \
  --output-json artifacts/pmpd_v2_geometry_config.json \
  --cases-csv artifacts/pmpd_v2_geometry_audit_cases.csv \
  --overwrite
```

Los 479 artefactos preprocesados ya están en
`data/pmpd_v2_preprocessed`. Para reconstruirlos:

```bash
python preprocess_dataset.py \
  --manifest data/pmpd_v2_manifest.csv \
  --geometry-config artifacts/pmpd_v2_geometry_config.json \
  --config configs/preprocessing/preserve_physical_size.yaml \
  --output-dir data/pmpd_v2_preprocessed \
  --overwrite
```

## Entrenamiento

Un solo fold:

```bash
python train.py \
  data=pmpd_v2 \
  model=anymc3d_dinov3_vitb \
  data.fold=1 \
  model.slice_chunk_size=32 \
  model.output_dir=outputs/pmpd_v2_cv \
  model.run_name=anymc3d-dinov3-vitb-pmpd-v2
```

Los cinco folds en dos GPUs, con agregación OOF automática al terminar:

```bash
./train_pmpd_v2_two_gpus.sh
```

Los checkpoints del experimento BCE balanceado quedan en
`checkpoints/anymc3d-dinov3-vitb-pmpd-v2-bce-prauc-es30/fold_N/`. Las
predicciones, atenciones, geometrías y métricas por fold se escriben en
`outputs/pmpd_v2_vitb_bce_prauc_es30_cv/fold_N/`; al terminar los cinco
folds también se generan las predicciones OOF y `cv_summary.json`.

La configuración usa DINOv3 ViT-B/16 oficial, LoRA, batch 4, chunks de 32
slices, BF16 y BCE con `pos_weight=N_negative/N_positive` calculado solo en
train. El early stopping maximiza `val_pr_auc`, tiene paciencia 30 y
`min_delta=0.0`: cualquier mejora estricta reinicia la paciencia. Se conserva
un checkpoint para mejor `val_pr_auc` y otro para mejor `val_auroc`.

La barra dinámica está desactivada. Cada época añade una línea persistente al
log y una fila a `fold_N/epoch_metrics.csv`. Para ver el historial completo de
todos los folds mientras entrena:

```bash
watch -n 5 ./show_training_metrics.sh
```

También se pueden seguir las líneas de época de ambas GPU con:

```bash
tail -f training_logs/pmpd_v2_vitb_bce_prauc_es30_cv/gpu*.log
```
